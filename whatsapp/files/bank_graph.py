from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from random import choice
import json
import re
import copy

from loguru import logger
from ollama import AsyncClient

from db.database import SessionLocal
from rag_service import get_rag_context
from core.agent_registry import get_agent
from utils.helpers import extract_phone_number, safe_parse_json
from agents.bank.agent_tools import get_account_balance, get_user_info, create_ticket
from response import (
    GREETING_RESPONSES,
    BALANCE_MISSING_PHONE,
    BALANCE_SUCCESS,
    USER_INFO_MISSING_PHONE,
    USER_INFO_SUCCESS,
    TICKET_CONFIRM_INTENT,
    TICKET_ASK_ISSUE,
    TICKET_ASK_PHONE,
    TICKET_FINAL,
    HUMAN_HANDOFF,
    THANK_YOU_RESPONSES,
)


# ── State ──────────────────────────────────────────────────────────────────────
class BotState(TypedDict):
    session_id: str
    message:    str
    agent_name: str
    intent:     Optional[str]
    response:   Optional[str]
    memory:     Dict[str, Any]


# ── Utilities ──────────────────────────────────────────────────────────────────
def safe_state(state: BotState) -> BotState:
    return copy.deepcopy(state)


def ensure_memory(state: BotState) -> BotState:
    if "memory" not in state:
        state["memory"] = {}

    memory = state["memory"]
    memory.setdefault("profile", {})
    memory.setdefault("history", [])
    memory.setdefault("ticket",  {})
    memory.setdefault("flow",    {})

    flow = memory["flow"]
    flow.setdefault("active",            None)
    flow.setdefault("step",              None)
    flow.setdefault("expandable",        False)
    flow.setdefault("last_expand_offer", False)

    if not isinstance(state.get("intent"), str):
        state["intent"] = "RAG"

    return state


# ── Intent Prompt ──────────────────────────────────────────────────────────────
INTENT_PROMPT = """
You are a routing system for a banking chatbot.

Return ONLY valid minified JSON. No explanation. No text.

Example:
{{"intent":"BALANCE","confidence":0.95}}

Allowed intents:
- GREETING
- BALANCE
- USER_INFO
- TICKET
- RAG
- EXPAND
- CONTROL
- SMALL_TALK

Routing Rules:

- GREETING: User is saying hello, hi, good morning, or starting conversation.
- SMALL_TALK: Casual non-banking chat, such as thanks, bye, okay, got it.
- CONTROL: Short acknowledgements like yes, no, ok, sure (ONLY when no active topic).
- EXPAND: User wants more explanation of the previous answer.
- BALANCE: Asking about account balance, available funds, or money inquiry.
- USER_INFO: Asking for profile, account details, email, or personal info.
- TICKET: User is PERSONALLY experiencing a problem RIGHT NOW.
  Examples: "my card is blocked", "I can't login", "my transaction failed".
  NOT TICKET if asking an informational question about complaints or processes.
- RAG: Any informational or knowledge-based question about banking services, policies, or procedures.
  ALWAYS use RAG when user asks "what", "how", "tell me", "explain", "what are the requirements".

CRITICAL RULES:
- If the message starts with "what", "how", "tell me", "explain", "can you tell me" → always RAG.
- Only use TICKET if the user is personally reporting an active issue, not asking about a process.

Conversation:
{history}

User:
{message}
"""

ALLOWED_INTENTS = {
    "GREETING", "BALANCE", "USER_INFO",
    "TICKET", "RAG", "EXPAND", "CONTROL", "SMALL_TALK",
}


# ── Nodes ──────────────────────────────────────────────────────────────────────
async def detect_intent(state: BotState) -> BotState:
    state = ensure_memory(state)
    state["intent"] = "RAG"

    try:
        client = AsyncClient()
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in state["memory"]["history"][-6:]
        )
        prompt = INTENT_PROMPT.format(history=history_text, message=state["message"])

        res = await client.generate(
            model="gemma4:e2b",
            prompt=prompt,
            options={"temperature": 0, "top_p": 0.9},
        )

        raw  = res.response.strip()
        logger.info(f"[INTENT RAW] {raw}")
        data = safe_parse_json(raw)

        if data:
            raw_intent      = str(data.get("intent", "RAG")).strip().upper()
            state["intent"] = raw_intent if raw_intent in ALLOWED_INTENTS else "RAG"
            logger.info(f"[INTENT RESOLVED] {state['intent']}")
        else:
            logger.warning(f"[INTENT FALLBACK] raw: {raw}")

    except Exception as e:
        logger.warning(f"[INTENT NODE ERROR] {e}, falling back to RAG")
        state["intent"] = "RAG"

    return state


def router(state: BotState) -> str:
    flow   = state.get("memory", {}).get("flow", {})
    active = flow.get("active")

    if active == "TICKET":    return "TICKET"
    if active == "BALANCE":   return "BALANCE"
    if active == "USER_INFO": return "USER_INFO"

    intent = state.get("intent", "RAG")
    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()
    return intent if intent in ALLOWED_INTENTS else "RAG"


async def control_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    text  = state["message"].lower().strip()
    flow  = state["memory"]["flow"]

    if text in {"no", "nope"}:
        flow["expandable"]        = False
        flow["last_expand_offer"] = False
        state["response"] = "Alright 👍"
        return state

    if text in {"yes", "yeah", "sure", "ok", "okay"}:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            state["intent"] = "EXPAND"
            return await expand_node(state)
        state["response"] = "Okay 👍 How can I help you?"
        return state

    state["response"] = "Alright 👍"
    state["intent"]   = "CONTROL"
    return state


async def ticket_node(state: BotState) -> BotState:
    state  = ensure_memory(state)
    flow   = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]
    step   = flow.get("step") or "confirm"
    text   = state["message"].lower()

    if step == "confirm":
        state["response"]  = choice(TICKET_CONFIRM_INTENT)
        flow["active"] = "TICKET"
        flow["step"]   = "await_confirmation"
        return state

    if step == "await_confirmation":
        if any(x in text for x in ["yes", "create", "ticket", "ok", "sure"]):
            flow["step"]      = "collect_issue"
            state["response"] = choice(TICKET_ASK_ISSUE)
        elif any(x in text for x in ["human", "agent", "representative"]):
            flow["active"] = None
            flow["step"]   = None
            state["response"] = choice(HUMAN_HANDOFF)
        else:
            flow["active"] = None
            flow["step"]   = None
            state["response"] = "Alright, let me know how else I can help you."
        return state

    if step == "collect_issue":
        ticket["issue"] = state["message"]
        flow["step"]    = "collect_phone"
        state["response"] = choice(TICKET_ASK_PHONE)
        return state

    if step == "collect_phone":
        phone = extract_phone_number(state["message"])
        if not phone:
            state["response"] = "Please provide a valid registered phone number."
            return state

        ticket["phone"] = phone

        client = AsyncClient()
        prompt = f"""Convert issue into JSON:
{{
  "category": "card_issue | auth_issue | transaction_issue | account_issue | general",
  "short_description": "max 15 words"
}}

Issue:
{ticket['issue']}"""

        res = await client.generate(model="gemma4:e2b", prompt=prompt)

        try:
            data = json.loads(res.response)
        except Exception:
            match = re.search(r"\{.*\}", res.response, re.DOTALL)
            data  = json.loads(match.group()) if match else {}

        db = SessionLocal()
        try:
            result = create_ticket(
                db,
                phone,
                category          = data.get("category", "general"),
                short_description = data.get("short_description", ticket["issue"][:80]),
            )
        finally:
            db.close()

        ticket_id         = result.get("ticket_id", "UNKNOWN")
        state["response"] = choice(TICKET_FINAL) + f" Ticket ID: {ticket_id}"
        flow["active"]    = None
        flow["step"]      = None
        return state

    state["response"] = "I understand you're facing an issue. Would you like me to raise a support ticket?"
    flow["step"] = None
    return state


async def balance_node(state: BotState) -> BotState:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "BALANCE"
        flow["step"]   = "ask_phone"
        state["response"] = choice(BALANCE_MISSING_PHONE)
        return state

    db = SessionLocal()
    try:
        result = get_account_balance(db, phone)
    finally:
        db.close()

    flow["active"] = None
    flow["step"]   = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(BALANCE_SUCCESS).format(
        balance=result["balance"], currency=result["currency"]
    )
    state["intent"] = "BALANCE"
    return state


async def user_info_node(state: BotState) -> BotState:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "USER_INFO"
        flow["step"]   = "ask_phone"
        state["response"] = choice(USER_INFO_MISSING_PHONE)
        return state

    db = SessionLocal()
    try:
        result = get_user_info(db, phone)
    finally:
        db.close()

    flow["active"] = None
    flow["step"]   = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(USER_INFO_SUCCESS).format(
        name=result["user_name"], email=result["email"]
    )
    state["intent"] = "USER_INFO"
    return state


async def rag_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))

    agent         = get_agent(state["agent_name"])
    vectorstore   = agent.get_vectorstore()
    system_prompt = agent.get_system_prompt()

    if vectorstore is None:
        state["response"] = "Please contact ABB support at 937."
        return state

    docs, _ = get_rag_context(state["message"], vectorstore)
    context = "\n".join(docs) if docs else ""

    if not context.strip():
        state["response"] = "Please contact ABB support at 937."
        return state

    client = AsyncClient()
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in state["memory"]["history"][-8:]
    )

    prompt = f"""Answer the user's question using ONLY the context below.

CRITICAL RULES:
- If the answer contains a list of fields, steps, or requirements — reproduce EVERY item exactly. Do NOT merge, skip, or paraphrase any item.
- Do NOT add information that is not in the context.
- If the context does not contain the answer, say: "Please contact ABB support at 937."

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{state['message']}

ANSWER:"""

    res = await client.generate(
        model="gemma4:e4b",
        prompt=prompt,
        system=system_prompt,
        options={"temperature": 0.2, "top_p": 0.9},
    )

    state["memory"]["flow"]["expandable"]        = True
    state["memory"]["flow"]["last_expand_offer"] = True
    state["response"] = res.response.strip()
    state["intent"]   = "RAG"
    # NOTE: history is appended by server.py uniformly for all nodes
    return state


async def expand_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    agent         = get_agent(state["agent_name"])
    system_prompt = agent.get_system_prompt()

    last_answer = next(
        (m["content"] for m in reversed(state["memory"]["history"]) if m["role"] == "assistant"),
        "",
    )

    client = AsyncClient()
    res    = await client.generate(
        model="gemma4:e4b",
        system=system_prompt,
        prompt=f"""
You are a BANK SUPPORT ASSISTANT.

Expand clearly and in detail.

PREVIOUS ANSWER:
{last_answer}

USER:
{state['message']}

RULE:
- Do NOT use external knowledge
- Only expand given answer

FINAL ANSWER:
""",
    )

    flow["expandable"]        = False
    flow["last_expand_offer"] = False
    state["response"] = res.response.strip()
    state["intent"]   = "EXPAND"
    # NOTE: history is appended by server.py uniformly for all nodes
    return state


async def greeting_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(GREETING_RESPONSES)
    state["intent"]   = "GREETING"
    return state


async def small_talk_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(THANK_YOU_RESPONSES)
    state["intent"]   = "SMALL_TALK"
    return state


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(BotState)

    graph.add_node("detect",    detect_intent)
    graph.add_node("control",   control_node)
    graph.add_node("greeting",  greeting_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("balance",   balance_node)
    graph.add_node("user_info", user_info_node)
    graph.add_node("ticket",    ticket_node)
    graph.add_node("rag",       rag_node)
    graph.add_node("expand",    expand_node)

    graph.add_edge(START, "detect")

    graph.add_conditional_edges(
        "detect",
        router,
        {
            "GREETING":   "greeting",
            "BALANCE":    "balance",
            "USER_INFO":  "user_info",
            "TICKET":     "ticket",
            "SMALL_TALK": "small_talk",
            "CONTROL":    "control",
            "RAG":        "rag",
            "EXPAND":     "expand",
        },
    )

    for node in [
        "greeting", "balance", "user_info", "ticket",
        "rag", "expand", "small_talk", "control",
    ]:
        graph.add_edge(node, END)

    return graph.compile()
