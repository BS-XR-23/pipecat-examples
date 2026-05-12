from typing import TypedDict, Optional, Dict, Any
from random import choice
import json
import os
from dotenv import load_dotenv
import re
import copy
from loguru import logger
from ollama import AsyncClient
from db.database import SessionLocal
from utils.helpers import extract_phone_number, safe_parse_json, extract_json
from agents.bank.agent_tools import get_account_balance, get_user_info, create_ticket
from services.rag_services.rag_service import get_rag_context
from utils.agent_logger import log_tool_call
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
    THANK_YOU_RESPONSES
)


load_dotenv()  # Load environment variables from .env file
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))

# -------------------- STATE --------------------
class BotState(TypedDict):
    session_id: str
    message: str
    agent_name: str
    agent: Any
    intent: Optional[str]
    response: Optional[str]
    memory: Dict[str, Any]


# -------------------- VECTORSTORE INIT --------------------
ollama_client = AsyncClient()

def safe_state(state):
    state = copy.deepcopy(state)
    return state


def ensure_memory(state: BotState):
    if "memory" not in state:
        state["memory"] = {}

    memory = state["memory"]

    memory.setdefault("profile", {})
    memory.setdefault("history", [])
    memory.setdefault("ticket", {})
    memory.setdefault("flow", {})

    flow = memory["flow"]
    flow.setdefault("active", None)
    flow.setdefault("step", None)
    flow.setdefault("expandable", False)
    flow.setdefault("last_expand_offer", False)

    if not isinstance(state.get("intent"), str):
        state["intent"] = "RAG"

    return state


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

- GREETING:
  User is saying hello, hi, good morning, good evening, or starting conversation.

- SMALL_TALK:
  Casual non-banking chat, such as thanks, bye, okay, got it, casual replies, or emotional filler.

- CONTROL:
  Short acknowledgements like yes, no, ok, okay, sure, cancel (ONLY when no active topic).

- EXPAND:
  User wants more explanation of the previous answer.

- BALANCE:
  Asking about account balance, available funds, or money inquiry.

- USER_INFO:
  Asking for profile, account details, email, or personal info.

- TICKET:
  User is PERSONALLY experiencing a problem RIGHT NOW.
  Examples: "my card is blocked", "I can't login", "my transaction failed", "I have an issue".
  NOT TICKET if the user is asking an informational question about complaints or processes.

- RAG:
  Any informational or knowledge-based question about banking services, policies, procedures, or requirements.
  Examples: "what documents do I need", "how do I submit a complaint", "what information is required for X".
  ALWAYS use RAG when the user asks "what", "how", "tell me", "explain", "what are the requirements".

CRITICAL RULES:
- If the message starts with "what", "how", "tell me", "explain", "can you tell me" → always RAG.
- Only use TICKET if the user is personally reporting an active issue, not asking about a process.

Conversation:
{history}

User:
{message}
"""


async def detect_intent(state: BotState):
    state = ensure_memory(state)
    state["intent"] = "RAG"  

    try:
        client = AsyncClient()

        history = state["memory"]["history"][-6:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in history
        )

        prompt = INTENT_PROMPT.format(
            history=history_text,
            message=state["message"]
        )

        res = await client.generate(
            model="gemma4:e2b",
            prompt=prompt,
            options={"temperature": 0, "top_p": 0.9}
        )

        raw = res.response.strip()
        logger.info(f"[INTENT RAW] {raw}")

        data = safe_parse_json(raw)

        if not data:
            logger.warning(f"[INTENT FALLBACK - no data] raw: {raw}")
            state["intent"] = "RAG"
        else:
            allowed = {
                "GREETING", "BALANCE", "USER_INFO",
                "TICKET", "RAG", "EXPAND",
                "CONTROL", "SMALL_TALK"
            }

            raw_intent = str(data.get("intent", "RAG")).strip().upper()
            state["intent"] = raw_intent if raw_intent in allowed else "RAG"
            logger.info(f"[INTENT RESOLVED] {state['intent']}")

    except Exception as e:
        logger.warning(f"[INTENT NODE ERROR] {e}, falling back to RAG")
        state["intent"] = "RAG"

    return state


def router(state: BotState):
    flow = state.get("memory", {}).get("flow", {})
    active = flow.get("active")

    if active == "TICKET":
        return "TICKET"
    if active == "BALANCE":
        return "BALANCE"
    if active == "USER_INFO":
        return "USER_INFO"

    intent = state.get("intent", "RAG")

    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()

    allowed = {
        "GREETING", "BALANCE", "USER_INFO",
        "TICKET", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK"
    }

    return intent if intent in allowed else "RAG"


async def control_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    text = state["message"].lower().strip()
    flow = state["memory"]["flow"]

    if text in ["no", "nope"]:
        flow["expandable"] = False
        flow["last_expand_offer"] = False

        state["response"] = "Alright 👍"
        return state

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            state["intent"] = "EXPAND"
            return await expand_node(state)

        state["response"] = "Okay 👍 How can I help you?"
        return state

    state["response"] = "Alright 👍"
    state.setdefault("intent", "CONTROL")
    return state


async def ticket_node(state: BotState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]

    step = flow.get("step") or "confirm"
    text = state["message"].lower()

    if step == "confirm":
        state["response"] = choice(TICKET_CONFIRM_INTENT)
        flow["active"] = "TICKET"
        flow["step"] = "await_confirmation"
        return state

    if step == "await_confirmation":
        if any(x in text for x in ["yes", "create", "ticket", "ok", "sure"]):
            flow["step"] = "collect_issue"
            state["response"] = choice(TICKET_ASK_ISSUE)
        elif any(x in text for x in ["human", "agent", "representative"]):
            flow["active"] = None
            flow["step"] = None
            state["response"] = choice(HUMAN_HANDOFF)
        else:
            flow["active"] = None
            flow["step"] = None
            state["response"] = "Alright, let me know how else I can help you."
        return state

    if step == "collect_issue":
        ticket["issue"] = state["message"]
        flow["step"] = "collect_phone"
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
            data = json.loads(match.group()) if match else {}

        db = SessionLocal()
        try:
            result = create_ticket(
                db,
                phone,
                category=data.get("category", "general"),
                short_description=data.get("short_description", ticket["issue"][:80])
            )
            tool_status = "failed" if "error" in result else "success"
            short_desc = (
                f"Agent created support ticket for {phone}: {data.get('short_description', '')}"
                if tool_status == "success"
                else f"Agent failed to create ticket for {phone}"
            )
        except Exception as e:
            result = {"error": str(e)}
            tool_status = "error"
            short_desc = f"Agent encountered an error creating ticket for {phone}"
        finally:
            db.close()
        
        await log_tool_call(
            # vendor_id=state["agent"].vendor_id,
            vendor_id=1,
            agent_name=state["agent_name"],
            agent_type="bank",
            tool_name="create_ticket",
            tool_status=tool_status,
            short_description=short_desc,
            user_identifier=phone,
            raw_tool_input={
                "phone": phone,
                "category": data.get("category"),
                "short_description": data.get("short_description")
            },
            raw_tool_output=result,
        )

        ticket_id = result.get("ticket_id", "UNKNOWN")
        if tool_status == "success":
            state["response"] = choice(TICKET_FINAL) + f" Ticket ID: {ticket_id}"
        else:
            state["response"] = "I'm sorry, I couldn't create a support ticket at this time. Please contact ABB support directly at 937."
        flow["active"] = None
        flow["step"] = None
        return state

    state["response"] = "I understand you're facing an issue. Would you like me to raise a support ticket?"
    flow["step"] = None
    return state
                

async def balance_node(state: BotState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "BALANCE"
        flow["step"] = "ask_phone"
        state["response"] = choice(BALANCE_MISSING_PHONE)
        return state

    db = SessionLocal()
    try:
        result = get_account_balance(db, phone)
        tool_status = "failed" if "error" in result else "success"
        short_desc = (
            f"Agent checked account balance for {phone}"
            if tool_status == "success"
            else f"Agent failed to retrieve balance for {phone}: {result.get('error')}"
        )
    except Exception as e:
        result = {"error": str(e)}
        tool_status = "error"
        short_desc = f"Agent encountered an error checking balance for {phone}"
    finally:
        db.close()

    await log_tool_call(
        # vendor_id=state["agent"].vendor_id,
        vendor_id=VENDOR_ID,
        agent_name=state["agent_name"],
        agent_type="bank",
        tool_name="get_account_balance",
        tool_status=tool_status,
        short_description=short_desc,
        user_identifier=phone,
        raw_tool_input={"phone": phone},
        raw_tool_output=result,
    )

    flow["active"] = None
    flow["step"] = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(BALANCE_SUCCESS).format(
        balance=result["balance"], currency=result["currency"]
    )
    state.setdefault("intent", "BALANCE")
    return state


async def user_info_node(state: BotState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "USER_INFO"
        flow["step"] = "ask_phone"
        state["response"] = choice(USER_INFO_MISSING_PHONE)
        return state

    db = SessionLocal()
    try:
        result = get_user_info(db, phone)
        tool_status = "failed" if "error" in result else "success"
        short_desc = (
            f"Agent retrieved profile info for {phone}"
            if tool_status == "success"
            else f"Agent failed to retrieve user info for {phone}: {result.get('error')}"
        )
    except Exception as e:
        result = {"error": str(e)}
        tool_status = "error"
        short_desc = f"Agent encountered an error fetching user info for {phone}"
    finally:
        db.close()

    await log_tool_call(
        # vendor_id=state["agent"].vendor_id,
        vendor_id=VENDOR_ID,
        agent_name=state["agent_name"],
        agent_type="bank",
        tool_name="get_user_info",
        tool_status=tool_status,
        short_description=short_desc,
        user_identifier=phone,
        raw_tool_input={"phone": phone},
        raw_tool_output=result,
    )

    flow["active"] = None
    flow["step"] = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(USER_INFO_SUCCESS).format(
        name=result["user_name"], email=result["email"]
    )
    state.setdefault("intent", "USER_INFO")
    return state


async def rag_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)
    agent = state["agent"]
    try:
        vectorstore = agent.get_vectorstore()
    except Exception as e:
        logger.error(f"[VECTORSTORE ERROR] {e}")
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_FAILED"
        return state

    if vectorstore is None:
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_FAILED"
        return state

    system_prompt = agent.get_system_prompt()
    docs, scores = get_rag_context(state["message"], vectorstore)

    if not docs:
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_EMPTY"
        return state

    context = "\n\n".join(docs)

    if not context.strip():
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_EMPTY"
        return state

    history = state["memory"]["history"][-8:]
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}" for h in history
    )

    prompt = f"""Answer the user's question using ONLY the context below.

CRITICAL RULES:
- Do NOT invent information
- Do NOT merge or skip list items
- If missing → say contact ABB support at 937
- Use ONLY provided context

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{state['message']}

ANSWER:"""

    client = AsyncClient()
    try:
        res = await client.generate(
            model="gemma4:e4b",
            prompt=prompt,
            system=system_prompt,
            options={
                "temperature": 0.2,
                "top_p": 0.9
            }
        )
        answer = res.response.strip()

    except Exception as e:
        logger.error(f"[LLM ERROR] {e}")
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "LLM_FAILED"
        return state

    state["memory"]["history"].append(
        {"role": "user", "content": state["message"]}
    )
    state["memory"]["history"].append(
        {"role": "assistant", "content": answer}
    )
    state["memory"]["flow"]["expandable"] = True
    state["memory"]["flow"]["last_expand_offer"] = True
    state["response"] = answer
    state.setdefault("intent", "RAG")

    return state


async def expand_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    flow = state["memory"]["flow"]
    agent = state["agent"]
    system_prompt = agent.get_system_prompt()

    last_answer = None
    for msg in reversed(state["memory"]["history"]):
        if msg["role"] == "assistant":
            last_answer = msg["content"]
            break

    last_answer = last_answer or ""

    client = AsyncClient()

    res = await client.generate(
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
"""
    )
    expanded = res.response.strip()

    flow["expandable"] = False
    flow["last_expand_offer"] = False

    state["memory"]["history"].append({"role": "user", "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": expanded})

    state["response"] = expanded
    state.setdefault("intent", "EXPAND")
    return state


async def greeting_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    state["response"] = choice(GREETING_RESPONSES)
    state.setdefault("intent", "GREETING")
    return state


async def small_talk_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    state["response"] = choice(THANK_YOU_RESPONSES)
    state.setdefault("intent", "SMALL_TALK")
    return state


def build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(BotState)

    graph.add_node("detect", detect_intent)
    graph.add_node("control", control_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("balance", balance_node)
    graph.add_node("user_info", user_info_node)
    graph.add_node("ticket", ticket_node)
    graph.add_node("rag", rag_node)
    graph.add_node("expand", expand_node)

    graph.add_edge(START, "detect")

    graph.set_entry_point("detect")

    graph.add_conditional_edges(
        "detect",
        router,
        {
            "GREETING": "greeting",
            "BALANCE": "balance",
            "USER_INFO": "user_info",
            "TICKET": "ticket",
            "SMALL_TALK": "small_talk",
            "CONTROL": "control",
            "RAG": "rag",
            "EXPAND": "expand",
        }
    )

    for node in [
        "greeting",
        "balance",
        "user_info",
        "ticket",
        "rag",
        "expand",
        "small_talk",
        "control",
    ]:
        graph.add_edge(node, END)

    return graph.compile()