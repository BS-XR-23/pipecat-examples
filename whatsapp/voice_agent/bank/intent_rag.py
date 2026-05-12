from typing import TypedDict, Optional, Dict, Any
from random import choice
import json
import os
import re
import asyncio
from dotenv import load_dotenv
import copy
from loguru import logger
from ollama import AsyncClient
from db_bank.database import SessionLocal
from utils.helpers import extract_phone_number, safe_parse_json, extract_json
from voice_agent.bank.agent_tools import get_account_balance, get_user_info, create_ticket
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

load_dotenv()
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))
OLLAMA_INTENT_MODEL = os.getenv("OLLAMA_INTENT_MODEL", "gemma4:e2b")
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e4b")
RAG_HISTORY_LIMIT = int(os.getenv("RAG_HISTORY_LIMIT", "4"))
RAG_K = int(os.getenv("RAG_K", "3"))


class VoiceState(TypedDict):
    session_id: str
    message: str
    memory: Dict[str, Any]


ollama_client = AsyncClient()

# ── Regex fast-path intent patterns ───────────────────────────────────────────
_FAST_INTENTS = [
    (re.compile(r"^\s*(hi|hello|hey|good\s*(morning|evening|afternoon))\b", re.I), "GREETING"),
    (re.compile(r"^\s*(thanks?|thank you|okay|ok|bye|got it|sure|alright)\s*$", re.I), "SMALL_TALK"),
    (re.compile(r"\b(balance|funds?|how much.*account)\b", re.I), "BALANCE"),
    (re.compile(r"\b(my info|my details|my profile|my email)\b", re.I), "USER_INFO"),
    (re.compile(r"\b(my card.*block|can't login|transaction.*fail|have an issue)\b", re.I), "TICKET"),
]


def safe_state(state):
    return copy.deepcopy(state)


def ensure_memory(state: VoiceState):
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


async def detect_intent(state: VoiceState) -> str:
    state = ensure_memory(state)
    message = state["message"]

    # Fast-path: skip Ollama for obvious intents
    for pattern, intent in _FAST_INTENTS:
        if pattern.search(message):
            logger.info(f"[VOICE INTENT FAST-PATH] {intent}")
            return intent

    # Slow-path: Ollama for ambiguous messages
    try:
        history = state["memory"]["history"][-RAG_HISTORY_LIMIT:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in history
        )

        prompt = INTENT_PROMPT.format(
            history=history_text,
            message=message
        )

        res = await ollama_client.generate(
            model=OLLAMA_INTENT_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.75}
        )

        raw = res.response.strip()
        logger.info(f"[VOICE INTENT RAW] {raw}")

        data = safe_parse_json(raw)

        if not data:
            logger.warning(f"[VOICE INTENT FALLBACK - no data] raw: {raw}")
            return "RAG"

        allowed = {
            "GREETING", "BALANCE", "USER_INFO",
            "TICKET", "RAG", "EXPAND",
            "CONTROL", "SMALL_TALK"
        }
        raw_intent = str(data.get("intent", "RAG")).strip().upper()
        intent = raw_intent if raw_intent in allowed else "RAG"
        logger.info(f"[VOICE INTENT RESOLVED] {intent}")
        return intent

    except Exception as e:
        logger.warning(f"[VOICE INTENT NODE ERROR] {e}, falling back to RAG")
        return "RAG"


def router(intent: str, memory: Dict[str, Any]) -> str:
    allowed = {
        "GREETING", "BALANCE", "USER_INFO",
        "TICKET", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK"
    }

    flow = memory.get("flow", {})
    active = flow.get("active")

    if active and active in allowed:
        return active

    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()
    return intent if intent in allowed else "RAG"


# ── Nodes ──────────────────────────────────────────────────────────────────────
async def handle_greeting(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    return choice(GREETING_RESPONSES)


async def handle_small_talk(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    return choice(THANK_YOU_RESPONSES)


async def handle_control(state: VoiceState) -> str:
    state = ensure_memory(state)
    text = state["message"].lower().strip()
    flow = state["memory"]["flow"]

    if text in ["no", "nope"]:
        flow["expandable"] = False
        flow["last_expand_offer"] = False
        return "Alright 👍"

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            # ── Actually invoke expand instead of returning a placeholder ──────
            return await handle_expand(state)
        return "Okay 👍 How can I help you?"

    return "Alright 👍"


# ── Balance ────────────────────────────────────────────────────────────────────
async def handle_balance(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    # ── Try to extract phone from the current message first ───────────────────
    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone  # ← persist to profile

    if not phone:
        flow["active"] = "BALANCE"
        flow["step"] = "ask_phone"
        return choice(BALANCE_MISSING_PHONE)

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

    # ── Await the log call (not fire-and-forget) ──────────────────────────────
    await log_tool_call(
        vendor_id=VENDOR_ID,
        agent_name="voice_agent",
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
        return result["error"]

    # ── Use the BALANCE_SUCCESS response template ─────────────────────────────
    return choice(BALANCE_SUCCESS).format(
        balance=result["balance"], currency=result["currency"]
    )


# ── User Info ──────────────────────────────────────────────────────────────────
async def handle_user_info(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    # ── Try to extract phone from the current message first ───────────────────
    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone  # ← persist to profile

    if not phone:
        flow["active"] = "USER_INFO"
        flow["step"] = "ask_phone"
        return choice(USER_INFO_MISSING_PHONE)

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

    # ── Await the log call (not fire-and-forget) ──────────────────────────────
    await log_tool_call(
        vendor_id=VENDOR_ID,
        agent_name="voice_agent",
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
        return result["error"]

    # ── Use the USER_INFO_SUCCESS response template ───────────────────────────
    return choice(USER_INFO_SUCCESS).format(
        name=result["user_name"], email=result["email"]
    )


# ── Ticket ─────────────────────────────────────────────────────────────────────
async def handle_ticket(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]

    step = flow.get("step") if flow.get("step") is not None else "confirm"
    logger.info(f"[TICKET STEP] active={flow.get('active')} step={step!r} msg={state['message']!r}")
    text = state["message"].lower()

    if step == "confirm":
        flow["active"] = "TICKET"
        flow["step"] = "await_confirmation"
        return choice(TICKET_CONFIRM_INTENT)

    if step == "await_confirmation":
        if any(x in text for x in ["yes", "create", "ticket", "ok", "sure"]):
            flow["step"] = "collect_issue"
            return choice(TICKET_ASK_ISSUE)
        elif any(x in text for x in ["human", "agent", "representative"]):
            flow["active"] = None
            flow["step"] = None
            return choice(HUMAN_HANDOFF)
        else:
            flow["active"] = None
            flow["step"] = None
            return "Alright, let me know how else I can help you."

    if step == "collect_issue":
        ticket["issue"] = state["message"]
        flow["step"] = "collect_phone"
        return choice(TICKET_ASK_PHONE)

    if step == "collect_phone":
        phone = extract_phone_number(state["message"])

        if not phone:
            return (
                "I couldn't catch that number. "
                "Please say your 11-digit registered phone number clearly, "
                "starting with zero one."
            )

        ticket["phone"] = phone

        prompt = f"""Convert issue into JSON:
{{
  "category": "card_issue | auth_issue | transaction_issue | account_issue | general",
  "short_description": "max 15 words"
}}

Issue:
{ticket['issue']}"""

        res = await ollama_client.generate(model=OLLAMA_INTENT_MODEL, prompt=prompt)

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

        # ── Await the log call (not fire-and-forget) ──────────────────────────
        await log_tool_call(
            vendor_id=VENDOR_ID,
            agent_name="voice_agent",
            agent_type="bank",
            tool_name="create_ticket",
            tool_status=tool_status,
            short_description=short_desc,
            user_identifier=phone,
            raw_tool_input={
                "phone": phone,
                "category": data.get("category"),
                "short_description": data.get("short_description"),
            },
            raw_tool_output=result,
        )

        flow["active"] = None
        flow["step"] = None
        ticket.clear()

        if "error" in result:
            return f"Sorry, I couldn't create the ticket. {result['error']}"

        # ── Append ticket_id to the success response ──────────────────────────
        ticket_id = result.get("ticket_id", "UNKNOWN")
        return choice(TICKET_FINAL) + f" Ticket ID: {ticket_id}"

    return "I understand you're facing an issue. Would you like me to raise a support ticket?"


# ── RAG ────────────────────────────────────────────────────────────────────────
async def handle_rag(state: VoiceState, vectorstore, system_prompt: str) -> str:
    state = ensure_memory(safe_state(state))
    flow = state["memory"]["flow"]

    if vectorstore is None:
        return "Please contact ABB support at 937."

    try:
        docs, scores = get_rag_context(state["message"], vectorstore, k=RAG_K)
    except Exception as e:
        logger.error(f"[VOICE RAG VECTOR ERROR] {e}")
        return "Please contact ABB support at 937."

    if not docs:
        return "Please contact ABB support at 937."

    context = "\n\n".join(docs)
    if not context.strip():
        return "Please contact ABB support at 937."

    history = state["memory"]["history"][-8:]
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in history
    )

    # ── Use the structured prompt with CRITICAL RULES ─────────────────────────
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
{state["message"]}

ANSWER:"""

    try:
        res = await ollama_client.generate(
            model=OLLAMA_RAG_MODEL,
            prompt=prompt,
            system=system_prompt,
            options={"temperature": 0.2, "top_p": 0.9}
        )
        answer = res.response.strip()
    except Exception as e:
        logger.error(f"[VOICE RAG LLM ERROR] {e}")
        return "Please contact ABB support at 937."

    # ── Update history and set expandable flags ───────────────────────────────
    state["memory"]["history"].append({"role": "user", "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": answer})
    flow["expandable"] = True
    flow["last_expand_offer"] = True

    return answer


# ── Expand ─────────────────────────────────────────────────────────────────────
async def handle_expand(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    flow = state["memory"]["flow"]

    last_answer = next(
        (m["content"] for m in reversed(state["memory"]["history"]) if m["role"] == "assistant"),
        ""
    )

    if not last_answer:
        return "I don't have a previous answer to expand on. How can I help you?"

    try:
        res = await ollama_client.generate(
            model=OLLAMA_RAG_MODEL,
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
    except Exception as e:
        logger.error(f"[EXPAND ERROR] {e}")
        return "I couldn't expand on that. Please try rephrasing your question."

    flow["expandable"] = False
    flow["last_expand_offer"] = False

    state["memory"]["history"].append({"role": "user", "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": expanded})

    return expanded


# ── Entry point ────────────────────────────────────────────────────────────────
async def process_intent_rag(state: VoiceState, vectorstore, system_prompt: str):
    state = ensure_memory(state)
    intent = await detect_intent(state)
    route = router(intent, state["memory"])

    logger.info(f"[VOICE ROUTE] intent={intent} → route={route}")

    if route == "GREETING":
        response = await handle_greeting(state)
    elif route == "SMALL_TALK":
        response = await handle_small_talk(state)
    elif route == "CONTROL":
        response = await handle_control(state)
    elif route == "EXPAND":
        response = await handle_expand(state)                   # ← wired up
    elif route == "BALANCE":
        response = await handle_balance(state)
    elif route == "USER_INFO":
        response = await handle_user_info(state)
    elif route == "TICKET":
        response = await handle_ticket(state)
    elif route == "RAG":
        response = await handle_rag(state, vectorstore, system_prompt)
    else:
        response = "I'm sorry, I didn't understand that."

    # ── Only append to history for non-RAG routes (RAG node does it itself) ──
    if route != "RAG":
        state["memory"]["history"].append({"role": "user", "content": state["message"]})
        state["memory"]["history"].append({"role": "assistant", "content": response})

    return response, state