from typing import TypedDict, Optional, Dict, Any
from random import choice
import json
import os
from dotenv import load_dotenv
import re
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
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e2b")
RAG_HISTORY_LIMIT = int(os.getenv("RAG_HISTORY_LIMIT", "4"))
RAG_K = int(os.getenv("RAG_K", "3"))

class VoiceState(TypedDict):
    session_id: str
    message: str
    memory: Dict[str, Any]

ollama_client = AsyncClient()

def safe_state(state):
    state = copy.deepcopy(state)
    return state

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

async def detect_intent(state: VoiceState):
    state = ensure_memory(state)

    try:
        history = state["memory"]["history"][-RAG_HISTORY_LIMIT:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in history
        )

        prompt = INTENT_PROMPT.format(
            history=history_text,
            message=state["message"]
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
            intent = "RAG"
        else:
            allowed = {
                "GREETING", "BALANCE", "USER_INFO",
                "TICKET", "RAG", "EXPAND",
                "CONTROL", "SMALL_TALK"
            }

            raw_intent = str(data.get("intent", "RAG")).strip().upper()
            intent = raw_intent if raw_intent in allowed else "RAG"
            logger.info(f"[VOICE INTENT RESOLVED] {intent}")

    except Exception as e:
        logger.warning(f"[VOICE INTENT NODE ERROR] {e}, falling back to RAG")
        intent = "RAG"

    return intent

def router(intent: str, memory: Dict[str, Any]):
    flow = memory.get("flow", {})
    active = flow.get("active")

    if active == "TICKET":
        return "TICKET"
    if active == "BALANCE":
        return "BALANCE"
    if active == "USER_INFO":
        return "USER_INFO"

    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()

    allowed = {
        "GREETING", "BALANCE", "USER_INFO",
        "TICKET", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK"
    }

    return intent if intent in allowed else "RAG"

async def handle_greeting(state: VoiceState):
    state = ensure_memory(state)
    state = safe_state(state)

    response = choice(GREETING_RESPONSES)
    return response

async def handle_small_talk(state: VoiceState):
    state = ensure_memory(state)
    state = safe_state(state)

    response = choice(THANK_YOU_RESPONSES)
    return response

async def handle_control(state: VoiceState):
    state = ensure_memory(state)
    text = state["message"].lower().strip()
    flow = state["memory"]["flow"]

    if text in ["no", "nope"]:
        flow["expandable"] = False
        flow["last_expand_offer"] = False
        return "Alright 👍"

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            return "Let me expand on that."
        return "Okay 👍 How can I help you?"

    return "Alright 👍"

async def handle_balance(state: VoiceState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    if flow.get("active") != "BALANCE":
        flow["active"] = "BALANCE"
        return choice(BALANCE_MISSING_PHONE)

    phone = extract_phone_number(state["message"])
    if not phone:
        # Keep flow active so user can retry without restarting
        return (
            "I couldn't catch that number. "
            "Please say your 11-digit registered phone number clearly, "
            "starting with zero one."
        )

    db = SessionLocal()
    try:
        result = get_account_balance(db, phone)
        await log_tool_call(
            vendor_id=VENDOR_ID,
            agent_name="voice_agent",
            agent_type="bank",
            tool_name="get_account_balance",
            tool_status="success" if "error" not in result else "error",
            short_description="Fetched account balance",
            user_identifier=phone,
            raw_tool_input={"phone": phone},
            raw_tool_output=result
        )
        flow["active"] = None
        return result["message"]
    except Exception as e:
        logger.error(f"[BALANCE ERROR] {e}")
        flow["active"] = None
        return "Sorry, I couldn't retrieve your balance. Please try again."
    finally:
        db.close()

async def handle_user_info(state: VoiceState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    if flow.get("active") != "USER_INFO":
        flow["active"] = "USER_INFO"
        return choice(USER_INFO_MISSING_PHONE)

    phone = extract_phone_number(state["message"])
    if not phone:
        # Keep flow active so user can retry without restarting
        return (
            "I couldn't catch that number. "
            "Please say your 11-digit registered phone number clearly, "
            "starting with zero one."
        )

    db = SessionLocal()
    try:
        result = get_user_info(db, phone)
        await log_tool_call(
            vendor_id=VENDOR_ID,
            agent_name="voice_agent",
            agent_type="bank",
            tool_name="get_user_info",
            tool_status="success" if "error" not in result else "error",
            short_description="Fetched user info",
            user_identifier=phone,
            raw_tool_input={"phone": phone},
            raw_tool_output=result
        )
        flow["active"] = None
        return result["message"]
    except Exception as e:
        logger.error(f"[USER_INFO ERROR] {e}")
        flow["active"] = None
        return "Sorry, I couldn't retrieve your info. Please try again."
    finally:
        db.close()

async def handle_ticket(state: VoiceState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]

    step = flow.get("step") or "confirm"
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
            # Keep step active so user can retry without restarting
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

        res = await ollama_client.generate(model="gemma4:e2b", prompt=prompt)

        try:
            data = json.loads(res.response)
        except Exception:
            match = re.search(r"\{.*\}", res.response, re.DOTALL)
            data = json.loads(match.group()) if match else {}

        db = SessionLocal()
        try:
            result = create_ticket(
                db=db,
                phone_number=phone,
                category=data.get("category", "general"),
                short_description=data.get("short_description", ticket["issue"][:50])
            )
            await log_tool_call(
                vendor_id=VENDOR_ID,
                agent_name="voice_agent",
                agent_type="bank",
                tool_name="create_ticket",
                tool_status="success" if "error" not in result else "error",
                short_description="Created support ticket",
                user_identifier=phone,
                raw_tool_input={"phone": phone, "issue": ticket["issue"]},
                raw_tool_output=result
            )
            flow["active"] = None
            flow["step"] = None
            ticket.clear()
            return choice(TICKET_FINAL)
        except Exception as e:
            logger.error(f"[TICKET CREATE ERROR] {e}")
            flow["active"] = None
            flow["step"] = None
            ticket.clear()
            return "Sorry, I couldn't create the ticket. Please contact support."
        finally:
            db.close()

async def handle_rag(state: VoiceState, vectorstore, system_prompt):
    state = ensure_memory(state)
    state = safe_state(state)

    try:
        docs, scores = get_rag_context(state["message"], vectorstore, k=RAG_K)

        if not docs:
            return "Please contact ABB support at 937."

        context = "\n\n".join(docs)

        if not context.strip():
            return "Please contact ABB support at 937."

        history = state["memory"]["history"][-8:]
        history_text = "\n".join(
            f"{h['role']}: {h['content']}"
            for h in history
        )

        prompt = f"""{system_prompt}

Context:
{context}

History:
{history_text}

User: {state["message"]}

Assistant:"""

        # Use Ollama to generate response for RAG
        response = await ollama_client.generate(
            model=OLLAMA_RAG_MODEL,
            prompt=prompt,
            options={"temperature": 0.2, "top_p": 0.9}
        )
        return response.response.strip()

    except Exception as e:
        logger.error(f"[VOICE RAG ERROR] {e}")
        return "Please contact ABB support at 937."

async def process_intent_rag(state: VoiceState, vectorstore, system_prompt):
    intent = await detect_intent(state)
    route = router(intent, state["memory"])

    if route == "GREETING":
        response = await handle_greeting(state)
    elif route == "SMALL_TALK":
        response = await handle_small_talk(state)
    elif route == "CONTROL":
        response = await handle_control(state)
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

    # Update history
    state["memory"]["history"].append({"role": "user", "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": response})

    return response