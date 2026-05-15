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
    THANK_YOU_RESPONSES,
)

load_dotenv()
VENDOR_ID                   = int(os.getenv("VENDOR_ID", "0"))
OLLAMA_INTENT_MODEL         = os.getenv("OLLAMA_INTENT_MODEL", "gemma4:e2b")
OLLAMA_RAG_MODEL            = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e4b")
RAG_HISTORY_LIMIT           = int(os.getenv("RAG_HISTORY_LIMIT", "4"))
RAG_K                       = int(os.getenv("RAG_K", "3"))
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.5"))


class VoiceState(TypedDict):
    session_id: str
    message: str
    memory: Dict[str, Any]


ollama_client = AsyncClient()


# ──────────────────────────────────────────────
# MEMORY HELPERS
# ──────────────────────────────────────────────
def safe_state(state: VoiceState) -> VoiceState:
    return copy.deepcopy(state)


def ensure_memory(state: VoiceState) -> VoiceState:
    state.setdefault("memory", {})
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
    # Suspension slots so RAG interruptions don't destroy in-progress flows
    flow.setdefault("suspended_flow", None)
    flow.setdefault("suspended_step", None)

    return state


# ──────────────────────────────────────────────
# SECONDARY CANCEL-vs-RAG CHECK
# When a CANCEL_FLOW message also contains an informational question,
# route to RAG so the user gets their answer instead of a bare abort.
# ──────────────────────────────────────────────
CANCEL_QUESTION_PROMPT = """
You are an intent router helper for a banking voice assistant.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{{"route_to_rag":<true|false>,"reason":"<short explanation>"}}

The user message may contain both a cancellation signal and an informational request.

Instructions:
- If the user is only asking to cancel or stop the current flow, return false.
- If the user is also asking a factual, informational question about banking products,
  services, policies, or procedures, return true.
- Do not rely on keyword matching alone; use semantic understanding.
- If the message is ambiguous, prefer cancellation unless a clear informational
  request is present.

Active flow: {active_flow}

Conversation history:
{history}

User message:
{message}
"""


async def should_route_cancel_flow_to_rag(state: VoiceState) -> bool:
    """
    Returns True when a CANCEL_FLOW message also contains an informational
    question that should be answered via RAG instead of simply aborting.
    """
    try:
        flow        = state["memory"]["flow"]
        active_flow = flow.get("active") or "None"
        history     = state["memory"]["history"][-6:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}" for h in history
        )

        prompt = CANCEL_QUESTION_PROMPT.format(
            active_flow=active_flow,
            history=history_text,
            message=state.get("message", ""),
        )

        res = await ollama_client.generate(
            model=OLLAMA_INTENT_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.75},
        )

        raw = res.response.strip()
        logger.info(f"[CANCEL CHECK RAW] {raw}")

        data = safe_parse_json(raw)
        if not isinstance(data, dict):
            data = extract_json(raw)
        if not isinstance(data, dict):
            logger.warning(
                "[CANCEL CHECK PARSE] expected JSON object, got %s",
                type(data).__name__,
            )
            return False

        return bool(data.get("route_to_rag", False))

    except Exception as e:
        logger.warning(f"[CANCEL CHECK ERROR] {e}")
        return False


# ──────────────────────────────────────────────
# INTENT DETECTION  (LLM-only, no regex fast-path)
# ──────────────────────────────────────────────
INTENT_PROMPT = """
You are a routing system for a banking chatbot.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{{"intent":"<INTENT>","confidence":<0.0-1.0>}}

Allowed intents:
- GREETING
- BALANCE
- USER_INFO
- TICKET
- RAG
- EXPAND
- CONTROL
- SMALL_TALK
- CANCEL_FLOW
- RESUME_FLOW

---------- ACTIVE CONTEXT ----------
Active flow    : {active_flow}
Flow step      : {flow_step}
Suspended flow : {suspended_flow}
------------------------------------

Routing Rules:

- CANCEL_FLOW [HIGHEST PRIORITY when active_flow is not None]:
  The user is EXPLICITLY rejecting, cancelling, or opting out of the current
  active flow or action.
  Examples: "no", "don't", "do not", "forget it", "cancel", "skip",
            "never mind", "I don't want a ticket", "do not create a ticket",
            "stop", "no thanks", "I said no", "I changed my mind".
  → Use CANCEL_FLOW whenever active_flow != "None" AND the message contains
    any negation or opt-out language.

- RESUME_FLOW:
  The user wants to continue or go back to a previously suspended flow.
  Only applicable when suspended_flow is not "None".
  Examples: "let's continue", "go back to the ticket", "resume",
            "continue where we left off", "yes let's keep going", "ok continue".

- GREETING:
  User is saying hello, hi, good morning, good evening, or starting a
  new conversation.

- SMALL_TALK:
  Casual non-banking chat: thanks, bye, okay, got it, casual replies, filler.

- CONTROL:
  Short affirmations like yes, ok, sure — ONLY when no active flow and the
  user is clearly not cancelling anything.

- EXPAND:
  User wants more detail or clarification of the previous assistant answer.

- BALANCE:
  Asking about account balance, available funds, or money inquiry.

- USER_INFO:
  Asking for profile, account details, email, or personal info.

- TICKET:
  User is PERSONALLY experiencing a problem RIGHT NOW and wants to report it.
  Examples: "my card is blocked", "I can't login", "my transaction failed",
            "create a ticket for me", "create a support ticket".
  NOT TICKET: questions about complaint processes, policies, or requirements.

- RAG:
  Any informational or knowledge-based question about banking services,
  policies, procedures, or requirements.
  Examples: "what documents do I need", "how do I submit a complaint",
            "tell me about loans", "what are the requirements for X".
  ALWAYS use RAG when the user asks "what", "how", "tell me", "explain",
  "can you tell me", "tell me about".

Critical Rules:
- If active_flow is not "None" AND the message contains negation words
  (no, don't, do not, not, never, cancel, stop, forget, never mind, skip)
  → ALWAYS return CANCEL_FLOW.
- If the message starts with "what", "how", "tell me", "explain",
  "can you tell me" → always RAG (unless active_flow cancellation applies).
- Only use TICKET if the user is personally reporting an active issue
  OR explicitly requesting ticket creation.
  NOT for asking about a complaint process or policy.

Conversation history (last {history_limit} turns):
{history}

User message:
{message}
"""


async def detect_intent(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow           = state["memory"]["flow"]
    active_flow    = flow.get("active")         or "None"
    flow_step      = flow.get("step")           or "None"
    suspended_flow = flow.get("suspended_flow") or "None"

    try:
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in state["memory"]["history"][-RAG_HISTORY_LIMIT:]
        )

        prompt = INTENT_PROMPT.format(
            active_flow    = active_flow,
            flow_step      = flow_step,
            suspended_flow = suspended_flow,
            history_limit  = RAG_HISTORY_LIMIT,
            history        = history_text,
            message        = state["message"],
        )

        res = await ollama_client.generate(
            model=OLLAMA_INTENT_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.75},
        )

        raw = res.response.strip()
        logger.info(f"[VOICE INTENT RAW] {raw}")

        data = safe_parse_json(raw)

        if not data:
            logger.warning(f"[VOICE INTENT FALLBACK – no data] raw: {raw}")
            return "RAG"

        allowed = {
            "GREETING", "BALANCE", "USER_INFO",
            "TICKET", "RAG", "EXPAND",
            "CONTROL", "SMALL_TALK", "CANCEL_FLOW", "RESUME_FLOW",
        }

        raw_intent = str(data.get("intent", "RAG")).strip().upper()
        confidence = float(data.get("confidence", 1.0))

        # Low-confidence → fall back to RAG rather than misrouting
        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            logger.warning(
                f"[VOICE INTENT LOW CONFIDENCE] {raw_intent} @ {confidence:.2f}, "
                "falling back to RAG"
            )
            return "RAG"

        intent = raw_intent if raw_intent in allowed else "RAG"
        logger.info(f"[VOICE INTENT RESOLVED] {intent} (confidence={confidence:.2f})")

        # Secondary check: CANCEL_FLOW message that also contains a question
        # → upgrade to RAG so the user gets their answer
        if intent == "CANCEL_FLOW" and active_flow != "None":
            if await should_route_cancel_flow_to_rag(state):
                logger.info(
                    "[VOICE INTENT] CANCEL_FLOW also contains an informational "
                    "request; routing to RAG"
                )
                return "RAG"

        return intent

    except Exception as e:
        logger.warning(f"[VOICE INTENT ERROR] {e}, falling back to RAG")
        return "RAG"


# ──────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────
def router(intent: str, memory: Dict[str, Any]) -> str:
    allowed = {
        "GREETING", "BALANCE", "USER_INFO",
        "TICKET", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK", "CANCEL_FLOW",
    }

    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()

    # Cancellation always wins
    if intent == "CANCEL_FLOW":
        return "CANCEL_FLOW"

    # RESUME_FLOW: restore a suspended flow
    if intent == "RESUME_FLOW":
        flow = memory.get("flow", {})
        if flow.get("suspended_flow"):
            return "RESUME_FLOW"
        # Nothing suspended — treat as neutral
        return "CONTROL"

    flow   = memory.get("flow", {})
    active = flow.get("active")

    # Resume an in-progress flow ONLY for neutral/continuation intents
    continuation_intents = {"CONTROL", "SMALL_TALK", "EXPAND"}
    if active and active in allowed and intent in continuation_intents:
        return active

    # A genuine new intent overrides the active flow — clear stale slots
    if active and intent not in continuation_intents:
        logger.info(
            f"[VOICE ROUTER] New intent '{intent}' overrides active flow '{active}'. "
            "Clearing active flow."
        )
        flow["active"] = None
        flow["step"]   = None

    # Any explicit new intent is honoured directly
    return intent if intent in allowed else "RAG"


# ──────────────────────────────────────────────
# CANCEL-FLOW HANDLER
# ──────────────────────────────────────────────
async def handle_cancel_flow(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    active_flow    = flow.get("active")
    suspended_flow = flow.get("suspended_flow")

    flow["active"]            = None
    flow["step"]              = None
    flow["suspended_flow"]    = None
    flow["suspended_step"]    = None
    flow["expandable"]        = False
    flow["last_expand_offer"] = False

    # Clear ticket data for whichever flow was running
    reported_flow = active_flow or suspended_flow
    if reported_flow == "TICKET":
        state["memory"]["ticket"] = {}

    cancellation_messages = {
        "TICKET": (
            "No problem, I won't raise a ticket. "
            "Let me know if there's anything else I can help you with."
        ),
        "BALANCE": (
            "Got it, I've cancelled the balance check. "
            "Let me know if you need anything else."
        ),
        "USER_INFO": (
            "Sure, I've stopped that. "
            "How else can I help you?"
        ),
    }
    return cancellation_messages.get(
        reported_flow,
        "Alright, I've cancelled that. How else can I help you?"
    )


# ──────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────
async def handle_greeting(state: VoiceState) -> str:
    return choice(GREETING_RESPONSES)


async def handle_small_talk(state: VoiceState) -> str:
    return choice(THANK_YOU_RESPONSES)


# ── Control ────────────────────────────────────────────────────────────────────
# Bug fix: receives system_prompt so it can pass it to handle_expand when the
# user says "yes" to a follow-up offer. Without it, expand runs without any
# persona, tone, or RAG rules applied.
async def handle_control(state: VoiceState, system_prompt: str = "") -> str:
    state = ensure_memory(state)
    text  = state["message"].lower().strip()
    flow  = state["memory"]["flow"]

    if text in ["no", "nope"]:
        flow["expandable"]        = False
        flow["last_expand_offer"] = False
        return "Alright! Let me know if there's anything else I can help you with."

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            # Bug fix: pass system_prompt through so expand LLM call is grounded
            return await handle_expand(state, system_prompt)
        return "Okay! How can I help you?"

    return "Got it! Let me know what you need."


# ── Balance ────────────────────────────────────────────────────────────────────
async def handle_balance(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    phone     = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "BALANCE"
        flow["step"]   = "ask_phone"
        return choice(BALANCE_MISSING_PHONE)

    db = SessionLocal()
    try:
        result = get_account_balance(db, phone)
        tool_status = "failed" if "error" in result else "success"
        short_desc  = (
            f"Agent checked account balance for {phone}"
            if tool_status == "success"
            else f"Agent failed to retrieve balance for {phone}: {result.get('error')}"
        )
    except Exception as e:
        result      = {"error": str(e)}
        tool_status = "error"
        short_desc  = f"Agent encountered an error checking balance for {phone}"
    finally:
        db.close()

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
    flow["step"]   = None

    if "error" in result:
        return result["error"]

    return choice(BALANCE_SUCCESS).format(
        balance=result["balance"], currency=result["currency"]
    )


# ── User Info ──────────────────────────────────────────────────────────────────
async def handle_user_info(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    phone     = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "USER_INFO"
        flow["step"]   = "ask_phone"
        return choice(USER_INFO_MISSING_PHONE)

    db = SessionLocal()
    try:
        result = get_user_info(db, phone)
        tool_status = "failed" if "error" in result else "success"
        short_desc  = (
            f"Agent retrieved profile info for {phone}"
            if tool_status == "success"
            else f"Agent failed to retrieve user info for {phone}: {result.get('error')}"
        )
    except Exception as e:
        result      = {"error": str(e)}
        tool_status = "error"
        short_desc  = f"Agent encountered an error fetching user info for {phone}"
    finally:
        db.close()

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
    flow["step"]   = None

    if "error" in result:
        return result["error"]

    return choice(USER_INFO_SUCCESS).format(
        name=result["user_name"], email=result["email"]
    )


# ── Ticket ─────────────────────────────────────────────────────────────────────
# Bug fixes applied:
#   1. intent == "TICKET" is now treated as affirmative in await_confirmation so
#      "create a ticket for me" / "create a support ticket" advances the flow
#      instead of being silently rejected.
#   2. "go ahead" is checked with `in text` (substring) instead of `in text.split()`
#      (word list) so the two-word phrase is matched correctly.
#   3. ticket dict is cleared on every exit path (cancel, human handoff, non-yes).
async def handle_ticket(state: VoiceState) -> str:
    state  = ensure_memory(state)
    flow   = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]

    step = flow.get("step") if flow.get("step") is not None else "confirm"
    logger.info(
        f"[TICKET STEP] active={flow.get('active')} "
        f"step={step!r} msg={state['message']!r}"
    )
    text = state["message"].lower()

    # ── Step: initial confirmation ask ─────────────────────────────────────────
    if step == "confirm":
        flow["active"] = "TICKET"
        flow["step"]   = "await_confirmation"
        return choice(TICKET_CONFIRM_INTENT)

    # ── Step: interpret the user's yes/no response ─────────────────────────────
    if step == "await_confirmation":
        # Human-agent request — narrow, low false-positive check
        if any(w in text for w in ["human", "agent", "representative", "person"]):
            flow["active"] = None
            flow["step"]   = None
            state["memory"]["ticket"] = {}
            return choice(HUMAN_HANDOFF)

        # Affirmative when:
        #   1. intent == TICKET — user explicitly re-requested ticket creation
        #      e.g. "create a ticket for me", "create a support ticket",
        #      "I have an account issue" (LLM re-detects as TICKET mid-flow)
        #   2. Short positive reply with CONTROL / SMALL_TALK intent
        #      e.g. "yes", "sure", "ok", "please"
        # Bug fix: "go ahead" checked with `in text` not `in text.split()` so
        # the two-word phrase matches correctly.
        intent        = state["memory"].get("_last_intent", "")
        positive_words = {"yes", "yeah", "ok", "okay", "sure", "please", "create"}
        affirmative_intents = {"CONTROL", "SMALL_TALK"}

        is_affirmative = (
            intent == "TICKET"
            or (
                intent in affirmative_intents
                and (
                    any(w in text.split() for w in positive_words)
                    or "go ahead" in text
                )
            )
        )

        if is_affirmative:
            flow["step"] = "collect_issue"
            return choice(TICKET_ASK_ISSUE)

        # Anything that is not a clear yes → exit gracefully
        flow["active"] = None
        flow["step"]   = None
        state["memory"]["ticket"] = {}
        return "Alright, let me know how else I can help you."

    # ── Step: collect issue description ───────────────────────────────────────
    if step == "collect_issue":
        ticket["issue"] = state["message"]
        flow["step"]    = "collect_phone"
        return choice(TICKET_ASK_PHONE)

    # ── Step: collect phone and create ticket ──────────────────────────────────
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
            data  = json.loads(match.group()) if match else {}

        db = SessionLocal()
        try:
            result = create_ticket(
                db,
                phone,
                category=data.get("category", "general"),
                short_description=data.get("short_description", ticket["issue"][:80]),
            )
            tool_status = "failed" if "error" in result else "success"
            short_desc  = (
                f"Agent created support ticket for {phone}: "
                f"{data.get('short_description', '')}"
                if tool_status == "success"
                else f"Agent failed to create ticket for {phone}"
            )
        except Exception as e:
            result      = {"error": str(e)}
            tool_status = "error"
            short_desc  = f"Agent encountered an error creating ticket for {phone}"
        finally:
            db.close()

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
        flow["step"]   = None
        ticket.clear()

        if "error" in result:
            return f"Sorry, I couldn't create the ticket. {result['error']}"

        ticket_id = result.get("ticket_id", "UNKNOWN")
        return choice(TICKET_FINAL) + f" Ticket ID: {ticket_id}"

    return "I understand you're facing an issue. Would you like me to raise a support ticket?"


# ── RAG ────────────────────────────────────────────────────────────────────────
async def handle_rag(state: VoiceState, vectorstore, system_prompt: str) -> str:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    # Suspend (don't destroy) the active flow so the user can resume it later
    if flow.get("active"):
        logger.info(
            f"[VOICE RAG] Suspending active flow '{flow['active']}' "
            "to answer informational question. Data preserved for resume."
        )
        flow["suspended_flow"] = flow["active"]
        flow["suspended_step"] = flow["step"]
        flow["active"] = None
        flow["step"]   = None

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

    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in state["memory"]["history"][-8:]
    )

    # Let the LLM know there is a suspended flow so it can offer to resume
    suspended_note = ""
    if flow.get("suspended_flow"):
        suspended_note = (
            f"\n\nNote: the user has a suspended '{flow['suspended_flow']}' flow. "
            "After answering their question, naturally offer to continue it if appropriate."
        )

    prompt = f"""You are a human customer support representative for ABB.

Answer naturally and conversationally like a real call center agent.

IMPORTANT:
- Use ONLY provided context for factual accuracy. However, you MAY generalize
  within the same product family (e.g., loan types, banking products) when
  terminology differs but meaning is equivalent.
- Do NOT invent information.
- Do NOT sound like a website, brochure, advertisement, or policy document.
- Speak in short, natural sentences.
- Answer the user's exact question first.
- If the context contains eligibility criteria and the user asks about documents,
  separate them clearly.
- Preserve all factual requirements and steps from the context.
- Do NOT copy large chunks verbatim unless necessary.
- Avoid marketing language and long paragraphs.
- Sound helpful and human.
- If the context doesn't cover the question, say:
  "Please contact ABB support at 937."{suspended_note}

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{state['message']}

A:"""

    try:
        res = await ollama_client.generate(
            model=OLLAMA_RAG_MODEL,
            prompt=prompt,
            system=system_prompt,
            options={"temperature": 0.2, "top_p": 0.9},
        )
        answer = res.response.strip()
    except Exception as e:
        logger.error(f"[VOICE RAG LLM ERROR] {e}")
        return "Please contact ABB support at 937."

    flow["expandable"]        = True
    flow["last_expand_offer"] = True
    return answer


# ── Expand ─────────────────────────────────────────────────────────────────────
# Bug fix: now accepts system_prompt parameter so the LLM is grounded with the
# correct persona and RAG rules. Previously called with no system prompt, causing
# the expand response to ignore tone, language, and factual constraints entirely.
async def handle_expand(state: VoiceState, system_prompt: str = "") -> str:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    last_answer = next(
        (
            m["content"]
            for m in reversed(state["memory"]["history"])
            if m["role"] == "assistant"
        ),
        "",
    )

    if not last_answer:
        return "I don't have a previous answer to expand on. How can I help you?"

    try:
        res = await ollama_client.generate(
            model=OLLAMA_RAG_MODEL,
            prompt=(
                f"Expand the following answer clearly and in detail.\n\n"
                f"PREVIOUS ANSWER:\n{last_answer}\n\n"
                f"USER REQUEST:\n{state['message']}\n\n"
                f"RULES:\n"
                f"- Only expand on the given answer.\n"
                f"- Do NOT use external knowledge.\n"
                f"- Sound conversational and helpful.\n"
                f"- If information is unavailable, say: "
                f"'Please contact ABB support at 937.'\n\n"
                f"EXPANDED ANSWER:"
            ),
            system=system_prompt,
            options={"temperature": 0.2, "top_p": 0.9},
        )
        expanded = res.response.strip()
    except Exception as e:
        logger.error(f"[EXPAND ERROR] {e}")
        return "I couldn't expand on that. Please try rephrasing your question."

    flow["expandable"]        = False
    flow["last_expand_offer"] = False
    return expanded


# ──────────────────────────────────────────────
# RESUME FLOW HANDLER
# Restores a suspended flow and immediately re-dispatches to the correct
# handler, combining a "resuming" message with the next step prompt.
# ──────────────────────────────────────────────
async def handle_resume_flow(
    state: VoiceState,
    vectorstore,
    system_prompt: str,
) -> str:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    suspended_flow = flow.get("suspended_flow")
    suspended_step = flow.get("suspended_step")

    # Restore
    flow["active"]         = suspended_flow
    flow["step"]           = suspended_step
    flow["suspended_flow"] = None
    flow["suspended_step"] = None

    logger.info(
        f"[VOICE RESUME_FLOW] Restored flow='{suspended_flow}' step='{suspended_step}'"
    )

    resume_messages = {
        "TICKET":    "Sure, let's continue with your support ticket. Here's where we left off:",
        "BALANCE":   "Of course, let's continue with the balance check.",
        "USER_INFO": "Sure, continuing with the account info lookup.",
    }
    intro = resume_messages.get(
        suspended_flow,
        "Sure, let's pick up where we left off."
    )

    # Re-dispatch to the correct handler and combine the messages
    if suspended_flow == "TICKET":
        next_response = await handle_ticket(state)
    elif suspended_flow == "BALANCE":
        next_response = await handle_balance(state)
    elif suspended_flow == "USER_INFO":
        next_response = await handle_user_info(state)
    else:
        return "How can I help you?"

    return f"{intro}\n\n{next_response}"


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
async def process_intent_rag(
    state: VoiceState,
    vectorstore,
    system_prompt: str,
) -> tuple[str, VoiceState]:
    state  = ensure_memory(state)
    intent = await detect_intent(state)
    route  = router(intent, state["memory"])

    # Store intent in memory so handle_ticket can read it in await_confirmation
    # without needing a separate parameter across every handler signature.
    state["memory"]["_last_intent"] = intent

    logger.info(f"[VOICE ROUTE] intent={intent} → route={route}")

    if route == "GREETING":
        response = await handle_greeting(state)
    elif route == "SMALL_TALK":
        response = await handle_small_talk(state)
    elif route == "CONTROL":
        # Bug fix: system_prompt threaded through so handle_control can pass it
        # to handle_expand when the user says "yes" to a follow-up offer.
        response = await handle_control(state, system_prompt)
    elif route == "CANCEL_FLOW":
        response = await handle_cancel_flow(state)
    elif route == "EXPAND":
        # Bug fix: system_prompt passed so the LLM is grounded correctly.
        response = await handle_expand(state, system_prompt)
    elif route == "BALANCE":
        response = await handle_balance(state)
    elif route == "USER_INFO":
        response = await handle_user_info(state)
    elif route == "TICKET":
        response = await handle_ticket(state)
    elif route == "RAG":
        response = await handle_rag(state, vectorstore, system_prompt)
    elif route == "RESUME_FLOW":
        response = await handle_resume_flow(state, vectorstore, system_prompt)
    else:
        response = "I'm sorry, I didn't understand that."

    # Persist turn to history (all routes handled uniformly here)
    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": response})

    return response, state