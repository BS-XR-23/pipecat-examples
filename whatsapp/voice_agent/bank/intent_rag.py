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
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.5"))


class VoiceState(TypedDict):
    session_id: str
    message: str
    memory: Dict[str, Any]


ollama_client = AsyncClient()


# ── Fast-path intent patterns ──────────────────────────────────────────────────
# These run before Ollama to short-circuit obvious intents cheaply.
# IMPORTANT: CANCEL patterns are checked first so "no don't" never bleeds into
# other patterns. The fast-path is intentionally conservative — ambiguous cases
# fall through to the LLM.
_FAST_INTENTS = [
    # Cancellation — must come first to override everything else
    (
        re.compile(
            r"\b(no|nope|don'?t|do not|cancel|stop|forget it|never mind|skip|"
            r"i don'?t want|not now|no thanks)\b",
            re.I
        ),
        "CANCEL_FLOW",
    ),
    # Greetings
    (re.compile(r"^\s*(hi|hello|hey|good\s*(morning|evening|afternoon))\b", re.I), "GREETING"),
    # Small talk / closings — only when the whole message is a filler phrase
    (re.compile(r"^\s*(thanks?|thank you|okay|ok|bye|got it|sure|alright)\s*$", re.I), "SMALL_TALK"),
    # Balance
    (re.compile(r"\b(balance|funds?|how much.*account)\b", re.I), "BALANCE"),
    # User info
    (re.compile(r"\b(my info|my details|my profile|my email)\b", re.I), "USER_INFO"),
    # Ticket — personal active problem (narrow pattern to avoid false positives)
    (re.compile(r"\b(my card.*block|can'?t login|transaction.*fail|have an issue)\b", re.I), "TICKET"),
]

# Cancellation fast-path only fires when a flow is actually active.
# The pattern is extracted separately so we can conditionally apply it.
_CANCEL_PATTERN = _FAST_INTENTS[0][0]


def safe_state(state):
    return copy.deepcopy(state)


async def should_route_cancel_flow_to_rag(message: str, active_flow: str, history_text: str) -> bool:
    try:
        prompt = CANCEL_QUESTION_PROMPT.format(
            active_flow=active_flow,
            history=history_text,
            message=message,
        )

        res = await ollama_client.generate(
            model=OLLAMA_INTENT_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.75},
        )

        raw = res.response.strip()
        logger.info(f"[CANCEL QUESTION RAW] {raw}")
        data = safe_parse_json(raw)
        if not isinstance(data, dict):
            data = extract_json(raw)
        if not isinstance(data, dict):
            logger.warning(
                "[CANCEL QUESTION PARSE] expected JSON object, got %s",
                type(data).__name__
            )
            return False

        return bool(data.get("route_to_rag", False))
    except Exception as e:
        logger.warning(f"[CANCEL QUESTION NODE ERROR] {e}")
        return False


CANCEL_QUESTION_PROMPT = """
You are an intent router helper for a banking voice assistant.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{"route_to_rag":<true|false>,"reason":"<short explanation>"}

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


# ── Intent prompt ──────────────────────────────────────────────────────────────
# Key improvements over the original:
#   1. Active flow + step injected so the LLM understands where in the
#      conversation it is before deciding intent.
#   2. CANCEL_FLOW added as an explicit intent with clear examples and a hard
#      rule: if active_flow is set and negation words appear → CANCEL_FLOW.
#   3. Confidence field required; low-confidence results fall back to RAG.
#   4. Negative TICKET examples prevent false positives on informational queries.
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

---------- ACTIVE CONTEXT ----------
Active flow : {active_flow}
Flow step   : {flow_step}
------------------------------------

Routing Rules:

- CANCEL_FLOW [HIGHEST PRIORITY when active_flow is not None]:
  The user is explicitly rejecting, cancelling, or opting out of the current
  active flow or action.
  Examples: "no", "don't", "do not", "forget it", "cancel", "skip",
            "never mind", "I don't want a ticket", "do not create a ticket",
            "stop", "no thanks", "I said no".
  → Use CANCEL_FLOW whenever active_flow != "None" AND the message contains
    any negation or opt-out language.

- GREETING:
  User is saying hello, hi, good morning, good evening, or starting conversation.

- SMALL_TALK:
  Casual non-banking chat: thanks, bye, okay, got it, casual replies, filler.

- CONTROL:
  Short affirmations like yes, ok, sure (ONLY when no active flow and the user
  is clearly not cancelling anything).

- EXPAND:
  User wants more detail or clarification of the previous assistant answer.

- BALANCE:
  Asking about account balance, available funds, or money inquiry.

- USER_INFO:
  Asking for profile, account details, email, or personal info.

- TICKET:
  User is PERSONALLY experiencing a problem RIGHT NOW and wants to report it.
  Examples: "my card is blocked", "I can't login", "my transaction failed".
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
- Only use TICKET if the user is personally reporting an active issue,
  NOT asking about a process.

Conversation history (last {history_limit} turns):
{history}

User message:
{message}
"""


async def detect_intent(state: VoiceState) -> str:
    state = ensure_memory(state)
    message = state["message"]
    flow = state["memory"]["flow"]
    active_flow = flow.get("active") or "None"

    history = state["memory"]["history"][-RAG_HISTORY_LIMIT:]
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in history
    )

    # ── Fast-path: regex shortcuts before hitting Ollama ──────────────────────
    # Cancellation fast-path only fires when there is an active flow to cancel,
    # preventing innocent "no" / "nope" from being misread mid-greeting.
    if active_flow != "None" and _CANCEL_PATTERN.search(message):
        if await should_route_cancel_flow_to_rag(message, active_flow, history_text):
            logger.info("[VOICE INTENT FAST-PATH] CANCEL_FLOW with informational request; routing to RAG")
            return "RAG"
        logger.info("[VOICE INTENT FAST-PATH] CANCEL_FLOW")
        return "CANCEL_FLOW"

    for pattern, intent in _FAST_INTENTS[1:]:  # skip CANCEL_FLOW entry
        if pattern.search(message):
            logger.info(f"[VOICE INTENT FAST-PATH] {intent}")
            return intent

    # ── Slow-path: Ollama for ambiguous messages ──────────────────────────────
    try:
        flow_step = flow.get("step") or "None"
        history = state["memory"]["history"][-RAG_HISTORY_LIMIT:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in history
        )

        prompt = INTENT_PROMPT.format(
            active_flow=active_flow,
            flow_step=flow_step,
            history_limit=RAG_HISTORY_LIMIT,
            history=history_text,
            message=message,
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
            logger.warning(f"[VOICE INTENT FALLBACK - no data] raw: {raw}")
            return "RAG"

        allowed = {
            "GREETING", "BALANCE", "USER_INFO",
            "TICKET", "RAG", "EXPAND",
            "CONTROL", "SMALL_TALK", "CANCEL_FLOW",
        }

        raw_intent = str(data.get("intent", "RAG")).strip().upper()
        confidence = float(data.get("confidence", 1.0))

        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            logger.warning(
                f"[VOICE INTENT LOW CONFIDENCE] {raw_intent} @ {confidence:.2f}, "
                "falling back to RAG"
            )
            return "RAG"

        intent = raw_intent if raw_intent in allowed else "RAG"
        if intent == "CANCEL_FLOW" and active_flow != "None":
            if await should_route_cancel_flow_to_rag(message, active_flow, history_text):
                logger.info("[VOICE INTENT RESOLVED] CANCEL_FLOW with informational request; routing to RAG")
                intent = "RAG"

        logger.info(f"[VOICE INTENT RESOLVED] {intent} (confidence={confidence:.2f})")
        return intent

    except Exception as e:
        logger.warning(f"[VOICE INTENT NODE ERROR] {e}, falling back to RAG")
        return "RAG"


# ── Router ─────────────────────────────────────────────────────────────────────
# Key changes:
#   • CANCEL_FLOW is checked first — it always breaks out of any active flow
#     and routes to handle_control, which handles the clean exit.
#   • Active-flow continuation only fires for neutral/continuation intents so
#     a genuine new request (RAG, BALANCE, etc.) can interrupt mid-flow.
def router(intent: str, memory: Dict[str, Any], message: str) -> str:
    allowed = {
        "GREETING", "BALANCE", "USER_INFO",
        "TICKET", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK", "CANCEL_FLOW",
    }

    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()

    # Cancellation always wins — route to control for a graceful exit.
    if intent == "CANCEL_FLOW":
        return "CONTROL"

    flow = memory.get("flow", {})
    active = flow.get("active")

    # Resume an in-progress flow only for neutral/continuation intents
    # (i.e. the user didn't cancel and just sent the next expected message)
    continuation_intents = {"CONTROL", "SMALL_TALK", "EXPAND"}
    if active and active in allowed and intent in continuation_intents:
        return active

    # New substantive intent (RAG, BALANCE, etc.) — honour it and let the
    # flow lapse; the node itself clears flow state when interrupted.
    return intent if intent in allowed else "RAG"


# ── Nodes ──────────────────────────────────────────────────────────────────────

async def handle_greeting(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    return choice(GREETING_RESPONSES)


async def handle_small_talk(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    return choice(THANK_YOU_RESPONSES)


# ── Control / cancellation ─────────────────────────────────────────────────────
# Now handles CANCEL_FLOW exits in addition to yes/no acknowledgements.
async def handle_control(state: VoiceState, intent: str = "CONTROL") -> str:
    state = ensure_memory(state)
    text = state["message"].lower().strip()
    flow = state["memory"]["flow"]

    # ── Cancellation exit ──────────────────────────────────────────────────────
    if intent == "CANCEL_FLOW":
        active_flow = flow.get("active")
        flow["active"] = None
        flow["step"] = None
        flow["expandable"] = False
        flow["last_expand_offer"] = False

        if active_flow == "TICKET":
            return (
                "No problem, I won't raise a ticket. "
                "Let me know if there's anything else I can help you with."
            )
        return "Alright, I've cancelled that. How else can I help you?"

    # ── Standard yes/no ────────────────────────────────────────────────────────
    if text in ["no", "nope"]:
        flow["expandable"] = False
        flow["last_expand_offer"] = False
        return "Alright 👍"

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            return await handle_expand(state)
        return "Okay 👍 How can I help you?"

    return "Alright 👍"


# ── Balance ────────────────────────────────────────────────────────────────────
async def handle_balance(state: VoiceState) -> str:
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

    return choice(BALANCE_SUCCESS).format(
        balance=result["balance"], currency=result["currency"]
    )


# ── User Info ──────────────────────────────────────────────────────────────────
async def handle_user_info(state: VoiceState) -> str:
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

    return choice(USER_INFO_SUCCESS).format(
        name=result["user_name"], email=result["email"]
    )


# ── Ticket ─────────────────────────────────────────────────────────────────────
# Key changes:
#   • CANCEL_FLOW intent bails out immediately at any step.
#   • await_confirmation uses intent-based logic instead of keyword matching,
#     so "do not create a ticket" is never misread as a yes.
async def handle_ticket(state: VoiceState, intent: str = "TICKET") -> str:
    state = ensure_memory(state)
    flow = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]

    # ── Bail out on any cancellation signal at any step ────────────────────────
    if intent == "CANCEL_FLOW":
        flow["active"] = None
        flow["step"] = None
        return (
            "No problem, I won't raise a ticket. "
            "Let me know if there's anything else I can help you with."
        )

    step = flow.get("step") if flow.get("step") is not None else "confirm"
    logger.info(
        f"[TICKET STEP] active={flow.get('active')} "
        f"step={step!r} msg={state['message']!r}"
    )
    text = state["message"].lower()

    # ── Step: initial confirmation ask ─────────────────────────────────────────
    if step == "confirm":
        flow["active"] = "TICKET"
        flow["step"] = "await_confirmation"
        return choice(TICKET_CONFIRM_INTENT)

    # ── Step: interpret the user's yes/no response ─────────────────────────────
    # Uses intent + a narrow positive-word check instead of raw keyword matching.
    # This means "do not create a ticket" (intent=CANCEL_FLOW, already caught
    # above) and "ticket? no" never accidentally match the affirmative branch.
    if step == "await_confirmation":
        # Human-agent request — check text explicitly (narrow, low-FP pattern)
        if any(w in text for w in ["human", "agent", "representative", "person"]):
            flow["active"] = None
            flow["step"] = None
            return choice(HUMAN_HANDOFF)

        # Genuine affirmation: short, clearly positive words only
        positive_words = {"yes", "yeah", "ok", "okay", "sure", "please", "go ahead", "create"}
        is_affirmative = (
            intent in {"CONTROL", "SMALL_TALK"}
            and any(w in text.split() for w in positive_words)
        )

        if is_affirmative:
            flow["step"] = "collect_issue"
            return choice(TICKET_ASK_ISSUE)

        # Anything that isn't a clear yes → exit the flow gracefully
        flow["active"] = None
        flow["step"] = None
        return "Alright, let me know how else I can help you."

    # ── Step: collect issue description ───────────────────────────────────────
    if step == "collect_issue":
        ticket["issue"] = state["message"]
        flow["step"] = "collect_phone"
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
            data = json.loads(match.group()) if match else {}

        db = SessionLocal()
        try:
            result = create_ticket(
                db,
                phone,
                category=data.get("category", "general"),
                short_description=data.get("short_description", ticket["issue"][:80]),
            )
            tool_status = "failed" if "error" in result else "success"
            short_desc = (
                f"Agent created support ticket for {phone}: "
                f"{data.get('short_description', '')}"
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

        ticket_id = result.get("ticket_id", "UNKNOWN")
        return choice(TICKET_FINAL) + f" Ticket ID: {ticket_id}"

    return "I understand you're facing an issue. Would you like me to raise a support ticket?"


# ── RAG ────────────────────────────────────────────────────────────────────────
async def handle_rag(state: VoiceState, vectorstore, system_prompt: str) -> str:
    state = ensure_memory(safe_state(state))
    flow = state["memory"]["flow"]

    # If the user fires a RAG question while a flow is active, lapse the flow
    # gracefully so they get a useful answer instead of confused continuation.
    if flow.get("active"):
        logger.info(
            f"[VOICE RAG] Interrupting active flow '{flow['active']}' "
            "to answer informational question."
        )
        flow["active"] = None
        flow["step"] = None

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

    prompt = f"""
            You are a human customer support representative for ABB.

            Answer naturally and conversationally like a real call center agent.

            IMPORTANT:
            - - Use ONLY provided context for factual accuracy. However, you MAY generalize within the same product family (e.g., loan types, banking products) when terminology differs but meaning is equivalent
            - Do NOT invent information
            - Do NOT sound like a website, brochure, advertisement, or policy document
            - Speak in short, natural sentences
            - Answer the user's exact question first
            - If the context contains eligibility criteria and the user asks about documents, separate them clearly
            - Preserve all factual requirements and steps from the context
            - Do NOT copy large chunks verbatim unless necessary
            - Avoid marketing language
            - Avoid long paragraphs
            - Sound helpful and human

            CONTEXT:
            {context}

            CONVERSATION HISTORY:
            {history_text}

            USER QUESTION:
            {state['message']}

            ASSISTANT:
            """

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

    state["memory"]["history"].append({"role": "user", "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": answer})
    flow["expandable"] = True
    flow["last_expand_offer"] = True

    return answer


# ── Expand ─────────────────────────────────────────────────────────────────────
async def handle_expand(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    flow = state["memory"]["flow"]
    agent = state["agent"]
    system_prompt = agent.get_system_prompt()

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
            system=system_prompt,
            prompt=f"""
                    You are a human ABB customer support agent.

                    The user asked a follow-up question.

                    Expand naturally based ONLY on the previous answer.

                    IMPORTANT:
                    - Keep the same factual information
                    - Do NOT invent new information
                    - Speak conversationally
                    - Keep sentences short and natural
                    - Do NOT sound promotional
                    - Do NOT repeat unnecessary details
                    - If information is unavailable, say:
                    "Please contact ABB support at 937."

                    PREVIOUS ANSWER:
                    {last_answer}

                    USER:
                    {state['message']}

                    ASSISTANT:
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
    route = router(intent, state["memory"], state["message"])

    logger.info(f"[VOICE ROUTE] intent={intent} → route={route}")

    # Pass the raw intent into nodes that need to distinguish CANCEL_FLOW
    # from a regular CONTROL message, without changing their signatures for
    # callers that don't need it.
    if route == "GREETING":
        response = await handle_greeting(state)
    elif route == "SMALL_TALK":
        response = await handle_small_talk(state)
    elif route == "CONTROL":
        # Forwards the original intent so handle_control can tell whether this
        # is a cancellation or a plain yes/no acknowledgement.
        response = await handle_control(state, intent=intent)
    elif route == "EXPAND":
        response = await handle_expand(state)
    elif route == "BALANCE":
        response = await handle_balance(state)
    elif route == "USER_INFO":
        response = await handle_user_info(state)
    elif route == "TICKET":
        # Forwards intent so handle_ticket can bail out on CANCEL_FLOW at any step.
        response = await handle_ticket(state, intent=intent)
    elif route == "RAG":
        response = await handle_rag(state, vectorstore, system_prompt)
    else:
        response = "I'm sorry, I didn't understand that."

    # RAG and EXPAND nodes manage their own history entries.
    if route not in {"RAG", "EXPAND"}:
        state["memory"]["history"].append({"role": "user", "content": state["message"]})
        state["memory"]["history"].append({"role": "assistant", "content": response})

    return response, state