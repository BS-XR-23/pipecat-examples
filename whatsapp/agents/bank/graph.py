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
    THANK_YOU_RESPONSES,
)


load_dotenv()
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))


# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
class BotState(TypedDict):
    session_id: str
    message: str
    agent_name: str
    agent: Any
    intent: Optional[str]
    response: Optional[str]
    memory: Dict[str, Any]


ollama_client = AsyncClient()


def safe_state(state: BotState) -> BotState:
    return copy.deepcopy(state)


def ensure_memory(state: BotState) -> BotState:
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
    # FIX (Bug #1 / resume): suspension slots so RAG interruptions don't
    # destroy an in-progress ticket/balance/user_info flow.
    flow.setdefault("suspended_flow", None)
    flow.setdefault("suspended_step", None)

    if not isinstance(state.get("intent"), str):
        state["intent"] = "RAG"

    return state


# ──────────────────────────────────────────────
# SECONDARY CANCEL-vs-RAG CHECK
# FIX (ordering): prompt constant defined BEFORE the function that uses it,
# eliminating the latent NameError present in the original.
# ──────────────────────────────────────────────
CANCEL_QUESTION_PROMPT = """
You are an intent router helper for a banking chatbot.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{{"route_to_rag":<true|false>,"reason":"<short explanation>"}}

The user message may contain both a cancellation signal and a factual question.

Instructions:
- If the user is only asking to cancel or stop the current flow, return false.
- If the user is also asking a factual, informational question about banking products,
  services, policies, or procedures, return true.
- Do not rely on keyword matching alone; use meaning and intent.
- If the message is ambiguous, prefer cancellation unless a clear informational
  request is present.

Active flow: {active_flow}

Conversation history:
{history}

User message:
{message}
"""


async def should_route_cancel_flow_to_rag_state(state: BotState) -> bool:
    try:
        flow = state["memory"]["flow"]
        active_flow = flow.get("active") or "None"
        history = state["memory"]["history"][-6:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}" for h in history
        )

        prompt = CANCEL_QUESTION_PROMPT.format(
            active_flow=active_flow,
            history=history_text,
            message=state.get("message", ""),
        )

        client = AsyncClient()
        res = await client.generate(
            model="gemma4:e2b",
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.9},
        )

        raw = res.response.strip()
        logger.info(f"[CANCEL QUESTION RAW] {raw}")

        data = safe_parse_json(raw)
        if not isinstance(data, dict):
            data = extract_json(raw)
        if not isinstance(data, dict):
            logger.warning(
                "[CANCEL QUESTION PARSE] expected JSON object, got %s",
                type(data).__name__,
            )
            return False

        return bool(data.get("route_to_rag", False))

    except Exception as e:
        logger.warning(f"[CANCEL QUESTION NODE ERROR] {e}")
        return False


# ──────────────────────────────────────────────
# INTENT DETECTION
# Changes from original:
#   1. RESUME_FLOW intent added — lets user explicitly re-join a suspended flow.
#   2. suspended_flow injected into the prompt for full LLM context.
#   3. active_flow, flow_step, confidence gating, and secondary RAG check
#      are all unchanged from the improved original.
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

- CANCEL_FLOW:
  The user is EXPLICITLY rejecting, cancelling, or opting out of the current
  active flow or action. This takes highest priority over all other intents
  when an active flow exists.
  Examples: "no", "don't", "do not", "forget it", "cancel", "skip", "never mind",
            "I don't want a ticket", "do not create a ticket", "stop", "no thanks".
  Use CANCEL_FLOW whenever active_flow is not "None" and the user is pushing
  back, negating, or changing direction.

- RESUME_FLOW:
  The user wants to continue or go back to a previously suspended flow.
  Only applicable when suspended_flow is not "None".
  Examples: "let's continue", "go back to the ticket", "resume",
            "continue where we left off", "yes let's keep going", "ok continue".

- GREETING:
  User is saying hello, hi, good morning, good evening, or starting conversation.

- SMALL_TALK:
  Casual non-banking chat, such as thanks, bye, okay, got it, casual replies,
  or emotional filler.

- CONTROL:
  Short acknowledgements like yes, ok, okay, sure (ONLY when no active flow,
  no suspended flow, and the user is not cancelling anything).

- EXPAND:
  User wants more detail or clarification of the previous assistant answer.

- BALANCE:
  Asking about account balance, available funds, or money inquiry.

- USER_INFO:
  Asking for profile, account details, email, or personal info.

- TICKET:
  User is PERSONALLY experiencing a problem RIGHT NOW and wants to report it.
  Examples: "my card is blocked", "I can't login", "my transaction failed".
  NOT TICKET: informational questions about complaint processes or requirements.

- RAG:
  Any informational or knowledge-based question about banking services, policies,
  procedures, or requirements.
  Examples: "what documents do I need", "how do I submit a complaint",
            "tell me about loans offered by ABB bank".
  ALWAYS use RAG when the user asks "what", "how", "tell me", "explain",
  "what are the requirements", "tell me about".

Critical Rules:
- If active_flow is not "None" and the message contains negation words
  ("no", "don't", "do not", "not", "never", "cancel", "stop", "forget",
  "never mind", "skip") → ALWAYS return CANCEL_FLOW.
- If suspended_flow is not "None" and the user indicates they want to continue
  or resume → return RESUME_FLOW.
- If the message includes a factual or informational question, route to RAG.
- If active_flow is set and the message appears to both cancel the flow and ask
  a question, use a secondary semantic check (the outer system handles it).
- Only use TICKET if the user is personally reporting an active issue,
  NOT asking about a process.

Conversation history (last 6 turns):
{history}

User message:
{message}
"""


async def detect_intent(state: BotState) -> BotState:
    state = ensure_memory(state)
    state["intent"] = "RAG"  # safe default

    try:
        client = AsyncClient()

        flow           = state["memory"]["flow"]
        active_flow    = flow.get("active")         or "None"
        flow_step      = flow.get("step")           or "None"
        suspended_flow = flow.get("suspended_flow") or "None"

        history = state["memory"]["history"][-6:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}" for h in history
        )

        prompt = INTENT_PROMPT.format(
            active_flow    = active_flow,
            flow_step      = flow_step,
            suspended_flow = suspended_flow,
            history        = history_text,
            message        = state["message"],
        )

        res = await client.generate(
            model="gemma4:e2b",
            prompt=prompt,
            options={"temperature": 0, "top_p": 0.9},
        )

        raw = res.response.strip()
        logger.info(f"[INTENT RAW] {raw}")

        data = safe_parse_json(raw)

        if not data:
            logger.warning(f"[INTENT FALLBACK – no data] raw: {raw}")
            state["intent"] = "RAG"
        else:
            allowed = {
                "GREETING", "BALANCE", "USER_INFO",
                "TICKET", "RAG", "EXPAND",
                "CONTROL", "SMALL_TALK", "CANCEL_FLOW", "RESUME_FLOW",
            }

            raw_intent = str(data.get("intent", "RAG")).strip().upper()
            confidence = float(data.get("confidence", 1.0))

            if confidence < 0.5:
                logger.warning(
                    f"[INTENT LOW CONFIDENCE] {raw_intent} @ {confidence:.2f}, "
                    "falling back to RAG"
                )
                state["intent"] = "RAG"
            else:
                state["intent"] = raw_intent if raw_intent in allowed else "RAG"

            logger.info(
                f"[INTENT RESOLVED] {state['intent']} "
                f"(confidence={confidence:.2f})"
            )

            # Secondary check: CANCEL_FLOW that also contains a factual question
            # → upgrade to RAG so the user gets their answer first.
            if state["intent"] == "CANCEL_FLOW" and active_flow != "None":
                if await should_route_cancel_flow_to_rag_state(state):
                    logger.info(
                        "[INTENT] CANCEL_FLOW message also contains an informational "
                        "request; routing to RAG"
                    )
                    state["intent"] = "RAG"

    except Exception as e:
        logger.warning(f"[INTENT NODE ERROR] {e}, falling back to RAG")
        state["intent"] = "RAG"

    return state


# ──────────────────────────────────────────────
# ROUTER
# Changes from original:
#   • RESUME_FLOW routes to the new "resume_flow" node.
#   • CANCEL_FLOW still routes to "CONTROL" (graceful exit).
#   • FIX (Bug #1): when a genuine new intent overrides an active flow,
#     the active flow slots are cleared here so the destination node
#     starts clean and doesn't resume a stale step.
# ──────────────────────────────────────────────
def router(state: BotState) -> str:
    intent = state.get("intent", "RAG")
    if not isinstance(intent, str):
        intent = "RAG"
    intent = intent.strip().upper()

    # Cancellation always wins
    if intent == "CANCEL_FLOW":
        return "CONTROL"

    # FIX (Bug #1 / resume): explicit resume intent
    if intent == "RESUME_FLOW":
        flow = state.get("memory", {}).get("flow", {})
        if flow.get("suspended_flow"):
            return "RESUME_FLOW"
        # Nothing suspended — treat as neutral continuation
        return "CONTROL"

    flow   = state.get("memory", {}).get("flow", {})
    active = flow.get("active")

    # Resume an in-progress flow for neutral/continuation intents only
    continuation_intents = {"CONTROL", "SMALL_TALK", "EXPAND"}
    if active and intent in continuation_intents:
        return active

    # FIX (Bug #1): a genuine new intent overrides the active flow —
    # clear the stale slots so the destination node starts fresh.
    if active and intent not in continuation_intents:
        logger.info(
            f"[ROUTER] New intent '{intent}' overrides active flow '{active}'. "
            "Clearing active flow."
        )
        flow["active"] = None
        flow["step"]   = None

    allowed = {
        "GREETING", "BALANCE", "USER_INFO",
        "TICKET", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK", "CANCEL_FLOW",
    }
    return intent if intent in allowed else "RAG"


# ──────────────────────────────────────────────
# RESUME FLOW NODE
# FIX (Bug #1 / resume): restores a suspended flow and immediately
# re-dispatches to the correct handler so the user receives both a
# "resuming" message and the next step prompt in one turn.
# ──────────────────────────────────────────────
async def resume_flow_node(state: BotState) -> BotState:
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
        f"[RESUME_FLOW] Restored flow='{suspended_flow}' step='{suspended_step}'"
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
        state = await ticket_node(state)
    elif suspended_flow == "BALANCE":
        state = await balance_node(state)
    elif suspended_flow == "USER_INFO":
        state = await user_info_node(state)
    else:
        state["response"] = "How can I help you?"
        return state

    state["response"] = f"{intro}\n\n{state['response']}"
    return state


# ──────────────────────────────────────────────
# CONTROL NODE
# Changes from original:
#   • FIX (Bug #1 / cancel): now clears suspended_flow/suspended_step on
#     CANCEL_FLOW, so a full cancel truly wipes all flow state.
#   • Also wipes ticket data when a suspended TICKET flow is cancelled.
#   • reported_flow pattern used so the cancellation message is accurate
#     whether the flow was active or suspended.
# ──────────────────────────────────────────────
async def control_node(state: BotState) -> BotState:
    state  = ensure_memory(safe_state(state))
    flow   = state["memory"]["flow"]
    intent = state.get("intent", "CONTROL").upper()

    # ── Cancellation exit ──────────────────────────────────────────────────
    if intent == "CANCEL_FLOW":
        active_flow    = flow.get("active")
        suspended_flow = flow.get("suspended_flow")

        flow["active"]         = None
        flow["step"]           = None
        flow["suspended_flow"] = None
        flow["suspended_step"] = None
        flow["expandable"]        = False
        flow["last_expand_offer"] = False

        # Clear ticket data for whichever flow was running
        reported_flow = active_flow or suspended_flow
        if reported_flow == "TICKET":
            state["memory"]["ticket"] = {}

        if reported_flow == "TICKET":
            state["response"] = (
                "No problem, I won't raise a ticket. "
                "Let me know if there's anything else I can help you with."
            )
        else:
            state["response"] = "Alright, I've cancelled that. How else can I help you?"

        return state

    # ── Standard yes/no handling ───────────────────────────────────────────
    text = state["message"].lower().strip()

    if text in ["no", "nope"]:
        flow["expandable"]        = False
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


# ──────────────────────────────────────────────
# TICKET NODE
# Changes from original:
#   • FIX (Bug #6): safe_state() deep-copy added at entry, consistent with
#     rag_node and other nodes.
#   • CANCEL_FLOW bail-out now also clears suspended_flow/suspended_step.
#   • No Bug #2 equivalent here — phone extraction only runs at collect_phone
#     step, which is correct.
# ──────────────────────────────────────────────
async def ticket_node(state: BotState) -> BotState:
    state  = ensure_memory(safe_state(state))
    flow   = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]
    intent = state.get("intent", "").upper()

    # ── Bail out immediately on any cancellation signal ────────────────────
    if intent == "CANCEL_FLOW":
        flow["active"]         = None
        flow["step"]           = None
        flow["suspended_flow"] = None
        flow["suspended_step"] = None
        state["memory"]["ticket"] = {}
        state["response"] = (
            "No problem, I won't raise a ticket. "
            "Let me know if there's anything else I can help you with."
        )
        return state

    step = flow.get("step") or "confirm"

    # ── Step: initial confirmation ask ─────────────────────────────────────
    if step == "confirm":
        state["response"] = choice(TICKET_CONFIRM_INTENT)
        flow["active"] = "TICKET"
        flow["step"]   = "await_confirmation"
        return state

    # ── Step: interpret user's yes/no/human-agent response ─────────────────
    if step == "await_confirmation":
        text_lower = state["message"].lower()

        if any(w in text_lower for w in ["human", "agent", "representative", "person"]):
            flow["active"]         = None
            flow["step"]           = None
            flow["suspended_flow"] = None
            flow["suspended_step"] = None
            state["memory"]["ticket"] = {}
            state["response"] = choice(HUMAN_HANDOFF)
            return state

        affirmative_intents = {"CONTROL", "SMALL_TALK"}
        positive_words = {"yes", "yeah", "ok", "okay", "sure", "please", "create"}

        is_affirmative = (
            intent == "TICKET"
            or (
                intent in affirmative_intents
                and any(w in text_lower.split() for w in positive_words)
            )
        )

        if is_affirmative:
            flow["step"]      = "collect_issue"
            state["response"] = choice(TICKET_ASK_ISSUE)
        else:
            flow["active"]         = None
            flow["step"]           = None
            flow["suspended_flow"] = None
            flow["suspended_step"] = None
            state["memory"]["ticket"] = {}
            state["response"] = "Alright, let me know how else I can help you."

        return state

    # ── Step: collect issue description ────────────────────────────────────
    if step == "collect_issue":
        ticket["issue"] = state["message"]
        flow["step"]    = "collect_phone"
        state["response"] = choice(TICKET_ASK_PHONE)
        return state

    # ── Step: collect phone and create ticket ──────────────────────────────
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
            agent_name=state["agent_name"],
            agent_type="bank",
            tool_name="create_ticket",
            tool_status=tool_status,
            short_description=short_desc,
            user_identifier=phone,
            raw_tool_input={
                "phone":             phone,
                "category":          data.get("category"),
                "short_description": data.get("short_description"),
            },
            raw_tool_output=result,
        )

        flow["active"]         = None
        flow["step"]           = None
        flow["suspended_flow"] = None
        flow["suspended_step"] = None
        ticket.clear()

        if "error" in result:
            state["response"] = (
                "I'm sorry, I couldn't create a support ticket at this time. "
                "Please contact ABB support directly at 937."
            )
        else:
            ticket_id = result.get("ticket_id", "UNKNOWN")
            state["response"] = choice(TICKET_FINAL) + f" Ticket ID: {ticket_id}"

        return state

    # Fallback
    state["response"] = (
        "I understand you're facing an issue. "
        "Would you like me to raise a support ticket?"
    )
    flow["step"] = None
    return state


# ──────────────────────────────────────────────
# BALANCE NODE
# FIX (Bug #6): safe_state() deep-copy added for consistency.
# Phone extraction from the initial message was already correct (no Bug #3).
# ──────────────────────────────────────────────
async def balance_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
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
    flow["step"]   = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(BALANCE_SUCCESS).format(
        balance=result["balance"], currency=result["currency"]
    )
    state.setdefault("intent", "BALANCE")
    return state


# ──────────────────────────────────────────────
# USER INFO NODE
# FIX (Bug #6): safe_state() deep-copy added for consistency.
# ──────────────────────────────────────────────
async def user_info_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
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
    flow["step"]   = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(USER_INFO_SUCCESS).format(
        name=result["user_name"], email=result["email"]
    )
    state.setdefault("intent", "USER_INFO")
    return state


# ──────────────────────────────────────────────
# RAG NODE
# FIX (Bug #1 / resume): suspends the active flow instead of destroying it,
# so the user can explicitly resume after getting their question answered.
# ──────────────────────────────────────────────
async def rag_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    agent = state["agent"]
    flow  = state["memory"]["flow"]

    if flow.get("active"):
        logger.info(
            f"[RAG] Suspending active flow '{flow['active']}' "
            "to answer informational question. Data preserved for resume."
        )
        flow["suspended_flow"] = flow["active"]
        flow["suspended_step"] = flow["step"]
        flow["active"] = None
        flow["step"]   = None

    try:
        vectorstore = agent.get_vectorstore()
    except Exception as e:
        logger.error(f"[VECTORSTORE ERROR] {e}")
        state["response"] = "Please contact ABB support at 937."
        state["intent"]   = "RAG_FAILED"
        return state

    if vectorstore is None:
        state["response"] = "Please contact ABB support at 937."
        state["intent"]   = "RAG_FAILED"
        return state

    system_prompt = agent.get_system_prompt()
    docs, scores  = get_rag_context(state["message"], vectorstore)

    if not docs:
        state["response"] = "Please contact ABB support at 937."
        state["intent"]   = "RAG_EMPTY"
        return state

    context = "\n\n".join(docs)
    if not context.strip():
        state["response"] = "Please contact ABB support at 937."
        state["intent"]   = "RAG_EMPTY"
        return state

    history = state["memory"]["history"][-8:]
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}" for h in history
    )

    # Let the LLM know there is a suspended flow so it can offer to resume
    suspended_note = ""
    if flow.get("suspended_flow"):
        suspended_note = (
            f"\n\nNote: the user has a suspended '{flow['suspended_flow']}' flow. "
            "After answering their question, naturally offer to continue it if appropriate."
        )

    prompt = f"""
You are a human customer support representative for ABB.

Answer naturally and conversationally like a real call center agent.

IMPORTANT:
- Use ONLY provided context for factual accuracy. However, you MAY generalize within
  the same product family (e.g., loan types, banking products) when terminology differs
  but meaning is equivalent
- Do NOT invent information
- Do NOT sound like a website, brochure, advertisement, or policy document
- Speak in short, natural sentences
- Answer the user's exact question first
- If the context contains eligibility criteria and the user asks about documents,
  separate them clearly
- Preserve all factual requirements and steps from the context
- Do NOT copy large chunks verbatim unless necessary
- Avoid marketing language
- Avoid long paragraphs
- Sound helpful and human
- If the context doesn't cover the question, say: "Please contact ABB support at 937."{suspended_note}

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{state['message']}

ASSISTANT:"""

    client = AsyncClient()
    try:
        res = await client.generate(
            model="gemma4:e4b",
            prompt=prompt,
            system=system_prompt,
            options={"temperature": 0.2, "top_p": 0.9},
        )
        answer = res.response.strip()
    except Exception as e:
        logger.error(f"[LLM ERROR] {e}")
        state["response"] = "Please contact ABB support at 937."
        state["intent"]   = "LLM_FAILED"
        return state

    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": answer})
    state["memory"]["flow"]["expandable"]        = True
    state["memory"]["flow"]["last_expand_offer"] = True

    state["response"] = answer
    state.setdefault("intent", "RAG")
    return state


# ──────────────────────────────────────────────
# EXPAND NODE
# ──────────────────────────────────────────────
async def expand_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    agent         = state["agent"]
    system_prompt = agent.get_system_prompt()

    last_answer = next(
        (m["content"] for m in reversed(state["memory"]["history"]) if m["role"] == "assistant"),
        "",
    )

    client = AsyncClient()
    res = await client.generate(
        model="gemma4:e4b",
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

ASSISTANT:""",
    )

    expanded = res.response.strip()

    flow["expandable"]        = False
    flow["last_expand_offer"] = False

    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": expanded})

    state["response"] = expanded
    state.setdefault("intent", "EXPAND")
    return state


# ──────────────────────────────────────────────
# GREETING / SMALL TALK NODES
# ──────────────────────────────────────────────
async def greeting_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(GREETING_RESPONSES)
    state.setdefault("intent", "GREETING")
    return state


async def small_talk_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(THANK_YOU_RESPONSES)
    state.setdefault("intent", "SMALL_TALK")
    return state


# ──────────────────────────────────────────────
# GRAPH
# Changes from original:
#   • "RESUME_FLOW" node added and wired into conditional edges.
#   • Router maps RESUME_FLOW → "RESUME_FLOW" (or "CONTROL" if nothing suspended).
# ──────────────────────────────────────────────
def build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(BotState)

    graph.add_node("detect",      detect_intent)
    graph.add_node("control",     control_node)
    graph.add_node("greeting",    greeting_node)
    graph.add_node("small_talk",  small_talk_node)
    graph.add_node("balance",     balance_node)
    graph.add_node("user_info",   user_info_node)
    graph.add_node("ticket",      ticket_node)
    graph.add_node("rag",         rag_node)
    graph.add_node("expand",      expand_node)
    graph.add_node("resume_flow", resume_flow_node)

    graph.add_edge(START, "detect")
    graph.set_entry_point("detect")

    graph.add_conditional_edges(
        "detect",
        router,
        {
            "GREETING":    "greeting",
            "BALANCE":     "balance",
            "USER_INFO":   "user_info",
            "TICKET":      "ticket",
            "SMALL_TALK":  "small_talk",
            "CONTROL":     "control",
            "RAG":         "rag",
            "EXPAND":      "expand",
            "RESUME_FLOW": "resume_flow",
        },
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
        "resume_flow",
    ]:
        graph.add_edge(node, END)

    return graph.compile()