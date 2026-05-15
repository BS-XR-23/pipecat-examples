from typing import TypedDict, Optional, Dict, Any
from random import choice
import os
import re
import copy
from dotenv import load_dotenv
from ollama import AsyncClient
from loguru import logger
from db.database import SessionLocal
from services.rag_services.rag_service_hotel import get_rag_context
from utils.helpers import safe_parse_json, extract_phone_number, extract_date, extract_number, normalize_phone, extract_json
from utils.agent_logger import log_tool_call
from agents.hotel.agent_tools import (
    search_available_rooms,
    get_room_prices,
    make_booking,
    check_booking_status,
    cancel_booking,
)
from response import (
    GREETING_RESPONSES_HOTEL,
    SMALL_TALK_RESPONSES_HOTEL,
    SEARCH_ASK_CHECKIN,
    SEARCH_ASK_CHECKOUT,
    SEARCH_ASK_GUESTS,
    BOOKING_ASK_ROOM,
    BOOKING_ASK_NAME,
    BOOKING_ASK_PHONE,
    BOOKING_SUCCESS,
    CANCEL_ASK_BOOKING_ID,
    CANCEL_ASK_PHONE,
    CANCEL_CONFIRM,
    CANCEL_SUCCESS,
    CHECK_ASK_BOOKING_ID,
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


def safe_state(state: BotState) -> BotState:
    return copy.deepcopy(state)


def ensure_memory(state: BotState) -> BotState:
    state.setdefault("memory", {})
    memory = state["memory"]
    memory.setdefault("profile", {})
    memory.setdefault("history", [])
    memory.setdefault("booking", {})
    memory.setdefault("search", {})
    memory.setdefault("flow", {})

    flow = memory["flow"]
    flow.setdefault("active", None)
    flow.setdefault("step", None)
    flow.setdefault("expandable", False)
    flow.setdefault("last_expand_offer", False)
    # FIX (Bug #1 / resume): suspension slots so RAG interruptions don't
    # destroy an in-progress booking/search flow.
    flow.setdefault("suspended_flow", None)
    flow.setdefault("suspended_step", None)

    if not isinstance(state.get("intent"), str):
        state["intent"] = "RAG"

    return state


# ──────────────────────────────────────────────
# SECONDARY CANCEL-vs-RAG CHECK
# ──────────────────────────────────────────────
CANCEL_QUESTION_PROMPT = """
You are an intent router helper for a hotel booking chatbot.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{{"route_to_rag":<true|false>,"reason":"<short explanation>"}}

The user message may contain both a cancellation signal and a factual question.

Instructions:
- If the user is only asking to cancel or stop the current flow, return false.
- If the user is also asking a factual, informational question about hotel services,
  amenities, policies, rooms, or procedures, return true.
- Do not rely on keyword matching alone; use meaning and intent.
- If the message is ambiguous, prefer cancellation unless a clear informational
  request is present.

Active flow: {active_flow}

Conversation history:
{history}

User message:
{message}
"""


async def should_route_cancel_flow_to_rag(state: BotState) -> bool:
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
# INTENT DETECTION
# Changes from original:
#   1. RESUME_FLOW intent added — lets user explicitly re-join a suspended flow.
#   2. suspended_flow injected into the prompt so the LLM has full context.
#   3. active_flow + flow_step unchanged.
#   4. Confidence gating + secondary RAG check unchanged.
# ──────────────────────────────────────────────
INTENT_PROMPT = """
You are a routing system for a hotel booking chatbot.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{{"intent":"<INTENT>","confidence":<0.0-1.0>}}

Allowed intents:
- GREETING
- SEARCH_ROOMS
- MAKE_BOOKING
- CANCEL_BOOKING
- CHECK_BOOKING
- CHECK_PRICES
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
  Examples: "no", "don't", "do not", "forget it", "cancel", "skip",
            "never mind", "stop", "no thanks", "I changed my mind".
  Use CANCEL_FLOW whenever active_flow is not "None" and the user is pushing
  back, negating, or changing direction.

- RESUME_FLOW:
  The user wants to continue or go back to a previously suspended flow.
  Only applicable when suspended_flow is not "None".
  Examples: "let's continue", "go back to the booking", "resume",
            "continue where we left off", "back to the search",
            "ok let's keep going", "yes let's continue".

- GREETING:
  User says hello, hi, good morning, or starts a new conversation.

- SMALL_TALK:
  Casual non-hotel chat: thanks, bye, okay, got it, emotional filler.

- CONTROL:
  Short acknowledgements like yes, ok, okay, sure — ONLY when no active flow
  and no suspended flow, and the user is not cancelling anything.

- EXPAND:
  User wants more detail or clarification about the previous assistant answer.

- SEARCH_ROOMS:
  User wants to find or browse available rooms.
  Examples: "find me a room", "what rooms are available", "search for rooms".

- MAKE_BOOKING:
  User wants to CREATE a new reservation.
  Examples: "I want to book a room", "make a reservation", "reserve a deluxe room".

- CANCEL_BOOKING:
  User wants to CANCEL an existing booking.
  Examples: "cancel my booking", "I want to cancel my reservation".

- CHECK_BOOKING:
  User wants to CHECK status of an existing booking.
  Examples: "check my booking", "what's the status of my reservation".

- CHECK_PRICES:
  User wants to know room rates or pricing.
  Examples: "how much is a room", "what are the prices", "room rates".

- RAG:
  Informational questions about hotel policies, amenities, services, or FAQs.
  Examples: "what's check-in time", "do you have a pool", "is breakfast included",
            "tell me about facilities", "how do I request late checkout".
  ALWAYS use RAG when user asks "what", "how", "tell me", "do you", "explain"
  and the question is NOT about creating/cancelling/checking a specific booking.

Critical Rules:
- If active_flow is not "None" and the message contains negation words
  ("no", "don't", "do not", "not", "never", "cancel", "stop", "forget",
  "never mind", "skip") → ALWAYS return CANCEL_FLOW.
- If suspended_flow is not "None" and the user indicates they want to continue
  or resume → return RESUME_FLOW.
- If the message includes a factual or informational question, route to RAG.
- If active_flow is set and the message appears to both cancel the flow AND ask
  a question, use a secondary semantic check (the outer system handles it).
- Only use MAKE_BOOKING if the user is explicitly trying to CREATE a booking,
  not asking about the booking process.
- Only use CANCEL_BOOKING if the user is explicitly trying to CANCEL an existing
  booking, not asking about cancellation policies.

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

        flow = state["memory"]["flow"]
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
                "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
                "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND",
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
                if await should_route_cancel_flow_to_rag(state):
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
#   • CANCEL_FLOW still routes to "control" (handles graceful exit).
#   • FIX (Bug #1): when a genuine new intent overrides an active flow,
#     the active flow slots are cleared here so the destination node
#     starts clean instead of resuming a stale step.
# ──────────────────────────────────────────────
def router(state: BotState) -> str:
    intent = state.get("intent", "RAG")
    if not isinstance(intent, str):
        intent = "RAG"
    intent = intent.strip().upper()

    # Cancellation always wins — control_node handles the graceful exit
    if intent == "CANCEL_FLOW":
        return "control"

    # FIX (Bug #1 / resume): explicit resume intent
    if intent == "RESUME_FLOW":
        flow = state.get("memory", {}).get("flow", {})
        if flow.get("suspended_flow"):
            return "resume_flow"
        # Nothing suspended — treat as neutral continuation
        return "control"

    flow   = state.get("memory", {}).get("flow", {})
    active = flow.get("active")

    # Resume an in-progress flow for neutral/continuation intents only
    continuation_intents = {"CONTROL", "SMALL_TALK", "EXPAND"}
    if active and intent in continuation_intents:
        return active.lower()

    # FIX (Bug #1): a genuine new intent overrides the active flow —
    # clear the stale slots so the destination node starts fresh.
    if active and intent not in continuation_intents:
        logger.info(
            f"[ROUTER] New intent '{intent}' overrides active flow '{active}'. "
            "Clearing active flow."
        )
        flow["active"] = None
        flow["step"]   = None

    allowed_routes = {
        "GREETING":       "greeting",
        "SMALL_TALK":     "small_talk",
        "CONTROL":        "control",
        "SEARCH_ROOMS":   "search_rooms",
        "MAKE_BOOKING":   "make_booking",
        "CANCEL_BOOKING": "cancel_booking",
        "CHECK_BOOKING":  "check_booking",
        "CHECK_PRICES":   "check_prices",
        "RAG":            "rag",
        "EXPAND":         "expand",
    }
    return allowed_routes.get(intent, "rag")


# ──────────────────────────────────────────────
# RESUME FLOW NODE
# FIX (Bug #1 / resume): restores a suspended flow and immediately
# re-runs the appropriate handler so the user receives both a
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
        "MAKE_BOOKING": (
            "Sure! Let's continue with your booking. Here's where we left off:"
        ),
        "SEARCH_ROOMS": (
            "Of course! Resuming the room search."
        ),
        "CANCEL_BOOKING": (
            "Got it, let's continue with the cancellation."
        ),
        "CHECK_BOOKING": (
            "Sure, continuing with the booking status check."
        ),
    }
    intro = resume_messages.get(
        suspended_flow,
        "Sure, let's pick up where we left off."
    )

    # Re-dispatch to the correct handler and combine the messages
    if suspended_flow == "MAKE_BOOKING":
        state = await make_booking_node(state)
    elif suspended_flow == "SEARCH_ROOMS":
        state = await search_rooms_node(state)
    elif suspended_flow == "CANCEL_BOOKING":
        state = await cancel_booking_node(state)
    elif suspended_flow == "CHECK_BOOKING":
        state = await check_booking_node(state)
    else:
        state["response"] = "How can I help you?"
        return state

    # Prepend the resume intro to whatever the handler produced
    state["response"] = f"{intro}\n\n{state['response']}"
    return state


# ──────────────────────────────────────────────
# CONTROL NODE
# Handles CANCEL_FLOW exits, yes/no acknowledgements, and expand triggers.
# FIX (Bug #1 / cancel): now also clears suspended_flow/suspended_step
# so a full cancel wipes both active and suspended state.
# ──────────────────────────────────────────────
async def control_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]
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

        # Clear transient data for whichever flow(s) were running
        if active_flow in {"MAKE_BOOKING", "CANCEL_BOOKING"} or \
           suspended_flow in {"MAKE_BOOKING", "CANCEL_BOOKING"}:
            state["memory"]["booking"] = {}
        if active_flow == "SEARCH_ROOMS" or suspended_flow == "SEARCH_ROOMS":
            state["memory"]["search"] = {}

        # Report whichever flow was actually running
        reported_flow = active_flow or suspended_flow

        cancellation_messages = {
            "SEARCH_ROOMS": (
                "No problem, I've cancelled the room search. "
                "Feel free to ask if you'd like to search again or need anything else."
            ),
            "MAKE_BOOKING": (
                "Got it, I've cancelled the booking process. "
                "Let me know if you'd like to start over or need any help."
            ),
            "CANCEL_BOOKING": (
                "Alright, I've stopped the cancellation process. "
                "Your booking remains active. How else can I help you?"
            ),
            "CHECK_BOOKING": (
                "Sure, I've stopped checking the booking. "
                "Let me know if there's anything else I can help you with."
            ),
        }
        state["response"] = cancellation_messages.get(
            reported_flow,
            "Alright, I've cancelled that. How else can I help you?"
        )
        return state

    # ── Standard yes/no handling ───────────────────────────────────────────
    text = state["message"].lower().strip()

    if text in ["no", "nope"]:
        flow["expandable"]        = False
        flow["last_expand_offer"] = False
        state["response"] = "Alright! Let me know if there's anything else I can help with."
        return state

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            state["intent"] = "EXPAND"
            return await expand_node(state)
        state["response"] = "Sure! How can I help you?"
        return state

    state["response"] = "Got it! Let me know what you need."
    state["intent"]   = "CONTROL"
    return state


# ──────────────────────────────────────────────
# CHECK PRICES NODE  (stateless – no flow needed)
# ──────────────────────────────────────────────
async def check_prices_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))

    db = SessionLocal()
    try:
        result = get_room_prices(db)
        tool_status = "success" if "prices" in result else "failed"
        short_desc = (
            "Agent retrieved current room prices"
            if tool_status == "success"
            else "Agent failed to retrieve room prices"
        )
    except Exception as e:
        result = {"error": str(e)}
        tool_status = "error"
        short_desc = "Agent encountered an error fetching room prices"
    finally:
        db.close()

    await log_tool_call(
        vendor_id=VENDOR_ID,
        agent_name=state["agent_name"],
        agent_type="hotel",
        tool_name="get_room_prices",
        tool_status=tool_status,
        short_description=short_desc,
        user_identifier=state["session_id"],
        raw_tool_input={},
        raw_tool_output=result,
    )

    prices = result.get("prices", [])
    if not prices:
        state["response"] = "I couldn't retrieve room prices right now. Please contact our front desk."
        return state

    lines = [
        f"{p['type']}: {p['currency']} {p['price']}/night — {p['amenities']} (up to {p['capacity']} guests)"
        for p in prices
    ]
    state["response"] = "Here are our current room rates:\n\n" + "\n".join(lines)
    state["intent"]   = "CHECK_PRICES"
    return state


# ──────────────────────────────────────────────
# SEARCH ROOMS NODE
# ──────────────────────────────────────────────
async def _execute_search(state: BotState) -> BotState:
    flow   = state["memory"]["flow"]
    search = state["memory"]["search"]

    logger.info(
        f"[_execute_search] check_in={search.get('check_in')} "
        f"check_out={search.get('check_out')} guests={search.get('guests')}"
    )

    db = SessionLocal()
    try:
        result = search_available_rooms(
            db,
            search["check_in"],
            search["check_out"],
            int(search.get("guests", 1)),
        )
        tool_status = "success" if "rooms" in result else "failed"
        short_desc = (
            f"Agent searched available rooms: {search['check_in']} → {search['check_out']} "
            f"for {search.get('guests', 1)} guest(s), found {len(result.get('rooms', []))} result(s)"
            if tool_status == "success"
            else f"Agent failed to search rooms for {search['check_in']} → {search['check_out']}"
        )
    except Exception as e:
        result = {"error": str(e)}
        tool_status = "error"
        short_desc = "Agent encountered an error searching available rooms"
        logger.error(f"[_execute_search ERROR] {e}")
    finally:
        db.close()

    await log_tool_call(
        vendor_id=VENDOR_ID,
        agent_name=state["agent_name"],
        agent_type="hotel",
        tool_name="search_available_rooms",
        tool_status=tool_status,
        short_description=short_desc,
        user_identifier=state["session_id"],
        raw_tool_input={
            "check_in":  search.get("check_in"),
            "check_out": search.get("check_out"),
            "guests":    search.get("guests", 1),
        },
        raw_tool_output=result,
    )

    rooms = result.get("rooms", [])
    if not rooms:
        state["response"] = (
            f"No rooms available from {search['check_in']} to {search['check_out']} "
            f"for {search.get('guests', 1)} guest(s). Please try different dates."
        )
    else:
        lines = [
            f"Room Number: {r['room_number']}\n"
            f"Type: {r['type']}\n"
            f"Price: {r['currency']} {r['price']}/night\n"
            f"Capacity: {r['capacity']} guests\n"
            for r in rooms
        ]
        state["response"] = (
            f"Available rooms from {search['check_in']} to {search['check_out']} "
            f"for {search.get('guests', 1)} guest(s):\n\n"
            + "\n".join(lines)
            + "\n\nWould you like to book one of these rooms?"
        )

    flow["active"] = None
    flow["step"]   = None
    state["intent"] = "SEARCH_ROOMS"
    return state


async def search_rooms_node(state: BotState) -> BotState:
    state  = ensure_memory(state)
    flow   = state["memory"]["flow"]
    intent = state.get("intent", "").upper()

    # ── Bail out on any cancellation signal ───────────────────────────────
    if intent == "CANCEL_FLOW":
        flow["active"]         = None
        flow["step"]           = None
        flow["suspended_flow"] = None
        flow["suspended_step"] = None
        state["memory"]["search"] = {}
        state["response"] = (
            "No problem, I've cancelled the room search. "
            "Let me know if there's anything else I can help with."
        )
        return state

    # FIX (Bug #4): clear stale search state whenever we enter with no
    # active flow, regardless of what step was previously recorded.
    if not flow.get("active"):
        state["memory"]["search"] = {}
        flow["step"] = "ask_checkin"

    search = state["memory"]["search"]
    step   = flow.get("step") or "ask_checkin"

    if step == "ask_checkin":
        flow["active"] = "SEARCH_ROOMS"

        date = extract_date(state["message"])
        if date:
            search["check_in"] = date

        if search.get("check_in"):
            all_dates = re.findall(r"\d{4}-\d{2}-\d{2}", state["message"])
            for d in all_dates:
                if d != search["check_in"]:
                    search["check_out"] = d
                    break

        if not search.get("check_in"):
            flow["step"]      = "ask_checkin"
            state["response"] = choice(SEARCH_ASK_CHECKIN)
            return state

        if not search.get("check_out"):
            flow["step"]      = "ask_checkout"
            state["response"] = choice(SEARCH_ASK_CHECKOUT)
            return state

        flow["step"]      = "ask_guests"
        state["response"] = choice(SEARCH_ASK_GUESTS)
        return state

    if step == "ask_checkout":
        date = extract_date(state["message"])
        if not date:
            state["response"] = "Please provide your check-out date (e.g. 2025-06-05)."
            return state
        search["check_out"] = date
        flow["step"]        = "ask_guests"
        state["response"]   = choice(SEARCH_ASK_GUESTS)
        return state

    if step == "ask_guests":
        guests = extract_number(state["message"]) or 1
        search["guests"] = int(guests)
        logger.info(
            f"[ask_guests] message='{state['message']}' extracted={guests} search={search}"
        )
        return await _execute_search(state)

    # Fallback
    flow["active"] = "SEARCH_ROOMS"
    flow["step"]   = "ask_checkin"
    state["response"] = choice(SEARCH_ASK_CHECKIN)
    return state


# ──────────────────────────────────────────────
# MAKE BOOKING NODE
# ──────────────────────────────────────────────
def extract_all_booking_fields(message: str, booking: dict):
    if not booking.get("check_in"):
        date = extract_date(message)
        if date:
            booking["check_in"] = date

    if booking.get("check_in") and not booking.get("check_out"):
        all_dates = re.findall(r"\d{4}-\d{2}-\d{2}", message)
        for d in all_dates:
            if d != booking.get("check_in"):
                booking["check_out"] = d
                break
        if not booking.get("check_out"):
            all_dates_alt = re.findall(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", message)
            for match in all_dates_alt:
                d_fmt = f"{match[2]}-{match[1].zfill(2)}-{match[0].zfill(2)}"
                if d_fmt != booking.get("check_in"):
                    booking["check_out"] = d_fmt
                    break

    if not booking.get("guest_phone"):
        phone = extract_phone_number(message)
        if phone:
            booking["guest_phone"] = normalize_phone(phone)

    if not booking.get("room_number"):
        room_match = re.search(
            r"\broom\s*(?:number|no|#)?\s*(\d{1,4})\b", message, re.IGNORECASE
        )
        if room_match:
            booking["room_number"] = room_match.group(1)

    if not booking.get("guest_name"):
        name_match = re.search(
            r"(?:my(?:\s+full)?\s+name\s+is|i am|i'm|name[:\s]+)\s*"
            r"([A-Za-z]+(?:\s[A-Za-z]+)*?)(?:\s+and|\s+my|\s+phone|\s+number|,|$)",
            message,
            re.IGNORECASE,
        )
        if name_match:
            booking["guest_name"] = name_match.group(1).strip()
        else:
            plain_match = re.fullmatch(
                r"\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\s*", message.strip()
            )
            if plain_match:
                booking["guest_name"] = plain_match.group(1).strip()

    if not booking.get("guests_count"):
        msg_no_dates = re.sub(r"\d{4}-\d{2}-\d{2}", "", message)
        msg_no_dates = re.sub(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}", "", msg_no_dates)
        guests = extract_number(msg_no_dates)
        if guests:
            booking["guests_count"] = guests


def next_missing_step(booking: dict) -> str:
    if not booking.get("check_in"):
        return "ask_checkin"
    if not booking.get("check_out"):
        return "ask_checkout"
    if not booking.get("room_number"):
        return "ask_room"
    if not booking.get("guest_name"):
        return "ask_name"
    if not booking.get("guest_phone"):
        return "ask_phone"
    return "confirm"


async def make_booking_node(state: BotState) -> BotState:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]
    search  = state["memory"].get("search", {})
    intent  = state.get("intent", "").upper()

    # ── Bail out on any cancellation signal ───────────────────────────────
    if intent == "CANCEL_FLOW":
        flow["active"]         = None
        flow["step"]           = None
        flow["suspended_flow"] = None
        flow["suspended_step"] = None
        state["memory"]["booking"] = {}
        state["response"] = (
            "Got it, I've cancelled the booking process. "
            "Let me know if you'd like to start over or need any help."
        )
        return state

    current_step = flow.get("step") or "ask_checkin"

    # Carry over dates from a preceding search flow
    if search.get("check_in") and not booking.get("check_in"):
        booking["check_in"] = search["check_in"]
    if search.get("check_out") and not booking.get("check_out"):
        booking["check_out"] = search["check_out"]

    # FIX (Bug #2): skip field extraction at the confirm step — extracting from
    # "yes / confirm" can overwrite valid, already-collected booking data.
    if current_step != "confirm":
        extract_all_booking_fields(state["message"], booking)

    flow["active"] = "MAKE_BOOKING"
    next_step = next_missing_step(booking)

    if next_step == "ask_checkin":
        flow["step"]      = "ask_checkin"
        state["response"] = choice(SEARCH_ASK_CHECKIN)
        return state

    if next_step == "ask_checkout":
        flow["step"]      = "ask_checkout"
        state["response"] = choice(SEARCH_ASK_CHECKOUT)
        return state

    if next_step == "ask_room":
        if flow.get("step") == "ask_room":
            state["response"] = "Which room number would you like to book?"
            return state

        flow["step"] = "ask_room"

        if booking.get("check_in") and booking.get("check_out"):
            db = SessionLocal()
            try:
                availability = search_available_rooms(
                    db,
                    check_in=booking["check_in"],
                    check_out=booking["check_out"],
                    guests=booking.get("guests_count", 1),
                )
            finally:
                db.close()

            if "error" in availability or not availability.get("rooms"):
                state["response"] = (
                    f"Sorry, no rooms are available between {booking['check_in']} "
                    f"and {booking['check_out']}. Please try different dates."
                )
                booking.pop("check_in", None)
                booking.pop("check_out", None)
                flow["step"] = "ask_checkin"
                return state

            def format_amenities(amenities):
                if not amenities:
                    return "standard"
                if isinstance(amenities, str):
                    return amenities
                return ", ".join(amenities)

            room_lines = "\n".join(
                f"  • Room {r['room_number']} — {r['type']}, "
                f"{r['currency']} {r['price']}/night, "
                f"capacity: {r['capacity']}, "
                f"amenities: {format_amenities(r['amenities'])}"
                for r in availability["rooms"]
            )
            state["response"] = (
                f"Here are the available rooms for "
                f"{booking['check_in']} → {booking['check_out']}:\n\n"
                f"{room_lines}\n\n"
                f"Which room number would you like to book?"
            )
        else:
            state["response"] = choice(BOOKING_ASK_ROOM)

        return state

    if next_step == "ask_name":
        flow["step"]      = "ask_name"
        state["response"] = choice(BOOKING_ASK_NAME)
        return state

    if next_step == "ask_phone":
        flow["step"]      = "ask_phone"
        state["response"] = choice(BOOKING_ASK_PHONE)
        return state

    # ── Confirmation step ──────────────────────────────────────────────────
    if current_step != "confirm":
        flow["step"] = "confirm"
        state["response"] = (
            f"Here's your booking summary:\n\n"
            f"Name: {booking.get('guest_name')}\n"
            f"Room: {booking.get('room_number')}\n"
            f"Check-in: {booking.get('check_in')}\n"
            f"Check-out: {booking.get('check_out')}\n"
            f"Phone: {booking.get('guest_phone')}\n\n"
            f"Shall I confirm this booking? (yes / no)"
        )
        return state

    if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure", "proceed"]):
        if booking.get("guest_phone"):
            booking["guest_phone"] = normalize_phone(booking["guest_phone"])

        db = SessionLocal()
        try:
            result = make_booking(
                db,
                guest_name=booking.get("guest_name"),
                guest_phone=booking.get("guest_phone"),
                room_number=booking.get("room_number"),
                check_in=booking.get("check_in"),
                check_out=booking.get("check_out"),
                guests_count=booking.get("guests_count", 1),
            )
            tool_status = "failed" if "error" in result else "success"
            short_desc = (
                f"Agent booked room {booking.get('room_number')} for "
                f"{booking.get('guest_name')} "
                f"({booking.get('check_in')} → {booking.get('check_out')})"
                if tool_status == "success"
                else f"Agent failed to book room {booking.get('room_number')}: "
                     f"{result.get('error')}"
            )
        except Exception as e:
            result = {"error": str(e)}
            tool_status = "error"
            short_desc = (
                f"Agent encountered an error making a booking for "
                f"{booking.get('guest_name')}"
            )
        finally:
            db.close()

        await log_tool_call(
            vendor_id=VENDOR_ID,
            agent_name=state["agent_name"],
            agent_type="hotel",
            tool_name="make_booking",
            tool_status=tool_status,
            short_description=short_desc,
            user_identifier=booking.get("guest_phone") or state["session_id"],
            raw_tool_input={
                "guest_name":   booking.get("guest_name"),
                "guest_phone":  booking.get("guest_phone"),
                "room_number":  booking.get("room_number"),
                "check_in":     booking.get("check_in"),
                "check_out":    booking.get("check_out"),
                "guests_count": booking.get("guests_count", 1),
            },
            raw_tool_output=result,
        )

        if "error" in result:
            booking.pop("room_number", None)
            flow["step"]      = "ask_room"
            state["response"] = (
                f"{result['error']}\n\n"
                f"Please provide a different room number to continue."
            )
            return state

        booking_id = result.get("booking_id", "UNKNOWN")
        state["response"] = (
            choice(BOOKING_SUCCESS)
            + f" Your Booking ID is: {booking_id}. Please keep this for your records."
        )
    else:
        state["response"] = (
            "Booking cancelled. Let me know if there's anything else I can help with."
        )

    state["memory"]["booking"] = {}
    flow["active"] = None
    flow["step"]   = None
    state["intent"] = "MAKE_BOOKING"
    return state


# ──────────────────────────────────────────────
# CANCEL BOOKING NODE
# FIX (Bug #3): booking ID is extracted from the initial message before
# the step dispatcher runs, so data the user already provided is not lost.
# ──────────────────────────────────────────────
async def cancel_booking_node(state: BotState) -> BotState:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]
    intent  = state.get("intent", "").upper()

    # ── Bail out on any cancellation signal ───────────────────────────────
    if intent == "CANCEL_FLOW":
        flow["active"]         = None
        flow["step"]           = None
        flow["suspended_flow"] = None
        flow["suspended_step"] = None
        state["memory"]["booking"] = {}
        state["response"] = (
            "Alright, I've stopped the cancellation process. "
            "Your booking remains active. How else can I help you?"
        )
        return state

    # FIX (Bug #3): always attempt extraction first, before step dispatch,
    # so data embedded in the triggering message is never silently discarded.
    if not booking.get("guest_phone"):
        phone = extract_phone_number(state["message"])
        if phone:
            booking["guest_phone"] = normalize_phone(phone)

    if not booking.get("booking_id"):
        raw = state["message"].strip().upper()
        bid_match = re.search(r"[A-Z0-9\-]{6,}", raw)
        if bid_match:
            booking["booking_id"] = bid_match.group(0)

    step = flow.get("step") or "ask_booking_id"
    flow["active"] = "CANCEL_BOOKING"

    if step == "ask_booking_id":
        # FIX (Bug #3): if we already extracted a booking ID from this
        # message, skip the prompt and advance directly.
        if booking.get("booking_id"):
            if booking.get("guest_phone"):
                flow["step"]      = "confirm"
                state["response"] = choice(CANCEL_CONFIRM).format(
                    booking_id=booking.get("booking_id")
                )
            else:
                flow["step"]      = "ask_phone"
                state["response"] = choice(CANCEL_ASK_PHONE)
            return state

        flow["step"]      = "collect_booking_id"
        state["response"] = choice(CANCEL_ASK_BOOKING_ID)
        return state

    if step == "collect_booking_id":
        if not booking.get("booking_id"):
            raw = state["message"].strip().upper()
            bid_match = re.search(r"[A-Z0-9\-]{6,}", raw)
            booking["booking_id"] = bid_match.group(0) if bid_match else raw

        if booking.get("guest_phone"):
            flow["step"]      = "confirm"
            state["response"] = choice(CANCEL_CONFIRM).format(
                booking_id=booking.get("booking_id")
            )
            return state

        flow["step"]      = "ask_phone"
        state["response"] = choice(CANCEL_ASK_PHONE)
        return state

    if step == "ask_phone":
        phone = extract_phone_number(state["message"])
        if not phone:
            state["response"] = (
                "Please provide the valid phone number used when making this booking."
            )
            return state
        booking["guest_phone"] = normalize_phone(phone)
        flow["step"]           = "confirm"
        state["response"]      = choice(CANCEL_CONFIRM).format(
            booking_id=booking.get("booking_id")
        )
        return state

    if step == "confirm":
        if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure"]):
            db = SessionLocal()
            try:
                result = cancel_booking(
                    db, booking.get("booking_id"), booking.get("guest_phone")
                )
                tool_status = "failed" if "error" in result else "success"
                short_desc = (
                    f"Agent cancelled booking {booking.get('booking_id')} "
                    f"for phone {booking.get('guest_phone')}"
                    if tool_status == "success"
                    else f"Agent failed to cancel booking {booking.get('booking_id')}: "
                         f"{result.get('error')}"
                )
            except Exception as e:
                result = {"error": str(e)}
                tool_status = "error"
                short_desc = (
                    f"Agent encountered an error cancelling booking "
                    f"{booking.get('booking_id')}"
                )
            finally:
                db.close()

            await log_tool_call(
                vendor_id=VENDOR_ID,
                agent_name=state["agent_name"],
                agent_type="hotel",
                tool_name="cancel_booking",
                tool_status=tool_status,
                short_description=short_desc,
                user_identifier=booking.get("guest_phone") or state["session_id"],
                raw_tool_input={
                    "booking_id":  booking.get("booking_id"),
                    "guest_phone": booking.get("guest_phone"),
                },
                raw_tool_output=result,
            )

            if "error" in result:
                state["response"] = result["error"]
            else:
                state["response"] = choice(CANCEL_SUCCESS).format(
                    booking_id=booking.get("booking_id")
                )
        else:
            state["response"] = "Cancellation aborted. Your booking remains active."

        state["memory"]["booking"] = {}
        flow["active"]  = None
        flow["step"]    = None
        state["intent"] = "CANCEL_BOOKING"
        return state

    # Fallback
    flow["active"] = "CANCEL_BOOKING"
    flow["step"]   = "collect_booking_id"
    state["response"] = choice(CANCEL_ASK_BOOKING_ID)
    return state


# ──────────────────────────────────────────────
# CHECK BOOKING NODE
# ──────────────────────────────────────────────
async def check_booking_node(state: BotState) -> BotState:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    intent  = state.get("intent", "").upper()
    step    = flow.get("step") or "ask_booking_id"

    # ── Bail out on any cancellation signal ───────────────────────────────
    if intent == "CANCEL_FLOW":
        flow["active"]         = None
        flow["step"]           = None
        flow["suspended_flow"] = None
        flow["suspended_step"] = None
        state["response"] = (
            "Sure, I've stopped checking the booking. "
            "Let me know if there's anything else I can help you with."
        )
        return state

    if step == "ask_booking_id":
        flow["active"] = "CHECK_BOOKING"
        flow["step"]   = "collect_booking_id"
        state["response"] = choice(CHECK_ASK_BOOKING_ID)
        return state

    if step == "collect_booking_id":
        booking_id = state["message"].strip().upper()

        db = SessionLocal()
        try:
            result = check_booking_status(db, booking_id)
            tool_status = "failed" if "error" in result else "success"
            short_desc = (
                f"Agent checked status of booking {booking_id}: "
                f"{result.get('status', 'unknown').upper()}"
                if tool_status == "success"
                else f"Agent failed to find booking {booking_id}"
            )
        except Exception as e:
            result = {"error": str(e)}
            tool_status = "error"
            short_desc = f"Agent encountered an error checking booking {booking_id}"
        finally:
            db.close()

        await log_tool_call(
            vendor_id=VENDOR_ID,
            agent_name=state["agent_name"],
            agent_type="hotel",
            tool_name="check_booking_status",
            tool_status=tool_status,
            short_description=short_desc,
            user_identifier=state["session_id"],
            raw_tool_input={"booking_id": booking_id},
            raw_tool_output=result,
        )

        if "error" in result:
            state["response"] = (
                "I couldn't find that booking. Please double-check your booking ID."
            )
        else:
            state["response"] = (
                f"Booking details:\n\n"
                f"Booking ID: {result.get('booking_id')}\n"
                f"Guest: {result.get('guest_name')}\n"
                f"Phone: {result.get('guest_phone')}\n"
                f"Room: {result.get('room_type')} (Room {result.get('room_number')})\n"
                f"Check-in: {result.get('check_in')}\n"
                f"Check-out: {result.get('check_out')}\n"
                f"Guests: {result.get('guests_count')}\n"
                f"Total: {result.get('currency')} {result.get('total_price')}\n"
                f"Status: {result.get('status', '').upper()}"
            )

        flow["active"]  = None
        flow["step"]    = None
        state["intent"] = "CHECK_BOOKING"
        return state

    # Fallback
    flow["active"] = "CHECK_BOOKING"
    flow["step"]   = "collect_booking_id"
    state["response"] = choice(CHECK_ASK_BOOKING_ID)
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
        logger.error(f"[RAG VECTOR ERROR] {e}")
        state["response"] = (
            "I don't have that information right now. "
            "Please contact our front desk."
        )
        state["intent"] = "RAG_FAILED"
        return state

    if vectorstore is None:
        state["response"] = (
            "I don't have that information right now. "
            "Please contact our front desk."
        )
        state["intent"] = "RAG_FAILED"
        return state

    system_prompt = agent.get_system_prompt()
    docs, scores  = get_rag_context(state["message"], vectorstore)

    if not docs:
        state["response"] = (
            "I don't have that information right now. "
            "Please contact our front desk for assistance."
        )
        state["intent"] = "RAG_EMPTY"
        return state

    context = "\n\n".join(docs)
    if not context.strip():
        state["response"] = (
            "I don't have that information right now. "
            "Please contact our front desk for assistance."
        )
        state["intent"] = "RAG_EMPTY"
        return state

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

    prompt = f"""You are a human hotel customer support representative.

Answer naturally and conversationally like a real front-desk agent.

IMPORTANT:
- Use ONLY the provided context for factual accuracy
- Do NOT invent information
- Do NOT sound like a website, brochure, or policy document
- Speak in short, natural sentences
- Answer the user's exact question first
- Avoid marketing language and long paragraphs
- Sound helpful and human
- If the context doesn't cover the question, say: "Please contact our front desk for assistance."{suspended_note}

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
        state["response"] = "I couldn't process your request. Please try again."
        state["intent"]   = "LLM_FAILED"
        return state

    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": answer})
    state["memory"]["flow"]["expandable"]        = True
    state["memory"]["flow"]["last_expand_offer"] = True

    state["response"] = answer
    state["intent"]   = "RAG_SUCCESS"
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
        prompt=f"""Expand the following answer clearly and in detail.

PREVIOUS ANSWER:
{last_answer}

USER REQUEST:
{state['message']}

RULES:
- Only expand on the given answer.
- Do NOT use external knowledge.
- Sound conversational and helpful.

EXPANDED ANSWER:""",
    )

    expanded = res.response.strip()
    flow["expandable"]        = False
    flow["last_expand_offer"] = False
    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": expanded})
    state["response"] = expanded
    state["intent"]   = "EXPAND"
    return state


# ──────────────────────────────────────────────
# GREETING / SMALL TALK NODES
# ──────────────────────────────────────────────
async def greeting_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(GREETING_RESPONSES_HOTEL)
    state["intent"]   = "GREETING"
    return state


async def small_talk_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(SMALL_TALK_RESPONSES_HOTEL)
    state["intent"]   = "SMALL_TALK"
    return state


# ──────────────────────────────────────────────
# GRAPH
# Changes from original:
#   • "resume_flow" node added and wired into the conditional edges.
#   • Router maps RESUME_FLOW → "resume_flow" (or "control" if nothing suspended).
# ──────────────────────────────────────────────
def build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(BotState)

    graph.add_node("detect",        detect_intent)
    graph.add_node("greeting",      greeting_node)
    graph.add_node("small_talk",    small_talk_node)
    graph.add_node("control",       control_node)
    graph.add_node("resume_flow",   resume_flow_node)
    graph.add_node("search_rooms",  search_rooms_node)
    graph.add_node("make_booking",  make_booking_node)
    graph.add_node("cancel_booking", cancel_booking_node)
    graph.add_node("check_booking", check_booking_node)
    graph.add_node("check_prices",  check_prices_node)
    graph.add_node("rag",           rag_node)
    graph.add_node("expand",        expand_node)

    graph.add_edge(START, "detect")
    graph.set_entry_point("detect")

    graph.add_conditional_edges(
        "detect",
        router,
        {
            "GREETING":       "greeting",
            "SMALL_TALK":     "small_talk",
            "CONTROL":        "control",
            "RESUME_FLOW":    "resume_flow",
            "SEARCH_ROOMS":   "search_rooms",
            "MAKE_BOOKING":   "make_booking",
            "CANCEL_BOOKING": "cancel_booking",
            "CHECK_BOOKING":  "check_booking",
            "CHECK_PRICES":   "check_prices",
            "RAG":            "rag",
            "EXPAND":         "expand",
        },
    )

    for node in [
        "greeting",
        "small_talk",
        "control",
        "resume_flow",
        "search_rooms",
        "make_booking",
        "cancel_booking",
        "check_booking",
        "check_prices",
        "rag",
        "expand",
    ]:
        graph.add_edge(node, END)

    return graph.compile()