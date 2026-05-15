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
from utils.helpers import safe_parse_json, extract_phone_number, normalize_phone, extract_json
from agents.hotel.agent_tools import (
    search_available_rooms,
    get_room_prices,
    make_booking,
    check_booking_status,
    cancel_booking,
)
from response import (
    GREETING_RESPONSES_HOTEL,
    THANK_YOU_RESPONSES,
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
VENDOR_ID           = int(os.getenv("VENDOR_ID", "0"))
OLLAMA_INTENT_MODEL = os.getenv("OLLAMA_INTENT_MODEL", "gemma4:e2b")
OLLAMA_RAG_MODEL    = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e4b")
RAG_HISTORY_LIMIT   = int(os.getenv("RAG_HISTORY_LIMIT", "6"))
RAG_K               = int(os.getenv("RAG_K", "3"))

ollama_client = AsyncClient()


# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
class VoiceState(TypedDict):
    session_id: str
    message: str
    memory: Dict[str, Any]


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
    memory.setdefault("booking", {})
    memory.setdefault("search", {})
    memory.setdefault("flow", {})

    flow = memory["flow"]
    flow.setdefault("active", None)
    flow.setdefault("step", None)
    flow.setdefault("expandable", False)
    flow.setdefault("last_expand_offer", False)
    # FIX (Bug #1 / resume): suspension slots for mid-flow RAG interruptions
    flow.setdefault("suspended_flow", None)
    flow.setdefault("suspended_step", None)

    return state


# ──────────────────────────────────────────────
# DATE / NUMBER / FIELD EXTRACTION
# ──────────────────────────────────────────────
_DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_DMY = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b")
_NUMBER   = re.compile(r"\b([1-9])\b")


def extract_date(text: str) -> Optional[str]:
    m = _DATE_ISO.search(text)
    if m:
        return m.group(1)
    m = _DATE_DMY.search(text)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return None


def extract_all_dates(text: str) -> list[str]:
    iso = _DATE_ISO.findall(text)
    dmy = [
        f"{m[2]}-{m[1].zfill(2)}-{m[0].zfill(2)}"
        for m in _DATE_DMY.findall(text)
    ]
    return iso + dmy


def extract_number(text: str) -> Optional[int]:
    clean = _DATE_ISO.sub("", text)
    clean = _DATE_DMY.sub("", clean)
    m = _NUMBER.search(clean)
    return int(m.group(1)) if m else None


def extract_room_number(text: str) -> Optional[str]:
    m = re.search(r"\b(?:room\s*)?([A-Za-z]?\d{1,4}[A-Za-z]?)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def extract_name(text: str) -> Optional[str]:
    m = re.search(
        r"(?:my name is|i am|i'm|name[:\s]+)\s*([A-Za-z]+(?:\s[A-Za-z]+)*?)"
        r"(?:\s+and|\s+my|\s+phone|\s+number|,|$)",
        text, re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def extract_all_booking_fields(message: str, booking: dict):
    """
    Extract and fill any missing booking fields from the message.
    Should NOT be called when the flow is already at the confirm step.
    (Bug #2 fix: guard in handle_make_booking prevents calling this at confirm.)
    """
    dates = extract_all_dates(message)

    if not booking.get("check_in") and dates:
        booking["check_in"] = dates[0]

    if booking.get("check_in") and not booking.get("check_out") and len(dates) >= 2:
        for d in dates:
            if d != booking["check_in"]:
                booking["check_out"] = d
                break

    if not booking.get("guest_phone"):
        phone = extract_phone_number(message)
        if phone:
            booking["guest_phone"] = normalize_phone(phone)

    if not booking.get("room_number"):
        room = extract_room_number(message)
        if room:
            booking["room_number"] = room

    if not booking.get("guest_name"):
        name = extract_name(message)
        if name:
            booking["guest_name"] = name

    if not booking.get("guests_count"):
        n = extract_number(message)
        if n:
            booking["guests_count"] = n


def next_missing_booking_step(booking: dict) -> str:
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


async def should_route_cancel_flow_to_rag(state: VoiceState) -> bool:
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

        res = await ollama_client.generate(
            model=OLLAMA_INTENT_MODEL,
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
#   1. RESUME_FLOW intent added so users can explicitly re-join a suspended flow.
#   2. active_flow + flow_step + suspended_flow injected for full context.
#   3. Confidence gating unchanged (< 0.5 → RAG).
#   4. Secondary RAG check for CANCEL_FLOW unchanged.
# ──────────────────────────────────────────────
INTENT_PROMPT = """
You are a routing system for a hotel booking voice chatbot.

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
  active flow. This takes highest priority over all other intents when an
  active flow exists.
  Examples: "no", "don't", "do not", "forget it", "cancel", "skip",
            "never mind", "stop", "no thanks", "I changed my mind".
  Use CANCEL_FLOW whenever active_flow is not "None" and the user is pushing
  back, negating, or changing direction.

- RESUME_FLOW:
  The user wants to continue or go back to a previously suspended flow.
  Only applicable when suspended_flow is not "None".
  Examples: "let's continue", "go back to the booking", "continue where we left off",
            "resume", "back to the search", "ok let's keep going", "yes let's continue".

- GREETING:
  User says hello, hi, good morning, or starts a new conversation.

- SMALL_TALK:
  Casual replies like thanks, bye, okay, got it, emotional filler.

- CONTROL:
  Short acknowledgements like yes, ok, okay, sure — ONLY when no active flow
  and no suspended flow, and the user is not cancelling anything.

- EXPAND:
  User wants more detail about the previous assistant answer.

- SEARCH_ROOMS:
  User wants to find or browse available rooms.
  Examples: "find me a room", "what rooms are available", "search for rooms".

- MAKE_BOOKING:
  User wants to CREATE a new reservation.
  Examples: "I want to book a room", "make a reservation", "reserve a suite".

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
  about hotel info and the question is NOT about creating/cancelling/checking
  a specific booking.

Critical Rules:
- If active_flow is not "None" and the message contains negation words
  ("no", "don't", "do not", "not", "never", "cancel", "stop", "forget",
  "never mind", "skip") → ALWAYS return CANCEL_FLOW.
- If suspended_flow is not "None" and the user indicates they want to continue
  or resume → return RESUME_FLOW.
- If the message includes a factual or informational question, route to RAG.
- Only use MAKE_BOOKING if the user is explicitly trying to CREATE a booking.
- Only use CANCEL_BOOKING if the user is explicitly trying to CANCEL an existing booking.

Conversation history (last {history_limit} turns):
{history}

User message:
{message}
"""


async def detect_intent(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow = state["memory"]["flow"]
    active_flow    = flow.get("active")    or "None"
    flow_step      = flow.get("step")      or "None"
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
            options={"temperature": 0.0, "top_p": 0.9},
        )

        raw = res.response.strip()
        logger.info(f"[VOICE INTENT RAW] {raw}")

        data = safe_parse_json(raw)

        if not data:
            logger.warning(f"[VOICE INTENT FALLBACK – no data] raw: {raw}")
            return "RAG"

        allowed = {
            "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
            "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND",
            "CONTROL", "SMALL_TALK", "CANCEL_FLOW", "RESUME_FLOW",
        }

        raw_intent = str(data.get("intent", "RAG")).strip().upper()
        confidence = float(data.get("confidence", 1.0))

        if confidence < 0.5:
            logger.warning(
                f"[VOICE INTENT LOW CONFIDENCE] {raw_intent} @ {confidence:.2f}, "
                "falling back to RAG"
            )
            return "RAG"

        intent = raw_intent if raw_intent in allowed else "RAG"
        logger.info(f"[VOICE INTENT RESOLVED] {intent} (confidence={confidence:.2f})")

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
# Changes from original:
#   • RESUME_FLOW restores the suspended flow instead of falling through.
#   • When a new explicit intent overrides an active flow, the active flow is
#     cleared here so handlers always start from a clean slate.
#   • Suspension slots are NOT touched here; handle_rag owns suspension,
#     handle_resume_flow owns restoration.
# ──────────────────────────────────────────────
def router(intent: str, memory: Dict[str, Any]) -> str:
    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()

    # Cancellation always wins
    if intent == "CANCEL_FLOW":
        return "CANCEL_FLOW"

    flow   = memory.get("flow", {})
    active = flow.get("active")

    # FIX (Bug #1 / resume): explicit resume intent restores suspended flow
    if intent == "RESUME_FLOW":
        suspended = flow.get("suspended_flow")
        if suspended:
            return "RESUME_FLOW"
        # Nothing suspended — treat as a continuation no-op
        return "CONTROL"

    # Resume an in-progress flow ONLY for neutral/continuation intents
    continuation_intents = {"CONTROL", "SMALL_TALK", "EXPAND"}
    if active and intent in continuation_intents:
        return active

    # FIX (Bug #1): a genuine new intent overrides the active flow — clear it
    # so the handler doesn't accidentally resume at a stale step.
    if active and intent not in continuation_intents:
        logger.info(
            f"[ROUTER] New intent '{intent}' overrides active flow '{active}'. "
            "Clearing active flow."
        )
        flow["active"] = None
        flow["step"]   = None

    allowed = {
        "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
        "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK", "CANCEL_FLOW", "RESUME_FLOW",
    }
    return intent if intent in allowed else "RAG"


# ──────────────────────────────────────────────
# RESUME-FLOW HANDLER
# Restores a suspended flow and returns a handoff message.
# ──────────────────────────────────────────────
async def handle_resume_flow(state: VoiceState) -> tuple[str, str]:
    """
    Restores the suspended flow and returns (response_text, restored_route).
    The caller dispatches to the restored route after this.
    """
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    suspended_flow = flow.get("suspended_flow")
    suspended_step = flow.get("suspended_step")

    # Restore
    flow["active"]         = suspended_flow
    flow["step"]           = suspended_step
    flow["suspended_flow"] = None
    flow["suspended_step"] = None

    resume_messages = {
        "MAKE_BOOKING": (
            "Sure! Let's continue with your booking. "
            "Here's where we left off:"
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

    logger.info(
        f"[RESUME_FLOW] Restored flow='{suspended_flow}' step='{suspended_step}'"
    )
    return intro, suspended_flow


# ──────────────────────────────────────────────
# CANCEL-FLOW HANDLER
# FIX: Only wipes booking/search data when no suspended flow exists,
# so a "cancel the RAG question" doesn't destroy a suspended booking flow.
# ──────────────────────────────────────────────
async def handle_cancel_flow(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]

    active_flow    = flow.get("active")
    suspended_flow = flow.get("suspended_flow")

    flow["active"]         = None
    flow["step"]           = None
    flow["suspended_flow"] = None
    flow["suspended_step"] = None
    flow["expandable"]        = False
    flow["last_expand_offer"] = False

    # Only clear transient data if there is no suspended flow to protect.
    # If a suspended flow existed, its data (booking/search) was already
    # preserved when RAG interrupted it — wipe now since the user cancelled.
    if active_flow in {"MAKE_BOOKING", "CANCEL_BOOKING"} or \
       suspended_flow in {"MAKE_BOOKING", "CANCEL_BOOKING"}:
        state["memory"]["booking"] = {}
    if active_flow == "SEARCH_ROOMS" or suspended_flow == "SEARCH_ROOMS":
        state["memory"]["search"] = {}

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
            "Let me know if there's anything else I can help with."
        ),
    }

    # Report whichever flow was actually running (active takes priority)
    reported_flow = active_flow or suspended_flow
    return cancellation_messages.get(
        reported_flow,
        "Alright, I've cancelled that. How else can I help you?"
    )


# ──────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────
async def handle_greeting(state: VoiceState) -> str:
    return choice(GREETING_RESPONSES_HOTEL)


async def handle_small_talk(state: VoiceState) -> str:
    return choice(THANK_YOU_RESPONSES)


async def handle_control(state: VoiceState) -> str:
    state = ensure_memory(state)
    text  = state["message"].lower().strip()
    flow  = state["memory"]["flow"]

    if text in ["no", "nope"]:
        flow["expandable"]        = False
        flow["last_expand_offer"] = False
        return "Alright! Let me know if there's anything else I can help with."

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            return await handle_expand(state)
        return "Sure! How can I help you?"

    return "Got it! Let me know what you need."


async def handle_check_prices(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))

    room_type = None
    for rt in ["standard", "deluxe", "suite", "single", "double", "twin"]:
        if rt in state["message"].lower():
            room_type = rt
            break

    db = SessionLocal()
    try:
        result = get_room_prices(db=db, room_type=room_type)

        if "error" in result:
            return f"Sorry, I couldn't retrieve prices. {result['error']}"

        prices = result.get("prices", [])
        if not prices:
            return "I couldn't retrieve room prices right now. Please contact our front desk."

        lines = [
            f"{p['type']} room: {p['currency']} {p['price']} per night — "
            f"{p['amenities']} — up to {p['capacity']} guests."
            for p in prices
        ]
        return "Here are our current room rates:\n\n" + "\n".join(lines)

    except Exception as e:
        logger.error(f"[CHECK_PRICES ERROR] {e}")
        return "Sorry, I couldn't get room prices. Please try again."
    finally:
        db.close()


# ── Search Rooms ───────────────────────────────────────────────────────────────
async def _execute_search(state: VoiceState) -> str:
    flow   = state["memory"]["flow"]
    search = state["memory"]["search"]

    db = SessionLocal()
    try:
        result = search_available_rooms(
            db=db,
            check_in=search["check_in"],
            check_out=search["check_out"],
            guests=int(search.get("guests", 1)),
        )
    except Exception as e:
        logger.error(f"[SEARCH_ROOMS EXECUTE ERROR] {e}")
        flow["active"] = None
        flow["step"]   = None
        return "Sorry, I couldn't search for rooms. Please try again."
    finally:
        db.close()

    flow["active"] = None
    flow["step"]   = None

    if "error" in result:
        return f"Sorry, {result['error']}"

    rooms = result.get("rooms", [])
    if not rooms:
        return (
            f"No rooms are available from {search['check_in']} to {search['check_out']} "
            f"for {search.get('guests', 1)} guest(s). Please try different dates."
        )

    lines = [
        f"Room {r['room_number']}, {r['type']} — "
        f"{r['currency']} {r['price']}/night — capacity: {r['capacity']} guests."
        for r in rooms
    ]
    return (
        f"Available rooms from {search['check_in']} to {search['check_out']} "
        f"for {search.get('guests', 1)} guest(s):\n\n"
        + "\n".join(lines)
        + "\n\nWould you like to book one of these rooms?"
    )


async def handle_search_rooms(state: VoiceState) -> str:
    state  = ensure_memory(state)
    flow   = state["memory"]["flow"]
    search = state["memory"]["search"]

    # FIX (Bug #4): clear stale search state whenever we enter with no active flow,
    # regardless of what step was previously recorded.
    step = flow.get("step") or "ask_checkin"
    if not flow.get("active"):
        state["memory"]["search"] = {}
        search = state["memory"]["search"]
        step = "ask_checkin"

    flow["active"] = "SEARCH_ROOMS"

    if step == "ask_checkin":
        dates = extract_all_dates(state["message"])
        if dates:
            search["check_in"] = dates[0]
        if len(dates) >= 2:
            search["check_out"] = dates[1]

        if not search.get("check_in"):
            flow["step"] = "ask_checkin"
            return choice(SEARCH_ASK_CHECKIN)

        if not search.get("check_out"):
            flow["step"] = "ask_checkout"
            return choice(SEARCH_ASK_CHECKOUT)

        flow["step"] = "ask_guests"
        return choice(SEARCH_ASK_GUESTS)

    if step == "ask_checkout":
        date = extract_date(state["message"])
        if not date:
            return "Please provide your check-out date in the format YYYY-MM-DD, for example 2025-12-28."
        search["check_out"] = date
        flow["step"] = "ask_guests"
        return choice(SEARCH_ASK_GUESTS)

    if step == "ask_guests":
        guests = extract_number(state["message"]) or 1
        search["guests"] = int(guests)
        logger.info(f"[SEARCH_ROOMS ask_guests] extracted={guests} search={search}")
        return await _execute_search(state)

    # Fallback — restart
    flow["step"] = "ask_checkin"
    state["memory"]["search"] = {}
    return choice(SEARCH_ASK_CHECKIN)


# ── Make Booking ───────────────────────────────────────────────────────────────
async def handle_make_booking(state: VoiceState) -> str:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]
    search  = state["memory"].get("search", {})

    # Pre-fill from a preceding search flow if dates are available
    if search.get("check_in") and not booking.get("check_in"):
        booking["check_in"] = search["check_in"]
    if search.get("check_out") and not booking.get("check_out"):
        booking["check_out"] = search["check_out"]

    # FIX (Bug #2): do NOT run field extraction when we are at the confirm step —
    # it can overwrite already-validated fields with garbage from "yes" / "confirm".
    current_step = flow.get("step")
    if current_step != "confirm":
        extract_all_booking_fields(state["message"], booking)

    flow["active"] = "MAKE_BOOKING"

    next_step = next_missing_booking_step(booking)

    if next_step == "ask_checkin":
        flow["step"] = "ask_checkin"
        return choice(SEARCH_ASK_CHECKIN)

    if next_step == "ask_checkout":
        flow["step"] = "ask_checkout"
        return choice(SEARCH_ASK_CHECKOUT)

    if next_step == "ask_room":
        # Already on ask_room — just re-prompt without re-searching
        if flow.get("step") == "ask_room":
            return "Which room number would you like to book?"

        # First time here — show available rooms to help the user pick
        if booking.get("check_in") and booking.get("check_out"):
            db = SessionLocal()
            try:
                result = search_available_rooms(
                    db=db,
                    check_in=booking["check_in"],
                    check_out=booking["check_out"],
                    guests=int(booking.get("guests_count", 1)),
                )
                rooms = result.get("rooms", [])
                if rooms:
                    lines = [
                        f"Room {r['room_number']} — {r['type']} — "
                        f"{r['currency']} {r['price']}/night — capacity: {r['capacity']}."
                        for r in rooms
                    ]
                    flow["step"] = "ask_room"
                    return (
                        "Here are the available rooms:\n\n"
                        + "\n".join(lines)
                        + f"\n\n{choice(BOOKING_ASK_ROOM)}"
                    )
            except Exception as e:
                logger.error(f"[MAKE_BOOKING SEARCH ERROR] {e}")
            finally:
                db.close()

        flow["step"] = "ask_room"
        return choice(BOOKING_ASK_ROOM)

    if next_step == "ask_name":
        if flow.get("step") == "ask_name":
            raw = state["message"].strip()
            if re.match(r"^[A-Za-z]+(?:\s[A-Za-z]+)+$", raw):
                booking["guest_name"] = raw.title()
                return await handle_make_booking(state)
        flow["step"] = "ask_name"
        return choice(BOOKING_ASK_NAME)

    if next_step == "ask_phone":
        flow["step"] = "ask_phone"
        return choice(BOOKING_ASK_PHONE)

    # All fields collected — show summary for confirmation
    if current_step != "confirm":
        flow["step"] = "confirm"
        return (
            f"Here's your booking summary:\n\n"
            f"Name: {booking.get('guest_name')}\n"
            f"Room: {booking.get('room_number')}\n"
            f"Check-in: {booking.get('check_in')}\n"
            f"Check-out: {booking.get('check_out')}\n"
            f"Guests: {booking.get('guests_count', 1)}\n"
            f"Phone: {booking.get('guest_phone')}\n\n"
            f"Shall I confirm this booking? (yes / no)"
        )

    # Confirmation step
    if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure", "proceed"]):
        if booking.get("guest_phone"):
            booking["guest_phone"] = normalize_phone(booking["guest_phone"])

        db = SessionLocal()
        try:
            result = make_booking(
                db=db,
                guest_name=booking.get("guest_name"),
                guest_phone=booking.get("guest_phone"),
                room_number=booking.get("room_number"),
                check_in=booking.get("check_in"),
                check_out=booking.get("check_out"),
                guests_count=int(booking.get("guests_count", 1)),
            )
        except Exception as e:
            logger.error(f"[MAKE_BOOKING ERROR] {e}")
            result = {"error": str(e)}
        finally:
            db.close()

        if "error" in result:
            booking.pop("room_number", None)
            flow["step"] = "ask_room"
            return (
                f"Sorry, {result['error']}\n\n"
                f"Please provide a different room number to continue."
            )

        booking_id = result.get("booking_id", "UNKNOWN")
        state["memory"]["booking"] = {}
        flow["active"] = None
        flow["step"]   = None
        return (
            choice(BOOKING_SUCCESS)
            + f" Your Booking ID is {booking_id}. Please keep this for your records."
        )

    else:
        state["memory"]["booking"] = {}
        flow["active"] = None
        flow["step"]   = None
        return "Booking cancelled. Let me know if there's anything else I can help with."


# ── Cancel Booking ─────────────────────────────────────────────────────────────
async def handle_cancel_booking(state: VoiceState) -> str:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]

    # FIX (Bug #3): always attempt extraction from the current message first,
    # before deciding which step to run.
    if not booking.get("guest_phone"):
        phone = extract_phone_number(state["message"])
        if phone:
            booking["guest_phone"] = normalize_phone(phone)

    # FIX (Bug #3): also attempt booking ID extraction on the very first message
    # so we don't discard data the user already provided.
    if not booking.get("booking_id"):
        raw = state["message"].strip().upper()
        bid_match = re.search(r"[A-Z0-9\-]{6,}", raw)
        if bid_match:
            booking["booking_id"] = bid_match.group(0)

    step = flow.get("step") or "ask_booking_id"
    flow["active"] = "CANCEL_BOOKING"

    if step == "ask_booking_id":
        # FIX (Bug #3): if we already extracted a booking ID from this message,
        # skip the prompt and advance the step.
        if booking.get("booking_id"):
            flow["step"] = "ask_phone" if not booking.get("guest_phone") else "confirm"
            if flow["step"] == "confirm":
                return choice(CANCEL_CONFIRM).format(booking_id=booking.get("booking_id"))
            return choice(CANCEL_ASK_PHONE)

        flow["step"] = "collect_booking_id"
        return choice(CANCEL_ASK_BOOKING_ID)

    if step == "collect_booking_id":
        if not booking.get("booking_id"):
            raw = state["message"].strip().upper()
            bid_match = re.search(r"[A-Z0-9\-]{6,}", raw)
            booking["booking_id"] = bid_match.group(0) if bid_match else raw

        if booking.get("guest_phone"):
            flow["step"] = "confirm"
            return choice(CANCEL_CONFIRM).format(booking_id=booking.get("booking_id"))

        flow["step"] = "ask_phone"
        return choice(CANCEL_ASK_PHONE)

    if step == "ask_phone":
        phone = extract_phone_number(state["message"])
        if not phone:
            return "Please provide the valid phone number used when making this booking."
        booking["guest_phone"] = normalize_phone(phone)
        flow["step"] = "confirm"
        return choice(CANCEL_CONFIRM).format(booking_id=booking.get("booking_id"))

    if step == "confirm":
        if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure"]):
            db = SessionLocal()
            try:
                result = cancel_booking(
                    db=db,
                    booking_id=booking.get("booking_id"),
                    guest_phone=booking.get("guest_phone"),
                )
            except Exception as e:
                logger.error(f"[CANCEL_BOOKING ERROR] {e}")
                result = {"error": str(e)}
            finally:
                db.close()

            state["memory"]["booking"] = {}
            flow["active"] = None
            flow["step"]   = None

            if "error" in result:
                return f"Sorry, I couldn't cancel the booking. {result['error']}"

            return choice(CANCEL_SUCCESS).format(booking_id=result.get("booking_id"))

        else:
            state["memory"]["booking"] = {}
            flow["active"] = None
            flow["step"]   = None
            return "Cancellation aborted. Your booking remains active."

    # Fallback
    flow["step"] = "collect_booking_id"
    return choice(CANCEL_ASK_BOOKING_ID)


# ── Check Booking ──────────────────────────────────────────────────────────────
async def handle_check_booking(state: VoiceState) -> str:
    state = ensure_memory(state)
    flow  = state["memory"]["flow"]
    step  = flow.get("step") or "ask_booking_id"

    flow["active"] = "CHECK_BOOKING"

    if step == "ask_booking_id":
        flow["step"] = "collect_booking_id"
        return choice(CHECK_ASK_BOOKING_ID)

    if step == "collect_booking_id":
        booking_id = state["message"].strip().upper()

        db = SessionLocal()
        try:
            result = check_booking_status(db=db, booking_id=booking_id)
        except Exception as e:
            logger.error(f"[CHECK_BOOKING ERROR] {e}")
            result = {"error": str(e)}
        finally:
            db.close()

        flow["active"] = None
        flow["step"]   = None

        if "error" in result:
            return "I couldn't find that booking. Please double-check your booking ID."

        return (
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

    # Fallback
    flow["step"] = "collect_booking_id"
    return choice(CHECK_ASK_BOOKING_ID)


# ── RAG ────────────────────────────────────────────────────────────────────────
async def handle_rag(state: VoiceState, vectorstore, system_prompt: str) -> str:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    # FIX (Bug #1 / resume): suspend the active flow instead of destroying it,
    # so the user can resume after the RAG question is answered.
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
        return "I don't have that information right now. Please contact our front desk."

    try:
        docs, scores = get_rag_context(state["message"], vectorstore, k=RAG_K)
    except Exception as e:
        logger.error(f"[RAG VECTOR ERROR] {e}")
        return "I don't have that information right now. Please contact our front desk."

    if not docs:
        return "I don't have that information right now. Please contact our front desk for assistance."

    context = "\n\n".join(docs)
    if not context.strip():
        return "I don't have that information right now. Please contact our front desk for assistance."

    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in state["memory"]["history"][-8:]
    )

    # Include the suspended flow in the prompt so the LLM can offer to resume
    suspended_note = ""
    if flow.get("suspended_flow"):
        suspended_note = (
            f"\n\nNote: the user has a suspended '{flow['suspended_flow']}' flow. "
            "After answering, naturally offer to continue it if appropriate."
        )

    prompt = f"""You are a human hotel customer support representative.

Answer naturally and conversationally like a real front-desk agent.

IMPORTANT:
- Use ONLY the provided context for factual accuracy.
- Do NOT invent information.
- Do NOT sound like a website, brochure, or policy document.
- Speak in short, natural sentences.
- Answer the user's exact question first.
- Avoid marketing language and long paragraphs.
- Sound helpful and human.
- If the context doesn't cover the question, say: "Please contact our front desk for assistance."{suspended_note}

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{state["message"]}

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
        logger.error(f"[RAG LLM ERROR] {e}")
        return "I couldn't process your request. Please try again."

    flow["expandable"]        = True
    flow["last_expand_offer"] = True
    return answer


# ── Expand ─────────────────────────────────────────────────────────────────────
async def handle_expand(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    last_answer = next(
        (m["content"] for m in reversed(state["memory"]["history"]) if m["role"] == "assistant"),
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
                f"- Sound conversational and helpful.\n\n"
                f"EXPANDED ANSWER:"
            ),
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
# ENTRY POINT
# Changes from original:
#   • RESUME_FLOW is handled by handle_resume_flow, which restores the
#     suspended flow and then re-dispatches to the correct handler so the
#     user gets both a "resuming" message and the next prompt in one turn.
#   • FIX (Bug #6): all flow handlers now receive a deep-copied state via
#     safe_state() to prevent cross-handler state mutation.
# ──────────────────────────────────────────────
async def process_intent_rag(
    state: VoiceState,
    vectorstore,
    system_prompt: str,
) -> tuple[str, VoiceState]:
    # FIX (Bug #6): deep-copy once at the entry point so every handler works
    # on an isolated copy; the final mutated state is what we return.
    state  = ensure_memory(safe_state(state))
    intent = await detect_intent(state)
    route  = router(intent, state["memory"])

    logger.info(f"[VOICE ROUTE] intent={intent} → route={route}")

    if route == "GREETING":
        response = await handle_greeting(state)
    elif route == "SMALL_TALK":
        response = await handle_small_talk(state)
    elif route == "CONTROL":
        response = await handle_control(state)
    elif route == "CANCEL_FLOW":
        response = await handle_cancel_flow(state)
    elif route == "RESUME_FLOW":
        # Restore the suspended flow, then immediately dispatch to it so the
        # user gets the next step's prompt in the same turn.
        intro, restored_route = await handle_resume_flow(state)
        if restored_route == "MAKE_BOOKING":
            follow_up = await handle_make_booking(state)
        elif restored_route == "SEARCH_ROOMS":
            follow_up = await handle_search_rooms(state)
        elif restored_route == "CANCEL_BOOKING":
            follow_up = await handle_cancel_booking(state)
        elif restored_route == "CHECK_BOOKING":
            follow_up = await handle_check_booking(state)
        else:
            follow_up = "How can I help you?"
        response = f"{intro}\n\n{follow_up}"
    elif route == "EXPAND":
        response = await handle_expand(state)
    elif route == "CHECK_PRICES":
        response = await handle_check_prices(state)
    elif route == "SEARCH_ROOMS":
        response = await handle_search_rooms(state)
    elif route == "MAKE_BOOKING":
        response = await handle_make_booking(state)
    elif route == "CANCEL_BOOKING":
        response = await handle_cancel_booking(state)
    elif route == "CHECK_BOOKING":
        response = await handle_check_booking(state)
    elif route == "RAG":
        response = await handle_rag(state, vectorstore, system_prompt)
    else:
        response = "I'm sorry, I didn't understand that. How can I help you?"

    # Persist turn to history
    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": response})

    return response, state