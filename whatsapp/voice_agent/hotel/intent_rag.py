from typing import TypedDict, Optional, Dict, Any
from random import choice
import os
import re
import asyncio
import copy
from dotenv import load_dotenv
from ollama import AsyncClient
from loguru import logger
from db.database import SessionLocal
from services.rag_services.rag_service_hotel import get_rag_context
from utils.helpers import safe_parse_json, extract_phone_number, normalize_phone
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
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))
OLLAMA_INTENT_MODEL = os.getenv("OLLAMA_INTENT_MODEL", "gemma4:e2b")
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e4b")
RAG_HISTORY_LIMIT = int(os.getenv("RAG_HISTORY_LIMIT", "6"))
RAG_K = int(os.getenv("RAG_K", "3"))

ollama_client = AsyncClient()


class VoiceState(TypedDict):
    session_id: str
    message: str
    memory: Dict[str, Any]


# ── Regex fast-path intent patterns ───────────────────────────────────────────
_FAST_INTENTS = [
    (re.compile(r"^\s*(hi|hello|hey|good\s*(morning|evening|afternoon))\b", re.I), "GREETING"),
    (re.compile(r"^\s*(thanks?|thank you|okay|ok|bye|got it|sure|alright)\s*$", re.I), "SMALL_TALK"),
    (re.compile(r"\b(search|find|available|show).*room", re.I), "SEARCH_ROOMS"),
    (re.compile(r"\b(how much|price|prices|rate|rates|cost)\b", re.I), "CHECK_PRICES"),
    (re.compile(r"\b(book|reserve|make.*booking|make.*reservation)\b", re.I), "MAKE_BOOKING"),
    (re.compile(r"\b(cancel|cancellation).*booking\b", re.I), "CANCEL_BOOKING"),
    (re.compile(r"\b(check|status|my booking|my reservation)\b", re.I), "CHECK_BOOKING"),
]

# ── Date / number extraction ───────────────────────────────────────────────────
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
    # Strip dates first so date digits don't confuse guest count
    clean = _DATE_ISO.sub("", text)
    clean = _DATE_DMY.sub("", clean)
    m = _NUMBER.search(clean)
    return int(m.group(1)) if m else None

def extract_room_number(text: str) -> Optional[str]:
    m = re.search(r"\b(?:room\s*)?([A-Za-z]?\d{1,4}[A-Za-z]?)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None

def extract_name(text: str) -> Optional[str]:
    m = re.search(
        r"(?:my name is|i am|i'm|name[:\s]+)\s*([A-Za-z]+(?:\s[A-Za-z]+)*?)(?:\s+and|\s+my|\s+phone|\s+number|,|$)",
        text, re.IGNORECASE
    )
    return m.group(1).strip() if m else None


# ── Booking field helpers ──────────────────────────────────────────────────────
def extract_all_booking_fields(message: str, booking: dict):
    """Try to extract any booking fields present in the message."""
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
            booking["guest_phone"] = normalize_phone(phone)    # ← normalize on extract

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


# ── Memory helpers ─────────────────────────────────────────────────────────────
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

    return state


# ── Intent detection ───────────────────────────────────────────────────────────
INTENT_PROMPT = """
You are a routing system for a hotel booking chatbot.

Return ONLY valid minified JSON. No explanation. No text.

Example:
{{"intent":"MAKE_BOOKING","confidence":0.95}}

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

Routing Rules:

- GREETING: User says hello, hi, good morning, or starts conversation.
- SMALL_TALK: Casual replies like thanks, bye, okay, got it.
- CONTROL: Short yes/no/ok/sure — ONLY when no active booking flow.
- EXPAND: User wants more detail about the previous answer.
- SEARCH_ROOMS: User wants to find or see available rooms.
- MAKE_BOOKING: User wants to create a reservation.
- CANCEL_BOOKING: User wants to cancel an existing booking.
- CHECK_BOOKING: User wants to check an existing booking's status.
- CHECK_PRICES: User wants to know room rates or pricing.
- RAG: Informational questions about hotel policies, amenities, services, or FAQs.
  ALWAYS use RAG when user asks "what", "how", "tell me", "do you", "explain" about hotel info.

CRITICAL RULES:
- If message starts with "what", "how", "tell me", "do you", "explain" and is NOT about
  creating/cancelling/checking a booking → RAG.
- Use MAKE_BOOKING only when user explicitly wants to CREATE a new booking.
- Use CANCEL_BOOKING only when user explicitly wants to CANCEL.
- Use CHECK_BOOKING only when user wants to CHECK status of an existing booking.

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

    # Slow-path: Ollama
    try:
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in state["memory"]["history"][-RAG_HISTORY_LIMIT:]
        )
        prompt = INTENT_PROMPT.format(history=history_text, message=message)

        res = await ollama_client.generate(
            model=OLLAMA_INTENT_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.9}
        )

        raw = res.response.strip()
        logger.info(f"[VOICE INTENT RAW] {raw}")

        data = safe_parse_json(raw)
        if data:
            allowed = {
                "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
                "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND", "CONTROL", "SMALL_TALK"
            }
            raw_intent = str(data.get("intent", "RAG")).strip().upper()
            intent = raw_intent if raw_intent in allowed else "RAG"
            logger.info(f"[VOICE INTENT RESOLVED] {intent}")
            return intent
        else:
            logger.warning(f"[VOICE INTENT FALLBACK] raw: {raw}")
            return "RAG"

    except Exception as e:
        logger.warning(f"[VOICE INTENT ERROR] {e}, falling back to RAG")
        return "RAG"


def router(intent: str, memory: Dict[str, Any]) -> str:
    allowed = {
        "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
        "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND", "CONTROL", "SMALL_TALK"
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
    return choice(GREETING_RESPONSES_HOTEL)


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
            guests=int(search.get("guests", 1))
        )
    except Exception as e:
        logger.error(f"[SEARCH_ROOMS EXECUTE ERROR] {e}")
        flow["active"] = None
        flow["step"] = None
        return "Sorry, I couldn't search for rooms. Please try again."
    finally:
        db.close()

    flow["active"] = None
    flow["step"] = None

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
    step   = flow.get("step") or "ask_checkin"

    # Fresh start — clear stale search data
    if step == "ask_checkin" and not flow.get("active"):
        state["memory"]["search"] = {}
        search = state["memory"]["search"]

    flow["active"] = "SEARCH_ROOMS"

    if step == "ask_checkin":
        # Try to extract dates from initial message
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

    # Pre-fill from search if available
    if search.get("check_in") and not booking.get("check_in"):
        booking["check_in"] = search["check_in"]
    if search.get("check_out") and not booking.get("check_out"):
        booking["check_out"] = search["check_out"]

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
        # If already on ask_room step, just re-prompt — don't re-search
        if flow.get("step") == "ask_room":
            return "Which room number would you like to book?"

        # Show available rooms to help the user pick
        if booking.get("check_in") and booking.get("check_out"):
            db = SessionLocal()
            try:
                result = search_available_rooms(
                    db=db,
                    check_in=booking["check_in"],
                    check_out=booking["check_out"],
                    guests=int(booking.get("guests_count", 1))
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
        # ✅ If already on ask_name step, treat raw message as the name directly
        if flow.get("step") == "ask_name":
            raw = state["message"].strip()
            if re.match(r"^[A-Za-z]+(?:\s[A-Za-z]+)+$", raw):  # at least two words
                booking["guest_name"] = raw.title()
                return await handle_make_booking(state)  # re-evaluate with name filled
        flow["step"] = "ask_name"
        return choice(BOOKING_ASK_NAME)

    if next_step == "ask_phone":
        flow["step"] = "ask_phone"
        return choice(BOOKING_ASK_PHONE)

    # All fields collected — show summary for confirmation
    if flow.get("step") != "confirm":
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
        flow["step"] = None
        return (
            choice(BOOKING_SUCCESS)
            + f" Your Booking ID is {booking_id}. Please keep this for your records."
        )

    else:
        state["memory"]["booking"] = {}
        flow["active"] = None
        flow["step"] = None
        return "Booking cancelled. Let me know if there's anything else I can help with."


# ── Cancel Booking ─────────────────────────────────────────────────────────────
async def handle_cancel_booking(state: VoiceState) -> str:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]
    step    = flow.get("step") or "ask_booking_id"

    # Try to extract and normalize phone from any step
    if not booking.get("guest_phone"):
        phone = extract_phone_number(state["message"])
        if phone:
            booking["guest_phone"] = normalize_phone(phone)    # ← normalize on extract

    flow["active"] = "CANCEL_BOOKING"

    if step == "ask_booking_id":
        flow["step"] = "collect_booking_id"
        return choice(CANCEL_ASK_BOOKING_ID)

    if step == "collect_booking_id":
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
        booking["guest_phone"] = normalize_phone(phone)        # ← normalize on extract
        flow["step"] = "confirm"
        return choice(CANCEL_CONFIRM).format(booking_id=booking.get("booking_id"))

    if step == "confirm":
        if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure"]):
            db = SessionLocal()
            try:
                result = cancel_booking(
                    db=db,
                    booking_id=booking.get("booking_id"),
                    guest_phone=booking.get("guest_phone")
                )
            except Exception as e:
                logger.error(f"[CANCEL_BOOKING ERROR] {e}")
                result = {"error": str(e)}
            finally:
                db.close()

            state["memory"]["booking"] = {}
            flow["active"] = None
            flow["step"] = None

            if "error" in result:
                return f"Sorry, I couldn't cancel the booking. {result['error']}"

            return choice(CANCEL_SUCCESS).format(booking_id=result.get("booking_id"))

        else:
            state["memory"]["booking"] = {}
            flow["active"] = None
            flow["step"] = None
            return "Cancellation aborted. Your booking remains active."

    # Fallback
    flow["step"] = "collect_booking_id"
    return choice(CANCEL_ASK_BOOKING_ID)


# ── Check Booking ──────────────────────────────────────────────────────────────
async def handle_check_booking(state: VoiceState) -> str:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    step    = flow.get("step") or "ask_booking_id"

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
        flow["step"] = None

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

    prompt = f"""Answer the user's question using ONLY the context below.

CRITICAL RULES:
- Do NOT invent information.
- Do NOT skip list items.
- If the answer is missing from the context → say to contact the front desk.
- Use ONLY the provided context.

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
        logger.error(f"[RAG LLM ERROR] {e}")
        return "I couldn't process your request. Please try again."

    flow["expandable"] = True
    flow["last_expand_offer"] = True
    return answer


# ── Expand ─────────────────────────────────────────────────────────────────────
async def handle_expand(state: VoiceState) -> str:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    last_answer = next(
        (m["content"] for m in reversed(state["memory"]["history"]) if m["role"] == "assistant"),
        ""
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
                f"- Do NOT use external knowledge.\n\n"
                f"EXPANDED ANSWER:"
            ),
            options={"temperature": 0.2, "top_p": 0.9}
        )
        expanded = res.response.strip()
    except Exception as e:
        logger.error(f"[EXPAND ERROR] {e}")
        return "I couldn't expand on that. Please try rephrasing your question."

    flow["expandable"] = False
    flow["last_expand_offer"] = False
    return expanded


# ── Entry point ────────────────────────────────────────────────────────────────
async def process_intent_rag(state: VoiceState, vectorstore, system_prompt: str) -> str:
    state  = ensure_memory(state)
    intent = await detect_intent(state)
    route  = router(intent, state["memory"])

    logger.info(f"[VOICE ROUTE] intent={intent} → route={route}")

    if route == "GREETING":
        response = await handle_greeting(state)
    elif route == "SMALL_TALK":
        response = await handle_small_talk(state)
    elif route == "CONTROL":
        response = await handle_control(state)
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

    # Update history
    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": response})

    return response, state