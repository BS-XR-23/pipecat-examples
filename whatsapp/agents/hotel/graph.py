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
from utils.helpers import safe_parse_json, extract_phone_number, extract_date, extract_number, normalize_phone
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

load_dotenv()  # Load environment variables from .env file
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))

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

    if not isinstance(state.get("intent"), str):
        state["intent"] = "RAG"

    return state


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
  Examples: "find me a room", "what rooms are available", "search for rooms", "available rooms for booking".

- MAKE_BOOKING: User wants to create a reservation.
  Examples: "I want to book a room", "make a reservation", "reserve a deluxe room".

- CANCEL_BOOKING: User wants to cancel an existing booking.
  Examples: "cancel my booking", "I want to cancel my reservation".

- CHECK_BOOKING: User wants to check an existing booking's status.
  Examples: "check my booking", "what's the status of my reservation".

- CHECK_PRICES: User wants to know room rates or pricing.
  Examples: "how much is a room", "what are the prices", "room rates".

- RAG: Informational questions about hotel policies, amenities, services, or FAQs.
  Examples: "what's check-in time", "do you have a pool", "is breakfast included".
  ALWAYS use RAG when user asks "what", "how", "tell me", "do you", "explain" about hotel info.

CRITICAL RULES:
- If message starts with "what", "how", "tell me", "do you", "explain" and is NOT about creating/cancelling/checking a booking → RAG.
- Use MAKE_BOOKING only when user explicitly wants to CREATE a new booking.
- Use CANCEL_BOOKING only when user explicitly wants to CANCEL.
- Use CHECK_BOOKING only when user wants to CHECK status of an existing booking.

Conversation:
{history}

User:
{message}
"""


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
            options={"temperature": 0, "top_p": 0.9}
        )

        raw = res.response.strip()
        logger.info(f"[INTENT RAW] {raw}")

        data = safe_parse_json(raw)
        if data:
            allowed = {
                "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
                "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND", "CONTROL", "SMALL_TALK"
            }
            raw_intent = str(data.get("intent", "RAG")).strip().upper()
            state["intent"] = raw_intent if raw_intent in allowed else "RAG"
            logger.info(f"[INTENT RESOLVED] {state['intent']}")
        else:
            logger.warning(f"[INTENT FALLBACK] raw: {raw}")

    except Exception as e:
        logger.warning(f"[INTENT NODE ERROR] {e}, falling back to RAG")
        state["intent"] = "RAG"

    return state


def router(state: BotState) -> str:
    flow = state.get("memory", {}).get("flow", {})
    active = flow.get("active")

    if active in {"SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING", "CHECK_BOOKING"}:
        return active

    intent = state.get("intent", "RAG")
    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()
    allowed = {
        "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
        "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND", "CONTROL", "SMALL_TALK"
    }
    return intent if intent in allowed else "RAG"


async def greeting_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(GREETING_RESPONSES_HOTEL)
    state["intent"] = "GREETING"
    return state


async def small_talk_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    state["response"] = choice(SMALL_TALK_RESPONSES_HOTEL)
    state["intent"] = "SMALL_TALK"
    return state


async def control_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    text = state["message"].lower().strip()
    flow = state["memory"]["flow"]

    if text in ["no", "nope"]:
        flow["expandable"] = False
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
    state["intent"] = "CONTROL"
    return state


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
        # vendor_id=state["agent"].vendor_id,
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
    state["intent"] = "CHECK_PRICES"
    return state


async def _execute_search(state: BotState) -> BotState:
    flow   = state["memory"]["flow"]
    search = state["memory"]["search"]

    logger.info(f"[_execute_search] check_in={search.get('check_in')} check_out={search.get('check_out')} guests={search.get('guests')} type={type(search.get('guests'))}")

    db = SessionLocal()
    try:
        result = search_available_rooms(
            db,
            search["check_in"],
            search["check_out"],
            int(search.get("guests", 1))
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
            "check_in": search.get("check_in"),
            "check_out": search.get("check_out"),
            "guests": search.get("guests", 1),
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
        # ── Handles: "room 301", "room number 301", "room no 301", "room #301" ─
        room_match = re.search(r"\broom\s*(?:number|no|#)?\s*(\d{1,4})\b", message, re.IGNORECASE)
        if room_match:
            booking["room_number"] = room_match.group(1)

    if not booking.get("guest_name"):
        # ── Try prefixed patterns first ───────────────────────────────────────
        name_match = re.search(
            r"(?:my(?:\s+full)?\s+name\s+is|i am|i'm|name[:\s]+)\s*([A-Za-z]+(?:\s[A-Za-z]+)*?)(?:\s+and|\s+my|\s+phone|\s+number|,|$)",
            message, re.IGNORECASE
        )
        if name_match:
            booking["guest_name"] = name_match.group(1).strip()
        else:
            # ── Fallback: entire message is just a name (e.g. "Kawsar Mahmud") ─
            plain_match = re.fullmatch(r"\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\s*", message.strip())
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


async def search_rooms_node(state: BotState) -> BotState:
    state  = ensure_memory(state)
    flow   = state["memory"]["flow"]
    step   = flow.get("step") or "ask_checkin"

    # Clear stale search data when starting a fresh search
    if step == "ask_checkin" and not flow.get("active"):
        state["memory"]["search"] = {}

    search = state["memory"]["search"]

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
            flow["step"] = "ask_checkin"
            state["response"] = choice(SEARCH_ASK_CHECKIN)
            return state

        if not search.get("check_out"):
            flow["step"] = "ask_checkout"
            state["response"] = choice(SEARCH_ASK_CHECKOUT)
            return state

        flow["step"] = "ask_guests"
        state["response"] = choice(SEARCH_ASK_GUESTS)
        return state

    if step == "ask_checkout":
        date = extract_date(state["message"])
        if not date:
            state["response"] = "Please provide your check-out date (e.g. 2025-06-05)."
            return state
        search["check_out"] = date
        flow["step"] = "ask_guests"
        state["response"] = choice(SEARCH_ASK_GUESTS)
        return state

    if step == "ask_guests":
        guests = extract_number(state["message"]) or 1
        search["guests"] = int(guests)
        logger.info(f"[ask_guests] message='{state['message']}' extracted={guests} search={search}")
        return await _execute_search(state)

    # Fallback
    flow["active"] = "SEARCH_ROOMS"
    flow["step"]   = "ask_checkin"
    state["response"] = choice(SEARCH_ASK_CHECKIN)
    return state


async def make_booking_node(state: BotState) -> BotState:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]
    search  = state["memory"].get("search", {})
    step    = flow.get("step") or "ask_checkin"

    if search.get("check_in") and not booking.get("check_in"):
        booking["check_in"] = search["check_in"]
    if search.get("check_out") and not booking.get("check_out"):
        booking["check_out"] = search["check_out"]

    extract_all_booking_fields(state["message"], booking)

    flow["active"] = "MAKE_BOOKING"
    next_step = next_missing_step(booking)

    if next_step == "ask_checkin":
        flow["step"] = "ask_checkin"
        state["response"] = choice(SEARCH_ASK_CHECKIN)
        return state

    if next_step == "ask_checkout":
        flow["step"] = "ask_checkout"
        state["response"] = choice(SEARCH_ASK_CHECKOUT)
        return state

    if next_step == "ask_room":
        # ── Already showed the list once — just re-prompt for the room number ──
        if flow.get("step") == "ask_room":
            state["response"] = "Which room number would you like to book?"
            return state

        flow["step"] = "ask_room"

        # ── First time here — search and display available rooms ───────────────
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
                f"Here are the available rooms for {booking['check_in']} → {booking['check_out']}:\n\n"
                f"{room_lines}\n\n"
                f"Which room number would you like to book?"
            )
        else:
            state["response"] = choice(BOOKING_ASK_ROOM)

        return state

    if next_step == "ask_name":
        flow["step"] = "ask_name"
        state["response"] = choice(BOOKING_ASK_NAME)
        return state

    if next_step == "ask_phone":
        flow["step"] = "ask_phone"
        state["response"] = choice(BOOKING_ASK_PHONE)
        return state

    if step != "confirm":
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

        # ── Normalize phone before hitting the DB ──────────────────────────────
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
                f"Agent booked room {booking.get('room_number')} for {booking.get('guest_name')} "
                f"({booking.get('check_in')} → {booking.get('check_out')})"
                if tool_status == "success"
                else f"Agent failed to book room {booking.get('room_number')}: {result.get('error')}"
            )
        except Exception as e:
            result = {"error": str(e)}
            tool_status = "error"
            short_desc = f"Agent encountered an error making a booking for {booking.get('guest_name')}"
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
            flow["step"] = "ask_room"
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
        state["response"] = "Booking cancelled. Let me know if there's anything else I can help with."

    state["memory"]["booking"] = {}
    flow["active"] = None
    flow["step"]   = None
    state["intent"] = "MAKE_BOOKING"
    return state


async def cancel_booking_node(state: BotState) -> BotState:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]
    step    = flow.get("step") or "ask_booking_id"

    if not booking.get("guest_phone"):
        phone = extract_phone_number(state["message"])
        if phone:
            booking["guest_phone"] = normalize_phone(phone)    # ← normalize on extract

    if step == "ask_booking_id":
        flow["active"] = "CANCEL_BOOKING"
        flow["step"]   = "collect_booking_id"
        state["response"] = choice(CANCEL_ASK_BOOKING_ID)
        return state

    if step == "collect_booking_id":
        raw = state["message"].strip().upper()
        bid_match = re.search(r"[A-Z0-9\-]{6,}", raw)
        booking["booking_id"] = bid_match.group(0) if bid_match else raw

        if booking.get("guest_phone"):
            flow["step"] = "confirm"
            state["response"] = choice(CANCEL_CONFIRM).format(booking_id=booking.get("booking_id"))
            return state

        flow["step"] = "ask_phone"
        state["response"] = choice(CANCEL_ASK_PHONE)
        return state

    if step == "ask_phone":
        phone = extract_phone_number(state["message"])
        if not phone:
            state["response"] = "Please provide the valid phone number used when making this booking."
            return state
        booking["guest_phone"] = normalize_phone(phone)        # ← normalize on extract
        flow["step"] = "confirm"
        state["response"] = choice(CANCEL_CONFIRM).format(booking_id=booking.get("booking_id"))
        return state

    if step == "confirm":
        if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure"]):
            db = SessionLocal()
            try:
                result = cancel_booking(db, booking.get("booking_id"), booking.get("guest_phone"))
                tool_status = "failed" if "error" in result else "success"
                short_desc = (
                    f"Agent cancelled booking {booking.get('booking_id')} "
                    f"for phone {booking.get('guest_phone')}"
                    if tool_status == "success"
                    else f"Agent failed to cancel booking {booking.get('booking_id')}: {result.get('error')}"
                )
            except Exception as e:
                result = {"error": str(e)}
                tool_status = "error"
                short_desc = f"Agent encountered an error cancelling booking {booking.get('booking_id')}"
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
                state["response"] = choice(CANCEL_SUCCESS).format(booking_id=booking.get("booking_id"))
        else:
            state["response"] = "Cancellation aborted. Your booking remains active."

        state["memory"]["booking"] = {}
        flow["active"] = None
        flow["step"]   = None
        state["intent"] = "CANCEL_BOOKING"
        return state

    flow["active"] = "CANCEL_BOOKING"
    flow["step"]   = "collect_booking_id"
    state["response"] = choice(CANCEL_ASK_BOOKING_ID)
    return state


async def check_booking_node(state: BotState) -> BotState:
    state   = ensure_memory(state)
    flow    = state["memory"]["flow"]
    booking = state["memory"]["booking"]
    step    = flow.get("step") or "ask_booking_id"

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
                f"Agent checked status of booking {booking_id}: {result.get('status', 'unknown').upper()}"
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
            state["response"] = "I couldn't find that booking. Please double-check your booking ID."
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

        flow["active"] = None
        flow["step"]   = None
        state["intent"] = "CHECK_BOOKING"
        return state

    flow["active"] = "CHECK_BOOKING"
    flow["step"]   = "collect_booking_id"
    state["response"] = choice(CHECK_ASK_BOOKING_ID)
    return state


async def rag_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    agent = state["agent"]
    try:
        vectorstore = agent.get_vectorstore()
    except Exception as e:
        logger.error(f"[RAG VECTOR ERROR] {e}")
        state["response"] = "I don't have that information right now. Please contact our front desk."
        state["intent"] = "RAG_FAILED"
        return state

    if vectorstore is None:
        state["response"] = "I don't have that information right now. Please contact our front desk."
        state["intent"] = "RAG_FAILED"
        return state

    system_prompt = agent.get_system_prompt()

    docs, scores = get_rag_context(state["message"], vectorstore)

    if not docs:
        state["response"] = "I don't have that information right now. Please contact our front desk for assistance."
        state["intent"] = "RAG_EMPTY"
        return state

    context = "\n\n".join(docs)

    if not context.strip():
        state["response"] = "I don't have that information right now. Please contact our front desk for assistance."
        state["intent"] = "RAG_EMPTY"
        return state

    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in state["memory"]["history"][-8:]
    )

    prompt = f"""Answer the user's question using ONLY the context below.

CRITICAL RULES:
- Do NOT invent information
- Do NOT skip list items
- If answer is missing → say contact front desk
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
        state["response"] = "I couldn't process your request. Please try again."
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
    state["intent"] = "RAG_SUCCESS"

    return state


async def expand_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    agent = state["agent"]
    system_prompt = agent.get_system_prompt()

    last_answer = next(
        (m["content"] for m in reversed(state["memory"]["history"]) if m["role"] == "assistant"),
        ""
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

EXPANDED ANSWER:"""
    )
    expanded = res.response.strip()
    flow["expandable"]        = False
    flow["last_expand_offer"] = False
    state["memory"]["history"].append({"role": "user",      "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": expanded})
    state["response"] = expanded
    state["intent"]   = "EXPAND"
    return state


def build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(BotState)

    graph.add_node("detect", detect_intent)
    graph.add_node("greeting", greeting_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("control", control_node)
    graph.add_node("search_rooms", search_rooms_node)
    graph.add_node("make_booking", make_booking_node)
    graph.add_node("cancel_booking", cancel_booking_node)
    graph.add_node("check_booking", check_booking_node)
    graph.add_node("check_prices", check_prices_node)
    graph.add_node("rag", rag_node)
    graph.add_node("expand", expand_node)
    graph.add_edge(START, "detect")

    graph.set_entry_point("detect")

    graph.add_conditional_edges(
        "detect",
        router,
        {
            "GREETING": "greeting",
            "SMALL_TALK": "small_talk",
            "CONTROL": "control",
            "SEARCH_ROOMS": "search_rooms",
            "MAKE_BOOKING": "make_booking",
            "CANCEL_BOOKING": "cancel_booking",
            "CHECK_BOOKING": "check_booking",
            "CHECK_PRICES": "check_prices",
            "RAG": "rag",
            "EXPAND": "expand",
        }
    )

    for node in [
        "greeting",
        "small_talk",
        "control",
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