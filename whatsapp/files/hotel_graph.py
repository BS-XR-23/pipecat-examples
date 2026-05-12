from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from random import choice
import re
import copy

from loguru import logger
from ollama import AsyncClient

from db.database import SessionLocal
from rag_service_hotel import get_rag_context
from core.agent_registry import get_agent
from utils.helpers import safe_parse_json, extract_phone_number
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


# ── State ──────────────────────────────────────────────────────────────────────
class BotState(TypedDict):
    session_id: str
    message:    str
    agent_name: str
    intent:     Optional[str]
    response:   Optional[str]
    memory:     Dict[str, Any]


# ── Utilities ──────────────────────────────────────────────────────────────────
def safe_state(state: BotState) -> BotState:
    return copy.deepcopy(state)


def ensure_memory(state: BotState) -> BotState:
    state.setdefault("memory", {})
    memory = state["memory"]
    memory.setdefault("profile", {})
    memory.setdefault("history", [])
    memory.setdefault("booking", {})
    memory.setdefault("search",  {})
    memory.setdefault("flow",    {})

    flow = memory["flow"]
    flow.setdefault("active",            None)
    flow.setdefault("step",              None)
    flow.setdefault("expandable",        False)
    flow.setdefault("last_expand_offer", False)

    if not isinstance(state.get("intent"), str):
        state["intent"] = "RAG"

    return state


def extract_date(text: str) -> Optional[str]:
    """Extract date as YYYY-MM-DD from natural text."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)
    match = re.search(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", text)
    if match:
        d, m, y = match.group(1), match.group(2), match.group(3)
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    return None


def extract_number(text: str) -> Optional[int]:
    """Extract a guest/room count, stripping dates first to avoid false matches."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}", "", text)
    match = re.search(r"\b([1-9][0-9]?)\b", text)
    return int(match.group(1)) if match else None


def extract_all_booking_fields(message: str, booking: dict) -> None:
    """Extract any booking-relevant fields from a message, without overwriting existing ones."""
    if not booking.get("check_in"):
        date = extract_date(message)
        if date:
            booking["check_in"] = date

    if booking.get("check_in") and not booking.get("check_out"):
        for d in re.findall(r"\d{4}-\d{2}-\d{2}", message):
            if d != booking["check_in"]:
                booking["check_out"] = d
                break
        if not booking.get("check_out"):
            for m in re.findall(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", message):
                d_fmt = f"{m[2]}-{m[1].zfill(2)}-{m[0].zfill(2)}"
                if d_fmt != booking["check_in"]:
                    booking["check_out"] = d_fmt
                    break

    if not booking.get("guest_phone"):
        phone = extract_phone_number(message)
        if phone:
            booking["guest_phone"] = phone

    if not booking.get("room_number"):
        room_match = re.search(r"\b(?:room\s*)?(\d{2,4})\b", message, re.IGNORECASE)
        if room_match:
            booking["room_number"] = room_match.group(1)

    if not booking.get("guest_name"):
        name_match = re.search(
            r"(?:my name is|i am|i'm|name[:\s]+)\s*([A-Za-z]+(?:\s[A-Za-z]+)*?)"
            r"(?:\s+and|\s+my|\s+phone|\s+number|,|$)",
            message, re.IGNORECASE
        )
        if name_match:
            booking["guest_name"] = name_match.group(1).strip()

    if not booking.get("guests_count"):
        msg_clean = re.sub(r"\d{4}-\d{2}-\d{2}", "", message)
        msg_clean = re.sub(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}", "", msg_clean)
        guests = extract_number(msg_clean)
        if guests:
            booking["guests_count"] = guests


def next_missing_step(booking: dict) -> str:
    if not booking.get("check_in"):    return "ask_checkin"
    if not booking.get("check_out"):   return "ask_checkout"
    if not booking.get("room_number"): return "ask_room"
    if not booking.get("guest_name"):  return "ask_name"
    if not booking.get("guest_phone"): return "ask_phone"
    return "confirm"


# ── Intent Prompt ──────────────────────────────────────────────────────────────
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
- If message starts with "what", "how", "tell me", "do you", "explain" and is NOT about creating/cancelling/checking a booking → RAG.
- Use MAKE_BOOKING only when user explicitly wants to CREATE a new booking.
- Use CANCEL_BOOKING only when user explicitly wants to CANCEL.
- Use CHECK_BOOKING only when user wants to CHECK status of an existing booking.

Conversation:
{history}

User:
{message}
"""

ALLOWED_INTENTS = {
    "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING",
    "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND", "CONTROL", "SMALL_TALK",
}


# ── Nodes ──────────────────────────────────────────────────────────────────────
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
            options={"temperature": 0, "top_p": 0.9},
        )

        raw  = res.response.strip()
        logger.info(f"[INTENT RAW] {raw}")
        data = safe_parse_json(raw)

        if data:
            raw_intent      = str(data.get("intent", "RAG")).strip().upper()
            state["intent"] = raw_intent if raw_intent in ALLOWED_INTENTS else "RAG"
            logger.info(f"[INTENT RESOLVED] {state['intent']}")
        else:
            logger.warning(f"[INTENT FALLBACK] raw: {raw}")

    except Exception as e:
        logger.warning(f"[INTENT NODE ERROR] {e}, falling back to RAG")
        state["intent"] = "RAG"

    return state


def router(state: BotState) -> str:
    """Route by active flow first, then by detected intent."""
    flow   = state.get("memory", {}).get("flow", {})
    active = flow.get("active")

    if active in {"SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING", "CHECK_BOOKING"}:
        return active

    intent = state.get("intent", "RAG")
    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()
    return intent if intent in ALLOWED_INTENTS else "RAG"


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


async def control_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    text  = state["message"].lower().strip()
    flow  = state["memory"]["flow"]

    if text in {"no", "nope"}:
        flow["expandable"]        = False
        flow["last_expand_offer"] = False
        state["response"] = "Alright! Let me know if there's anything else I can help with."
        return state

    if text in {"yes", "yeah", "sure", "ok", "okay"}:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            state["intent"] = "EXPAND"
            return await expand_node(state)
        state["response"] = "Sure! How can I help you?"
        return state

    state["response"] = "Got it! Let me know what you need."
    state["intent"]   = "CONTROL"
    return state


async def check_prices_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))

    db = SessionLocal()
    try:
        result = get_room_prices(db)
    finally:
        db.close()

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


async def _execute_search(state: BotState) -> BotState:
    """Shared search executor used by search_rooms_node."""
    flow   = state["memory"]["flow"]
    search = state["memory"]["search"]

    db = SessionLocal()
    try:
        result = search_available_rooms(
            db,
            search["check_in"],
            search["check_out"],
            search.get("guests", 1),
        )
    finally:
        db.close()

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
    search = state["memory"]["search"]
    step   = flow.get("step") or "ask_checkin"

    if step == "ask_checkin":
        flow["active"] = "SEARCH_ROOMS"

        date = extract_date(state["message"])
        if date:
            search["check_in"] = date

        guests = extract_number(state["message"])
        if guests:
            search["guests"] = guests

        if search.get("check_in"):
            for d in re.findall(r"\d{4}-\d{2}-\d{2}", state["message"]):
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

        if not search.get("guests"):
            flow["step"] = "ask_guests"
            state["response"] = choice(SEARCH_ASK_GUESTS)
            return state

        return await _execute_search(state)

    if step == "ask_checkout":
        date = extract_date(state["message"])
        if not date:
            state["response"] = "Please provide your check-out date (e.g. 2025-06-05)."
            return state
        search["check_out"] = date

        if not search.get("guests"):
            flow["step"] = "ask_guests"
            state["response"] = choice(SEARCH_ASK_GUESTS)
            return state

        return await _execute_search(state)

    if step == "ask_guests":
        search["guests"] = extract_number(state["message"]) or 1
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

    # Pre-fill dates from a prior search
    if search.get("check_in")  and not booking.get("check_in"):
        booking["check_in"]  = search["check_in"]
    if search.get("check_out") and not booking.get("check_out"):
        booking["check_out"] = search["check_out"]

    # Greedily extract any available fields from every message
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
        flow["step"] = "ask_room"
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

    # Show summary only once — not on re-entry after an error
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

    # Confirmation step
    if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure", "proceed"]):
        db = SessionLocal()
        try:
            result = make_booking(
                db,
                guest_name   = booking.get("guest_name"),
                guest_phone  = booking.get("guest_phone"),
                room_number  = booking.get("room_number"),
                check_in     = booking.get("check_in"),
                check_out    = booking.get("check_out"),
                guests_count = booking.get("guests_count", 1),
            )
        finally:
            db.close()

        if "error" in result:
            booking.pop("room_number", None)
            flow["step"] = "ask_room"
            state["response"] = f"{result['error']}\n\nPlease provide a different room number to continue."
            return state

        booking_id = result.get("booking_id", "UNKNOWN")
        state["response"] = (
            choice(BOOKING_SUCCESS) + f" Your Booking ID is: {booking_id}. Please keep this for your records."
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

    # Opportunistically extract phone at every step
    if not booking.get("guest_phone"):
        phone = extract_phone_number(state["message"])
        if phone:
            booking["guest_phone"] = phone

    if step == "ask_booking_id":
        flow["active"] = "CANCEL_BOOKING"
        flow["step"]   = "collect_booking_id"
        state["response"] = choice(CANCEL_ASK_BOOKING_ID)
        return state

    if step == "collect_booking_id":
        raw       = state["message"].strip().upper()
        bid_match = re.search(r"[A-Z0-9\-]{6,}", raw)
        booking["booking_id"] = bid_match.group(0) if bid_match else raw

        if booking.get("guest_phone"):
            flow["step"] = "confirm"
            state["response"] = choice(CANCEL_CONFIRM).format(booking_id=booking["booking_id"])
            return state

        flow["step"] = "ask_phone"
        state["response"] = choice(CANCEL_ASK_PHONE)
        return state

    if step == "ask_phone":
        phone = extract_phone_number(state["message"])
        if not phone:
            state["response"] = "Please provide the valid phone number used when making this booking."
            return state
        booking["guest_phone"] = phone
        flow["step"] = "confirm"
        state["response"] = choice(CANCEL_CONFIRM).format(booking_id=booking.get("booking_id"))
        return state

    if step == "confirm":
        if any(x in state["message"].lower() for x in ["yes", "confirm", "ok", "sure"]):
            db = SessionLocal()
            try:
                result = cancel_booking(db, booking.get("booking_id"), booking.get("guest_phone"))
            finally:
                db.close()

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

    # Fallback
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
        finally:
            db.close()

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

    # Fallback
    flow["active"] = "CHECK_BOOKING"
    flow["step"]   = "collect_booking_id"
    state["response"] = choice(CHECK_ASK_BOOKING_ID)
    return state


async def rag_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))

    agent         = get_agent(state["agent_name"])
    vectorstore   = agent.get_vectorstore()
    system_prompt = agent.get_system_prompt()

    if vectorstore is None:
        state["response"] = "I don't have that information right now. Please contact our front desk."
        return state

    docs, _ = get_rag_context(state["message"], vectorstore)
    context = "\n".join(docs) if docs else ""

    if not context.strip():
        state["response"] = "I don't have that information right now. Please contact our front desk for assistance."
        return state

    client = AsyncClient()
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in state["memory"]["history"][-8:]
    )

    prompt = f"""Answer the user's question using ONLY the context below.

CRITICAL RULES:
- If the answer contains a list of fields, steps, or requirements — reproduce EVERY item exactly. Do NOT merge, skip, or paraphrase.
- Do NOT add information that is not in the context.
- If the context does not contain the answer, say: "Please contact our front desk for assistance."

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{state['message']}

ANSWER:"""

    res = await client.generate(
        model="gemma4:e4b",
        prompt=prompt,
        system=system_prompt,
        options={"temperature": 0.2, "top_p": 0.9},
    )

    state["memory"]["flow"]["expandable"]        = True
    state["memory"]["flow"]["last_expand_offer"] = True
    state["response"] = res.response.strip()
    state["intent"]   = "RAG"
    # NOTE: history is appended by server.py uniformly for all nodes
    return state


async def expand_node(state: BotState) -> BotState:
    state = ensure_memory(safe_state(state))
    flow  = state["memory"]["flow"]

    agent         = get_agent(state["agent_name"])
    system_prompt = agent.get_system_prompt()

    last_answer = next(
        (m["content"] for m in reversed(state["memory"]["history"]) if m["role"] == "assistant"),
        "",
    )

    client = AsyncClient()
    res    = await client.generate(
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

EXPANDED ANSWER:""",
    )

    flow["expandable"]        = False
    flow["last_expand_offer"] = False
    state["response"] = res.response.strip()
    state["intent"]   = "EXPAND"
    # NOTE: history is appended by server.py uniformly for all nodes
    return state


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(BotState)

    graph.add_node("detect",         detect_intent)
    graph.add_node("greeting",       greeting_node)
    graph.add_node("small_talk",     small_talk_node)
    graph.add_node("control",        control_node)
    graph.add_node("search_rooms",   search_rooms_node)
    graph.add_node("make_booking",   make_booking_node)
    graph.add_node("cancel_booking", cancel_booking_node)
    graph.add_node("check_booking",  check_booking_node)
    graph.add_node("check_prices",   check_prices_node)
    graph.add_node("rag",            rag_node)
    graph.add_node("expand",         expand_node)

    graph.add_edge(START, "detect")

    graph.add_conditional_edges(
        "detect",
        router,
        {
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
        },
    )

    for node in [
        "greeting", "small_talk", "control",
        "search_rooms", "make_booking", "cancel_booking",
        "check_booking", "check_prices", "rag", "expand",
    ]:
        graph.add_edge(node, END)

    return graph.compile()
