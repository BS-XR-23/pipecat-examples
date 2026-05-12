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
from voice_agent.hotel.agent_tools import search_available_rooms, get_room_prices, make_booking, check_booking_status, cancel_booking
from services.rag_services.rag_service_hotel import get_rag_context
from utils.agent_logger import log_tool_call
# Assume response constants are defined similarly for hotel
from response import (
    GREETING_RESPONSES,
    THANK_YOU_RESPONSES
)

load_dotenv()
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))
OLLAMA_INTENT_MODEL = os.getenv("OLLAMA_INTENT_MODEL", "gemma4:e2b")
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e2b")
RAG_HISTORY_LIMIT = int(os.getenv("RAG_HISTORY_LIMIT", "4"))
RAG_K = int(os.getenv("RAG_K", "2"))

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
    memory.setdefault("booking", {})
    memory.setdefault("search", {})
    memory.setdefault("flow", {})

    flow = memory["flow"]
    flow.setdefault("active", None)
    flow.setdefault("step", None)
    flow.setdefault("expandable", False)
    flow.setdefault("last_expand_offer", False)

    return state

INTENT_PROMPT = """
You are a routing system for a hotel booking chatbot.

Return ONLY valid minified JSON. No explanation. No text.

Example:
{{"intent":"SEARCH_ROOMS","confidence":0.95}}

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
                "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING", "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND", "CONTROL", "SMALL_TALK"
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

    if active:
        return active

    if not isinstance(intent, str):
        return "RAG"

    intent = intent.strip().upper()

    allowed = {
        "GREETING", "SEARCH_ROOMS", "MAKE_BOOKING", "CANCEL_BOOKING", "CHECK_BOOKING", "CHECK_PRICES", "RAG", "EXPAND", "CONTROL", "SMALL_TALK"
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

async def handle_search_rooms(state: VoiceState):
    # Implement search logic, perhaps ask for dates, etc.
    # For simplicity, call the tool
    try:
        result = search_available_rooms()
        return result["message"]
    except Exception as e:
        logger.error(f"[SEARCH_ROOMS ERROR] {e}")
        return "Sorry, I couldn't search rooms. Please try again."

async def handle_check_prices(state: VoiceState):
    try:
        result = get_room_prices()
        return result["message"]
    except Exception as e:
        logger.error(f"[CHECK_PRICES ERROR] {e}")
        return "Sorry, I couldn't get prices. Please try again."

async def handle_make_booking(state: VoiceState):
    # Need to collect info, similar to ticket
    # For now, placeholder
    return "Booking functionality is under development."

async def handle_check_booking(state: VoiceState):
    # Placeholder
    return "Check booking functionality is under development."

async def handle_cancel_booking(state: VoiceState):
    # Placeholder
    return "Cancel booking functionality is under development."

async def handle_rag(state: VoiceState, vectorstore, system_prompt):
    state = ensure_memory(state)
    state = safe_state(state)

    try:
        docs, scores = get_rag_context(state["message"], vectorstore, k=RAG_K)

        if not docs:
            return "Please contact hotel support."

        context = "\n\n".join(docs)

        if not context.strip():
            return "Please contact hotel support."

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
        return "Please contact hotel support."

async def process_intent_rag(state: VoiceState, vectorstore, system_prompt):
    intent = await detect_intent(state)
    route = router(intent, state["memory"])

    if route == "GREETING":
        response = await handle_greeting(state)
    elif route == "SMALL_TALK":
        response = await handle_small_talk(state)
    elif route == "CONTROL":
        response = await handle_control(state)
    elif route == "EXPAND":
        response = await handle_control(state)  # reuses control logic; replace with handle_expand when implemented
    elif route == "SEARCH_ROOMS":
        response = await handle_search_rooms(state)
    elif route == "CHECK_PRICES":
        response = await handle_check_prices(state)
    elif route == "MAKE_BOOKING":
        response = await handle_make_booking(state)
    elif route == "CHECK_BOOKING":
        response = await handle_check_booking(state)
    elif route == "CANCEL_BOOKING":
        response = await handle_cancel_booking(state)
    elif route == "RAG":
        response = await handle_rag(state, vectorstore, system_prompt)
    else:
        response = "I'm sorry, I didn't understand that."

    # Update history
    state["memory"]["history"].append({"role": "user", "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": response})

    return response