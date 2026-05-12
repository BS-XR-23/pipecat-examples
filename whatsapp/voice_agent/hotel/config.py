from voice_agent.hotel.agent_tools import *
from voice_agent.hotel.intent_rag import process_intent_rag

HOTEL_VOICE_CONFIG = {
    "name": "hotel",
    "tools": {
        "search_available_rooms": search_available_rooms,
        "get_room_prices": get_room_prices,
        "make_booking": make_booking,
        "check_booking_status": check_booking_status,
        "cancel_booking": cancel_booking,
    },
    "process_intent_rag": process_intent_rag,
    "memory_template": {
        "profile": {},
        "history": [],
        "booking": {},
        "search": {},
        "flow": {
            "active": None,
            "step": None,
            "expandable": False,
            "last_expand_offer": False,
        }
    },
    "vector_db": "/home/xr23/Projects/pipecat-examples/whatsapp/vector_store/Sayeman_hotel",
    "document": "/home/xr23/Projects/pipecat-examples/whatsapp/documents/Sayeman_hotel.txt",
    "prompt": "/home/xr23/Projects/pipecat-examples/whatsapp/system_prompt_hotel.txt"
}