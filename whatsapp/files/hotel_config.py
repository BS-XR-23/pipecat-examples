from agents.hotel.agent_tools import *
from agents.hotel.graph import build_graph

HOTEL_CONFIG = {
    "name": "hotel",
    "tools": {
        "search_available_rooms": search_available_rooms,
        "get_room_prices": get_room_prices,
        "make_booking": make_booking,
        "check_booking_status": check_booking_status,
        "cancel_booking": cancel_booking,
    },
    "graph": build_graph,
    "memory": {
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
    "vector_db": ".../Sayeman_hotel",
    "document": ".../Sayeman_hotel.txt",
    "prompt": ".../system_prompt_hotel.txt"
}