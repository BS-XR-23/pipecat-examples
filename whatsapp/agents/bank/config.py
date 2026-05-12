from agents.bank.agent_tools import *
from agents.bank.graph import build_graph

BANK_CONFIG = {
    "name": "bank",
    "tools": {
        "get_account_balance": get_account_balance,
        "get_user_info": get_user_info,
        "create_ticket": create_ticket,
    },
    "graph": build_graph,
    "memory": {
        "profile": {},
        "history": [],
        "ticket": {},
        "flow": {
            "active": None,
            "step": None,
            "expandable": False,
            "last_expand_offer": False,
        }
    },
    "vector_db": "/home/xr23/Projects/pipecat-examples/whatsapp/vector_store/ABB_Bank",
    "document": "/home/xr23/Projects/pipecat-examples/whatsapp/documents/ABB_Bank.txt",
    "prompt": "/home/xr23/Projects/pipecat-examples/whatsapp/system_prompt_bank.txt"
}