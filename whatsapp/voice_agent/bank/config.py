from voice_agent.bank.agent_tools import *
from voice_agent.bank.intent_rag import process_intent_rag

BANK_VOICE_CONFIG = {
    "name": "bank",
    "tools": {
        "get_account_balance": get_account_balance,
        "get_user_info": get_user_info,
        "create_ticket": create_ticket,
    },
    "process_intent_rag": process_intent_rag,
    "memory_template": {
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