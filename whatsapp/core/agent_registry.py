from core.agent_config import AgentConfig
from agents.bank.config import BANK_CONFIG
from agents.hotel.config import HOTEL_CONFIG

AGENTS = {
    "bank": AgentConfig(BANK_CONFIG),
    "hotel": AgentConfig(HOTEL_CONFIG),
}

def get_agent(agent_name: str) -> AgentConfig:
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")
    return AGENTS[agent_name]