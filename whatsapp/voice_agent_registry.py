from voice_agent.bank.config import BANK_VOICE_CONFIG
from voice_agent.hotel.config import HOTEL_VOICE_CONFIG
from loguru import logger

VOICE_AGENTS = {
    "bank": BANK_VOICE_CONFIG,
    "hotel": HOTEL_VOICE_CONFIG,
}

# Runtime overrides set by /configure-agent
_voice_agent_overrides: dict = {}


def reload_voice_agent_paths(
    agent_name: str,
    vector_db_id: str = None,
    document_path: str = None,
    system_prompt: str = None,
    hash_address: str = None,
    llm_model: str = None,
    embedding_model: str = None,
) -> None:
    """Store path overrides for a voice agent. Called by /configure-agent."""
    if agent_name not in VOICE_AGENTS:
        raise ValueError(f"Unknown voice agent: {agent_name}")

    _voice_agent_overrides[agent_name] = {k: v for k, v in {
        "vector_db":         vector_db_id,
        "document":          document_path,
        "system_prompt_raw": system_prompt,   # ✅ fixed: was "prompt", must match AgentConfig key
        "hash_address":      hash_address,
        "llm_model":         llm_model,
        "embedding_model":   embedding_model,
    }.items() if v is not None}

    logger.info(f"[voice_agent_registry] overrides stored for {agent_name!r}: {list(_voice_agent_overrides[agent_name].keys())}")


def get_voice_agent(agent_name: str) -> dict:
    if agent_name not in VOICE_AGENTS:
        raise ValueError(f"Unknown voice agent: {agent_name}")

    # Shallow-copy base config so originals are never mutated
    agent = dict(VOICE_AGENTS[agent_name])

    # Merge runtime overrides on top
    overrides = _voice_agent_overrides.get(agent_name, {})
    if overrides:
        logger.info(f"[get_voice_agent] applying overrides for {agent_name!r}: {list(overrides.keys())}")
        agent.update(overrides)
        logger.debug(f"[get_voice_agent] applied overrides for {agent_name!r}")

    logger.info(f"[get_voice_agent] returning config for {agent_name!r} | vector_db={agent.get('vector_db')} | prompt={agent.get('system_prompt_raw')}")

    return agent