import httpx
import os
from datetime import datetime
from loguru import logger
from typing import Optional
from dotenv import load_dotenv
from ollama import AsyncClient

load_dotenv()

INVENTORY_BASE_URL = "https://yards-classes-its-angle.trycloudflare.com"
AGENT_LOG_ENDPOINT  = f"{INVENTORY_BASE_URL}/agents/internal/agent-logs"
VENDOR_TOKEN_ENDPOINT = f"{INVENTORY_BASE_URL}/auth/vendor/token"

VENDOR_EMAIL    = os.getenv("VENDOR_EMAIL", "")
VENDOR_PASSWORD = os.getenv("VENDOR_PASSWORD", "")
VENDOR_ID       = int(os.getenv("VENDOR_ID", "0"))

_cached_token: Optional[str] = None
_ollama_client = AsyncClient()


# ── Auth ───────────────────────────────────────────────────────────────────────
async def _get_vendor_token() -> Optional[str]:
    """Fetch and cache a vendor JWT. Re-fetches if called after expiry."""
    global _cached_token
    if _cached_token:
        return _cached_token

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                VENDOR_TOKEN_ENDPOINT,
                data={                        # OAuth2 form data, not JSON
                    "username": VENDOR_EMAIL,
                    "password": VENDOR_PASSWORD,
                }
            )
            if response.status_code == 200:
                _cached_token = response.json().get("access_token")
                return _cached_token
            logger.warning(f"[AGENT LOGGER] Token fetch failed: {response.text}")
            return None
    except Exception as e:
        logger.warning(f"[AGENT LOGGER] Could not fetch vendor token: {e}")
        return None


# ── LLM description generator ──────────────────────────────────────────────────
async def _generate_description(tool_name: str, context: str, timestamp: str) -> str:
    """
    Ask gemma4:e2b to produce a single, concise log sentence.
    Falls back to the raw context string if Ollama is unreachable.
    """
    prompt = f"""You are a log summarizer for an AI chatbot system.

Given a tool name, a brief context of what happened, and the exact time, write
ONE concise sentence (maximum 20 words) that clearly describes the action.
Do NOT invent details. Use only the information provided below.

Tool    : {tool_name}
Context : {context}
Time    : {timestamp}

Respond with ONLY the one-sentence description. No preamble, no quotes."""

    try:
        res = await _ollama_client.generate(
            model="gemma4:e2b",
            prompt=prompt,
            options={"temperature": 0.1, "top_p": 0.9},
        )
        description = res.response.strip()
        logger.info(f"[AGENT LOGGER] LLM description: {description}")
        return description
    except Exception as e:
        logger.warning(f"[AGENT LOGGER] LLM description failed ({e}), using raw context")
        return context


# ── Public API ─────────────────────────────────────────────────────────────────
async def log_tool_call(
    vendor_id: int,
    agent_name: str,
    agent_type: str,
    tool_name: str,
    tool_status: str,
    short_description: str,          # human context fed to the LLM
    user_identifier: Optional[str] = None,
):
    """
    Log a tool invocation to the inventory server.

    raw_tool_input / raw_tool_output have been intentionally removed.
    A concise, LLM-generated description is created from tool_name +
    short_description + the current timestamp and stored instead.
    """
    token = await _get_vendor_token()
    if not token:
        logger.warning("[AGENT LOGGER] Skipping log — no valid vendor token.")
        return

    # ── Timestamp ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── LLM-enhanced description ───────────────────────────────────────────────
    description = await _generate_description(tool_name, short_description, timestamp)

    payload = {
        "vendor_id":         vendor_id,
        "agent_name":        agent_name,
        "agent_type":        agent_type,
        "tool_name":         tool_name,
        "tool_status":       tool_status,
        "short_description": description,
        "user_identifier":   user_identifier,
        "created_at":        timestamp,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                AGENT_LOG_ENDPOINT,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )

            # ── Token likely expired — clear cache and retry once ──────────────
            if response.status_code == 401:
                global _cached_token
                _cached_token = None
                token = await _get_vendor_token()
                if token:
                    response = await client.post(
                        AGENT_LOG_ENDPOINT,
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"},
                    )

            if response.status_code != 200:
                logger.warning(f"[AGENT LOGGER] Log creation failed: {response.text}")

    except Exception as e:
        logger.warning(f"[AGENT LOGGER] Could not reach inventory server: {e}")