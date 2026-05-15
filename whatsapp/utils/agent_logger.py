import httpx
import os
import json
from loguru import logger
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

INVENTORY_BASE_URL = "https://industry-suddenly-tear-alto.trycloudflare.com"
AGENT_LOG_ENDPOINT = f"{INVENTORY_BASE_URL}/agents/internal/agent-logs"
VENDOR_TOKEN_ENDPOINT = f"{INVENTORY_BASE_URL}/auth/vendor/token"

# Store in your .env — these are the credentials of the vendor 
# that owns this WhatsApp chatbot deployment
VENDOR_EMAIL = os.getenv("VENDOR_EMAIL", "")
VENDOR_PASSWORD = os.getenv("VENDOR_PASSWORD", "")
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))

_cached_token: Optional[str] = None


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
                    "password": VENDOR_PASSWORD
                }
            )
            if response.status_code == 200:
                _cached_token = response.json().get("access_token")
                return _cached_token
            else:
                logger.warning(f"[AGENT LOGGER] Token fetch failed: {response.text}")
                return None
    except Exception as e:
        logger.warning(f"[AGENT LOGGER] Could not fetch vendor token: {e}")
        return None


async def log_tool_call(
    vendor_id: int,
    agent_name: str,
    agent_type: str,
    tool_name: str,
    tool_status: str,
    short_description: str,
    user_identifier: Optional[str] = None,
    raw_tool_input: Optional[dict] = None,
    raw_tool_output: Optional[dict] = None,
):
    token = await _get_vendor_token()
    if not token:
        logger.warning("[AGENT LOGGER] Skipping log — no valid vendor token.")
        return

    payload = {
        "vendor_id": vendor_id,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "tool_name": tool_name,
        "tool_status": tool_status,
        "short_description": short_description,
        "user_identifier": user_identifier,
        "raw_tool_input": json.dumps(raw_tool_input) if raw_tool_input else None,
        "raw_tool_output": json.dumps(raw_tool_output) if raw_tool_output else None,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                AGENT_LOG_ENDPOINT,
                json=payload,
                headers={"Authorization": f"Bearer {token}"}
            )

            # Token likely expired — clear cache and retry once
            if response.status_code == 401:
                global _cached_token
                _cached_token = None
                token = await _get_vendor_token()
                if token:
                    response = await client.post(
                        AGENT_LOG_ENDPOINT,
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"}
                    )

            if response.status_code != 200:
                logger.warning(f"[AGENT LOGGER] Log creation failed: {response.text}")

    except Exception as e:
        logger.warning(f"[AGENT LOGGER] Could not reach inventory server: {e}")