"""WhatsApp WebRTC Bot Server

A FastAPI server that handles WhatsApp webhook events and manages WebRTC connections
for real-time communication with WhatsApp users.

Environment Variables Required:
- WHATSAPP_TOKEN: WhatsApp Business API access token
- WHATSAPP_WEBHOOK_VERIFICATION_TOKEN: Token for webhook verification
- WHATSAPP_PHONE_NUMBER_ID: WhatsApp Business phone number ID

Usage:
    python server.py --host 0.0.0.0 --port 8080 --verbose
"""

import os
import asyncio
import json
import argparse
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from collections import defaultdict

import aiohttp
import uvicorn
from dotenv import load_dotenv
from loguru import logger
from ollama import AsyncClient
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
from pipecat.transports.whatsapp.client import WhatsAppClient

from bot import run_bot
from core.agent_registry import get_agent
from whatsapp_message.schemas import (
    WhatsAppMessageWebhookRequest,
)

load_dotenv(override=True)

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_WEBHOOK_VERIFICATION_TOKEN = os.getenv("WHATSAPP_WEBHOOK_VERIFICATION_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

if not all([WHATSAPP_TOKEN, WHATSAPP_WEBHOOK_VERIFICATION_TOKEN, WHATSAPP_PHONE_NUMBER_ID]):
    missing_vars = [
        var
        for var, val in [
            ("WHATSAPP_TOKEN", WHATSAPP_TOKEN),
            ("WHATSAPP_WEBHOOK_VERIFICATION_TOKEN", WHATSAPP_WEBHOOK_VERIFICATION_TOKEN),
            ("WHATSAPP_PHONE_NUMBER_ID", WHATSAPP_PHONE_NUMBER_ID),
        ]
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# -------------------- GLOBAL STATE --------------------
whatsapp_client: Optional[Any] = None
shutdown_event = asyncio.Event()

MAX_HISTORY = 30

# session_id -> agent memory dict
user_context: Dict[str, Any] = defaultdict(dict)
# sender phone -> assigned agent name
session_to_agent: Dict[str, str] = {}

DEFAULT_AGENT = "hotel"


# -------------------- LIFESPAN --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whatsapp_client

    async with aiohttp.ClientSession() as session:
        whatsapp_client = WhatsAppClient(
            whatsapp_token=WHATSAPP_TOKEN,
            phone_number_id=WHATSAPP_PHONE_NUMBER_ID,
            session=session
        )
        logger.info("WhatsApp client initialized")

        ollama_client = AsyncClient()

        async def warmup():
            try:
                await ollama_client.generate(
                    model="gemma4:e4b",
                    prompt="warmup",
                    stream=False
                )
                logger.info("LLM warmed up")
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")

        asyncio.create_task(warmup())

        yield

        logger.info("Shutting down...")
        if whatsapp_client:
            await whatsapp_client.terminate_all_calls()


app = FastAPI(
    title="Unified WhatsApp Agent Server",
    version="2.0.0",
    lifespan=lifespan,
)


# -------------------- SIGNAL HANDLING --------------------
def signal_handler() -> None:
    logger.info("Received shutdown signal, initiating graceful shutdown...")
    shutdown_event.set()


# -------------------- CORE LLM RESPONSE --------------------
async def _generate_llm_response(session_id: str, message: str, agent_name: str) -> str:
    try:
        agent = get_agent(agent_name)

        # Initialise memory for new sessions
        if session_id not in user_context:
            user_context[session_id] = json.loads(json.dumps(agent.memory_template))

        memory = user_context[session_id]
        graph  = agent.get_graph()

        state = {
            "session_id": session_id,
            "message":    message,
            "agent_name": agent_name,
            "intent":     "RAG",
            "response":   None,
            "memory":     memory,
        }

        result = await graph.ainvoke(state)

        # Persist updated memory returned by the graph
        if result.get("memory"):
            user_context[session_id] = result["memory"]

        final_text = result.get("response") or "Sorry, I couldn't process your request."

        # ── Uniform history tracking (graph nodes do NOT append history themselves) ──
        history = user_context[session_id].setdefault("history", [])
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": final_text})

        # Trim to MAX_HISTORY
        if len(history) > MAX_HISTORY:
            user_context[session_id]["history"] = history[-MAX_HISTORY:]

        return final_text

    except Exception as e:
        logger.error(f"GRAPH ERROR: {e}")
        return "Something went wrong."


# -------------------- WEBHOOK HANDLERS --------------------
async def _handle_message_webhook(body: WhatsAppMessageWebhookRequest):
    try:
        for entry in body.entry:
            for change in entry.changes:
                if change.field != "messages" or not change.value.messages:
                    continue

                for msg in change.value.messages:
                    if msg.type != "text" or not msg.text:
                        continue

                    sender    = msg.from_
                    user_text = msg.text.body.strip()

                    if not user_text:
                        continue

                    agent_name = session_to_agent.get(sender, DEFAULT_AGENT)

                    # Validate agent; fall back to default if unknown
                    try:
                        get_agent(agent_name)
                    except Exception as e:
                        logger.warning(f"[AGENT FALLBACK] {e}")
                        agent_name = DEFAULT_AGENT

                    logger.info(f"[ROUTING] {sender} -> {agent_name}")

                    llm_response = await _generate_llm_response(
                        session_id=sender,
                        message=user_text,
                        agent_name=agent_name,
                    )

                    if not llm_response or not llm_response.strip():
                        llm_response = "Sorry, I couldn't process your request. Please try again."

                    await _send_whatsapp_text_message(sender, llm_response.strip())

        return {"status": "success"}

    except Exception as e:
        logger.error(f"[WEBHOOK ERROR] {e}")
        return {"status": "error-handled"}


async def _send_whatsapp_text_message(to: str, text: str) -> dict:
    """Send a text message to a WhatsApp user via the WhatsApp Cloud API."""
    if not whatsapp_client:
        raise RuntimeError("WhatsApp client is not initialized")

    if not text or not text.strip():
        logger.warning("Attempted to send empty WhatsApp message. Skipping.")
        return {"status": "skipped-empty"}

    send_url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload  = {
        "messaging_product": "whatsapp",
        "to":   to,
        "type": "text",
        "text": {"body": text.strip()},
    }
    headers = {
        "Authorization":  f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type":   "application/json",
    }

    async with whatsapp_client._whatsapp_api._session.post(
        send_url, headers=headers, json=payload
    ) as response:
        data = await response.json()

    if response.status >= 300:
        raise HTTPException(
            status_code=500,
            detail={
                "error":    "Failed to send WhatsApp message",
                "status":   response.status,
                "response": data,
            },
        )
    return data


async def _verify_whatsapp_webhook(request: Request) -> int:
    """Verify a WhatsApp webhook request from Meta."""
    params = dict(request.query_params)
    logger.debug(f"Webhook verification request received with params: {list(params.keys())}")

    try:
        result = await whatsapp_client.handle_verify_webhook_request(
            params=params,
            expected_verification_token=WHATSAPP_WEBHOOK_VERIFICATION_TOKEN,
        )
        logger.info("Webhook verification successful")
        return result
    except ValueError as e:
        logger.warning(f"Webhook verification failed: {e}")
        raise HTTPException(status_code=403, detail="Verification failed")


# _handle_call_webhook no longer needs background_tasks; uses asyncio.create_task directly
async def _handle_call_webhook(body: WhatsAppWebhookRequest):
    """Process WhatsApp call webhook payload and start the call bot."""
    if body.object != "whatsapp_business_account":
        logger.warning(f"Invalid webhook object type: {body.object}")
        raise HTTPException(status_code=400, detail="Invalid object type")

    logger.info(f"Processing WhatsApp call webhook: {body.dict()}")

    async def connection_callback(connection: SmallWebRTCConnection):
        try:
            logger.info(f"Starting bot for WebRTC connection: {connection.pc_id}")
            asyncio.create_task(run_bot(connection))
            logger.debug(f"Bot task queued for connection: {connection.pc_id}")
        except Exception as e:
            logger.error(f"Failed to start bot for connection {connection.pc_id}: {e}")
            try:
                await connection.disconnect()
            except Exception as disconnect_error:
                logger.error(f"Failed to disconnect after error: {disconnect_error}")

    try:
        result = await whatsapp_client.handle_webhook_request(body, connection_callback)
        logger.debug(f"Webhook processed successfully: {result}")
        return {"status": "success", "message": "Call webhook processed successfully"}

    except ValueError as ve:
        logger.warning(f"Invalid webhook request format: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except Exception as e:
        logger.error(f"Internal error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing webhook")


# -------------------- ROUTES --------------------
@app.post("/set-agent-session")
async def set_agent_session(payload: dict):
    """Called from vendor dashboard to assign an agent to a user session."""
    session_id = payload.get("session_id")
    agent_name = payload.get("agent_name")

    if not session_id or not agent_name:
        raise HTTPException(status_code=400, detail="session_id and agent_name required")

    # Validate agent name before accepting
    try:
        get_agent(agent_name)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name}")

    session_to_agent[session_id] = agent_name
    logger.info(f"[AGENT ASSIGNED] {session_id} -> {agent_name}")
    return {"status": "success", "session_id": session_id, "agent_name": agent_name}


@app.get("/", summary="Verify WhatsApp webhook")
async def verify_webhook(request: Request):
    return await _verify_whatsapp_webhook(request)


@app.get("/message", summary="Verify WhatsApp message webhook")
async def verify_message_webhook(request: Request):
    return await _verify_whatsapp_webhook(request)


@app.post("/")
async def root_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    if payload.get("object") != "whatsapp_business_account":
        return {"status": "ignored"}

    entries = payload.get("entry", [])
    if not entries:
        return {"status": "ignored"}

    message_fields = False
    call_fields    = False

    for entry in entries:
        for change in entry.get("changes", []):
            field = change.get("field")
            if field == "messages":
                message_fields = True
            elif field == "calls":
                call_fields = True

    if message_fields:
        body = WhatsAppMessageWebhookRequest.model_validate(payload)
        background_tasks.add_task(_handle_message_webhook, body)
        return {"status": "accepted"}

    if call_fields:
        body = WhatsAppWebhookRequest.model_validate(payload)
        # _handle_call_webhook no longer needs background_tasks
        background_tasks.add_task(_handle_call_webhook, body)
        return {"status": "accepted"}

    return {"status": "ignored"}


@app.post("/call", summary="Handle WhatsApp call webhooks")
async def whatsapp_webhook(body: WhatsAppWebhookRequest):
    return await _handle_call_webhook(body)


@app.post("/message", summary="Handle WhatsApp incoming text message webhooks")
async def message_webhook(body: WhatsAppMessageWebhookRequest):
    return await _handle_message_webhook(body)


# -------------------- SERVER RUNNER --------------------
async def run_server_with_signal_handling(host: str, port: int) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    config = uvicorn.Config(app, host=host, port=port, log_config=None)
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    logger.info(f"WhatsApp WebRTC server started on {host}:{port}")
    logger.info("Press Ctrl+C to stop the server")

    await shutdown_event.wait()
    logger.info("Shutting down server.")

    if whatsapp_client:
        await whatsapp_client.terminate_all_calls()

    server.should_exit = True
    await server_task
    logger.info("Server shutdown completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhatsApp WebRTC Bot Server")
    parser.add_argument("--host", default="localhost", help="Host (default: localhost)")
    parser.add_argument("--port", type=int, default=7860,  help="Port (default: 7860)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    logger.info("Starting WhatsApp WebRTC Bot Server...")
    logger.debug(f"Configuration: host={args.host}, port={args.port}, verbose={args.verbose}")

    try:
        asyncio.run(run_server_with_signal_handling(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
