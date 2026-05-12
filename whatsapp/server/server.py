#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WhatsApp WebRTC Bot Server

A FastAPI server that handles WhatsApp webhook events and manages WebRTC connections
for real-time communication with WhatsApp users. The server integrates with WhatsApp's
Business API to receive incoming calls and messages, then establishes WebRTC connections
to enable audio/video communication through a bot.

Key features:
- WhatsApp webhook verification and message handling
- WebRTC connection management with ICE server support
- Graceful shutdown handling with signal management
- Background task processing for bot instances
- Connection cleanup and resource management

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
from typing import Dict, Any
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional
import aiohttp
import uvicorn
from dotenv import load_dotenv
from google import genai
from collections import defaultdict, deque
from db.database import SessionLocal
from bot import run_bot
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
from pipecat.transports.whatsapp.client import WhatsAppClient
from langchain_ollama import OllamaEmbeddings
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ollama import AsyncClient
from core.agent_registry import get_agent, AGENTS
from utils.helpers import normalize_session_id
from whatsapp_message.schemas import (
    WhatsAppMessageWebhookRequest,
)


load_dotenv(override=True)

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_WEBHOOK_VERIFICATION_TOKEN = os.getenv("WHATSAPP_WEBHOOK_VERIFICATION_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

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
ollama_client: Optional[AsyncClient] = None

shutdown_event = asyncio.Event()

MAX_HISTORY = 30
user_context = defaultdict(dict)
session_memory = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
# session_to_agent: Dict[str, str] = {} 
session_to_agent = {}
session_memory = {}

# DEFAULT_AGENT = "bank"
DEFAULT_AGENT = "hotel"
app = FastAPI(
    title="Unified WhatsApp Agent Server",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def signal_handler() -> None:
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully.

    Sets the shutdown event to initiate graceful server shutdown.
    This allows the server to complete ongoing requests and cleanup resources.
    """
    logger.info("Received shutdown signal, initiating graceful shutdown...")
    shutdown_event.set()


def init_memory(session_id: str, agent):
    if session_id not in user_context:
        user_context[session_id] = json.loads(
            json.dumps(agent.memory_template)
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whatsapp_client, ollama_client

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


app.router.lifespan_context = lifespan


async def _generate_llm_response(session_id: str, message: str, agent_name: str):
    db = SessionLocal()
    try:
        agent = get_agent(agent_name)

        if session_id not in user_context:
            user_context[session_id] = json.loads(
                json.dumps(agent.memory_template)
            )

        memory = user_context[session_id]

        graph = agent.get_graph()

        state = {
            "session_id": session_id,
            "message": message,
            "agent_name": agent_name,
            "agent": agent,
            "intent": "RAG",
            "response": None,
            "memory": memory  
        }

        result = await graph.ainvoke(state)

        if result.get("memory"):
            user_context[session_id] = result["memory"]

        final_text = result.get("response") or "Sorry, I couldn't process your request."

        user_context[session_id].setdefault("history", [])

        if not any(
            m.get("content") == message
            for m in user_context[session_id]["history"][-2:]
        ):
            user_context[session_id]["history"].append(
                {"role": "user", "content": message}
            )

        user_context[session_id]["history"].append(
            {"role": "assistant", "content": final_text}
        )

        if len(user_context[session_id]["history"]) > MAX_HISTORY:
            user_context[session_id]["history"] = user_context[session_id]["history"][-MAX_HISTORY:]

        return final_text

    except Exception as e:
        logger.error(f"GRAPH ERROR: {e}")
        return "Something went wrong."

    finally:
        db.close()


async def _handle_message_webhook(body: WhatsAppMessageWebhookRequest):

    try:
        for entry in body.entry:
            for change in entry.changes:

                if change.field != "messages" or not change.value.messages:
                    continue

                for msg in change.value.messages:

                    if msg.type != "text" or not msg.text:
                        continue

                    # -------------------- NORMALIZED SESSION ID --------------------
                    sender = normalize_session_id(msg.from_)
                    user_text = msg.text.body.strip()

                    if not user_text:
                        continue

                    # -------------------- ROUTING (LOCKED PER SESSION) --------------------
                    agent_name = session_to_agent.get(sender)

                    if not agent_name:
                        agent_name = DEFAULT_AGENT
                        session_to_agent[sender] = agent_name
                        logger.info(f"[ROUTING INIT] {sender} -> {agent_name}")
                    else:
                        logger.info(f"[ROUTING LOCKED] {sender} -> {agent_name}")

                    # -------------------- AGENT INIT --------------------
                    try:
                        agent = get_agent(agent_name)
                    except Exception as e:
                        logger.warning(f"[AGENT FALLBACK] {e}")
                        agent_name = DEFAULT_AGENT
                        agent = get_agent(agent_name)
                        session_to_agent[sender] = agent_name

                    # -------------------- SESSION MEMORY --------------------
                    if sender not in session_memory:
                        session_memory[sender] = deque(maxlen=MAX_HISTORY)

                    # -------------------- LLM CALL --------------------
                    llm_response = await _generate_llm_response(
                            session_id=normalize_session_id(sender),
                            message=user_text,
                            agent_name=agent_name
                        )

                    if not llm_response or not llm_response.strip():
                        llm_response = "Sorry, I couldn't process your request. Please try again."

                    await _send_whatsapp_text_message(sender, llm_response.strip())

        return {"status": "success"}

    except Exception as e:
        logger.error(f"[WEBHOOK ERROR] {e}")
        return {"status": "error-handled"}


@app.post("/configure-agent")
async def configure_agent(payload: dict):

    agent_type = payload.get("agent_type")

    if not agent_type or agent_type not in AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent_type: {agent_type}. Allowed: {list(AGENTS.keys())}"
        )

    AGENTS[agent_type].reload_paths(
        vector_db_id       = payload.get("vector_db_id"),   # ✅ FIXED
        document_path      = payload.get("document_address"),
        system_prompt_raw  = payload.get("system_prompt"),
        hash_address       = payload.get("hash_address"),
        llm_model          = payload.get("llm_model"),
        embedding_model    = payload.get("embedding_model"),
    )

    logger.info(f"[AGENT CONFIGURED] {agent_type} reloaded")

    HARDCODED_SESSION = "01701001398"
    normalized_session = normalize_session_id(HARDCODED_SESSION)

    session_to_agent[normalized_session] = agent_type

    logger.info(
        f"[HARDCODED SESSION ASSIGNED] {normalized_session} -> {agent_type}"
    )

    return {
        "status": "success",
        "agent_type": agent_type,
        "agent_name": payload.get("agent_name", agent_type),
        "session_bound": HARDCODED_SESSION
    }


@app.post("/set-agent-session")
async def set_agent_session(payload: dict):
    """
    Maps a session_id to a specific agent name.
    Kept separate from /configure-agent so concerns don't mix.
    """
    session_id = normalize_session_id(payload.get("session_id"))
    agent_name = payload.get("agent_name")

    if not agent_name:
        raise HTTPException(status_code=400, detail="agent_name required")

    if session_id:
        session_to_agent[session_id] = agent_name
        logger.info(f"[SESSION ASSIGNED] {session_id} -> {agent_name}")

    return {
        "status":     "success",
        "agent_name": agent_name
    }


async def _send_whatsapp_text_message(to: str, text: str) -> dict:
    """Send a text message to a WhatsApp user via the WhatsApp Cloud API."""
    if not whatsapp_client:
        raise RuntimeError("WhatsApp client is not initialized")

    if not text or not text.strip():
        logger.warning("Attempted to send empty WhatsApp message. Skipping.")
        return {"status": "skipped-empty"}

    send_url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    # send_url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text.strip()},
    }
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    async with whatsapp_client._whatsapp_api._session.post(
        send_url, headers=headers, json=payload
    ) as response:
        data = await response.json()
        if response.status >= 300:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to send WhatsApp message",
                    "status": response.status,
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
            params=params, expected_verification_token=WHATSAPP_WEBHOOK_VERIFICATION_TOKEN
        )
        logger.info("Webhook verification successful")
        return result
    except ValueError as e:
        logger.warning(f"Webhook verification failed: {e}")
        raise HTTPException(status_code=403, detail="Verification failed")


## From Chatbot Inventory Dashboard - assigns agent to session when vendor selects an agent for a user


@app.get(
    "/",
    summary="Verify WhatsApp webhook",
    description="Handles WhatsApp webhook verification requests from Meta",
)
async def verify_webhook(request: Request):
    return await _verify_whatsapp_webhook(request)


@app.get(
    "/message",
    summary="Verify WhatsApp message webhook",
    description="Handles WhatsApp webhook verification requests for the message endpoint",
)
async def verify_message_webhook(request: Request):
    return await _verify_whatsapp_webhook(request)


async def _handle_call_webhook(body: WhatsAppWebhookRequest, background_tasks: BackgroundTasks):
    """Process WhatsApp call webhook payload and start the call bot."""
    if body.object != "whatsapp_business_account":
        logger.warning(f"Invalid webhook object type: {body.object}")
        raise HTTPException(status_code=400, detail="Invalid object type")

    logger.info(f"Processing WhatsApp call webhook: {body.dict()}")

    async def connection_callback(connection: SmallWebRTCConnection):
        """Handle new WebRTC connections from WhatsApp calls.

        Called when a WebRTC connection is established for a WhatsApp call.
        Spawns a bot instance to handle the conversation.

        Args:
            connection: The established WebRTC connection
        """
        try:
            logger.info(f"Starting bot for WebRTC connection: {connection.pc_id}")
            background_tasks.add_task(run_bot, connection)
            logger.debug(f"Bot task queued successfully for connection: {connection.pc_id}")
        except Exception as e:
            logger.error(f"Failed to start bot for connection {connection.pc_id}: {e}")
            try:
                await connection.disconnect()
                logger.debug(f"Connection {connection.pc_id} disconnected after error")
            except Exception as disconnect_error:
                logger.error(f"Failed to disconnect connection after error: {disconnect_error}")

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
 

@app.post("/")
async def root_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    if payload.get("object") != "whatsapp_business_account":
        return {"status": "ignored"}

    entries = payload.get("entry", [])
    if not entries:
        return {"status": "ignored"}

    message_fields = False
    call_fields = False

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
        background_tasks.add_task(_handle_call_webhook, body, background_tasks)
        return {"status": "accepted"}

    return {"status": "ignored"}


@app.post(
    "/call",
    summary="Handle WhatsApp call webhooks",
    description="Processes incoming WhatsApp voice call webhook events and starts call bots",
)
async def whatsapp_webhook(body: WhatsAppWebhookRequest, background_tasks: BackgroundTasks):
    return await _handle_call_webhook(body, background_tasks)


@app.post(
    "/message",
    summary="Handle WhatsApp incoming text message webhooks",
    description=(
        "Receives incoming WhatsApp text message webhook events, sends them to the LLM, "
        "and replies back to the WhatsApp sender."
    ),
)
async def message_webhook(body: WhatsAppMessageWebhookRequest):
    return await _handle_message_webhook(body)


async def run_server_with_signal_handling(host: str, port: int) -> None:
    """Run the FastAPI server with proper signal handling.

    Sets up signal handlers for graceful shutdown and manages the server lifecycle.
    Handles SIGINT (Ctrl+C) and SIGTERM signals to ensure proper cleanup.

    Args:
        host: The host address to bind the server to
        port: The port number to listen on
    """
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Configure and create the server
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
    )
    server = uvicorn.Server(config)

    # Start server in background task
    server_task = asyncio.create_task(server.serve())
    logger.info(f"WhatsApp WebRTC server started on {host}:{port}")
    logger.info("Press Ctrl+C to stop the server")

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Initiate graceful shutdown
    logger.info("Shutting down server.")

    # Cleanup WhatsApp client resources
    if whatsapp_client:
        await whatsapp_client.terminate_all_calls()

    # Stop the server
    server.should_exit = True
    await server_task
    logger.info("Server shutdown completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WhatsApp WebRTC Bot Server - Handles WhatsApp webhooks and WebRTC connections"
    )
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    # Validate configuration
    logger.info("Starting WhatsApp WebRTC Bot Server...")
    logger.debug(f"Configuration: host={args.host}, port={args.port}, verbose={args.verbose}")

    # Run the server
    try:
        asyncio.run(run_server_with_signal_handling(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
