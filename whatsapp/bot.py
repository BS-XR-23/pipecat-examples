import asyncio
import copy
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallResultProperties
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from voice_agent_registry import get_voice_agent

load_dotenv(override=True)

MAX_HISTORY = 30

# Shared parameter definition — every agent tool takes the same single argument
_USER_MESSAGE_PROPERTY = {
    "user_message": {
        "type": "string",
        "description": "The user's exact transcribed spoken message.",
    }
}


def _build_tools_schema(function_name: str, description: str) -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=function_name,
                description=description,
                properties=_USER_MESSAGE_PROPERTY,
                required=["user_message"],
            )
        ]
    )


AGENT_TOOL_REGISTRY: dict[str, dict] = {
    "bank": {
        "function_name": "query_banking_agent",
        "tools_schema": _build_tools_schema(
            "query_banking_agent",
            "Process the user's spoken message through the ABB Bank agent pipeline. "
            "Call this for EVERY user message. Speak the returned 'response' exactly as-is.",
        ),
        "fallback_prompt": "You are a voice assistant for ABB Bank.",
    },
    "hotel": {
        "function_name": "query_hotel_agent",
        "tools_schema": _build_tools_schema(
            "query_hotel_agent",
            "Process the user's spoken message through the hotel concierge agent pipeline. "
            "Call this for EVERY user message. Speak the returned 'response' exactly as-is.",
        ),
        "fallback_prompt": "You are a voice assistant for the hotel concierge service.",
    },
}


def _get_tool_config(agent_name: str) -> dict:
    config = AGENT_TOOL_REGISTRY.get(agent_name)
    if config is None:
        logger.warning(
            f"[run_bot] no tool config for agent={agent_name!r}, falling back to 'bank'"
        )
        config = AGENT_TOOL_REGISTRY["bank"]
    return config


def _build_tool_addendum(function_name: str) -> str:
    return (
        "\n\n--- VOICE BEHAVIOR INSTRUCTIONS ---\n"
        "You are operating as a voice assistant over a phone call.\n"
        f"When the user speaks, call the {function_name} function with the user's exact words as the user_message argument.\n"
        "After the function returns, speak the 'response' value aloud, word for word.\n"
        "Do NOT call the function again after receiving the response.\n"
        "Do NOT paraphrase, summarize, or modify the response.\n"
        "Do NOT answer from your own knowledge — always rely on the function result.\n"
        "Speak naturally and conversationally."
    )


async def run_bot(
    webrtc_connection,
    agent_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    agent_name = agent_name or os.getenv("VOICE_AGENT", "bank")
    session_id = session_id or str(uuid.uuid4())

    logger.info(f"[run_bot] new call | agent={agent_name} | session={session_id}")

    # ── Resolve tool config for this agent ─────────────────────────────────────
    tool_config = _get_tool_config(agent_name)
    function_name: str = tool_config["function_name"]
    tools_schema: ToolsSchema = tool_config["tools_schema"]

    # ── Load agent resources ───────────────────────────────────────────────────
    voice_agent = get_voice_agent(agent_name)
    process_intent_rag = voice_agent["process_intent_rag"]

    # Resources loaded lazily on first message to avoid blocking WebRTC connection
    agent_config = None
    system_prompt = None
    vectorstore = None

    async def _lazy_load_resources():
        """Load RAG resources on first message, not during init."""
        nonlocal agent_config, system_prompt, vectorstore
        
        if agent_config is not None:
            return  # Already loaded
        
        try:
            from core.agent_config import AgentConfig
            agent_config = AgentConfig(voice_agent)
            system_prompt = agent_config.get_system_prompt()
            vectorstore = agent_config.get_vectorstore()
            logger.info(
                f"[run_bot] loaded RAG config for agent={agent_name} | "
                f"vector_db={agent_config.vector_db_id} | "
                f"prompt_path={agent_config.system_prompt_path}"
            )
        except Exception as e:
            logger.warning(f"[run_bot] RAG config load failed ({e}), using fallback")
            system_prompt = tool_config["fallback_prompt"]
            vectorstore = None

    # ── Per-session memory ─────────────────────────────────────────────────────
    memory: dict = copy.deepcopy(
        voice_agent.get(
            "memory_template",
            {
                "profile": {},
                "history": [],
                "ticket": {},
                "flow": {
                    "active": None,
                    "step": None,
                    "expandable": False,
                    "last_expand_offer": False,
                },
            },
        )
    )

    # ── GeminiLive function call handler ───────────────────────────────────────
    async def handle_agent_query(params) -> None:
        user_message = (params.arguments.get("user_message") or "").strip()

        if not user_message:
            await params.result_callback(
                {"response": "I didn't catch that. Could you please repeat?"},
                properties=FunctionCallResultProperties(run_llm=True),
            )
            return

        # Lazy-load RAG resources on first message
        await _lazy_load_resources()

        state = {
            "session_id": session_id,
            "message": user_message,
            "memory": memory,
        }

        try:
            result = await process_intent_rag(state, vectorstore, system_prompt)
            response_text = result[0] if isinstance(result, tuple) else result
            if len(memory.get("history", [])) > MAX_HISTORY:
                memory["history"] = memory["history"][-MAX_HISTORY:]
        except Exception as e:
            logger.error(f"[{function_name}] RAG error: {e}")
            response_text = "Sorry, something went wrong. Please try again."

        await params.result_callback(
            {"response": response_text},
            properties=FunctionCallResultProperties(run_llm=True),
        )

    # ── Transport ──────────────────────────────────────────────────────────────
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=1,  # ✅ FIXED
        ),
    )

    # ── LLM ───────────────────────────────────────────────────────────────────

    # Use a minimal system prompt for startup; full context will be in the tool call
    startup_system_prompt = (
        f"You are a voice assistant. Listen to the user and call the {function_name} tool "
        f"with their exact message. Speak the returned response aloud."
    )

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GeminiLiveLLMService.Settings(
            voice="Puck",
            system_instruction=startup_system_prompt,
        ),
        tools=tools_schema,
    )

    llm.register_function(function_name, handle_agent_query)

    # ── Context ───────────────────────────────────────────────────────────────────
    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": "Hello",  # triggers _create_initial_response on Gemini connect
            }
        ]
    )
    # DO NOT call context.set_tools() — tools are already at init level above

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.85,
                    start_secs=0.6,
                    stop_secs=0.5,
                    min_volume=0.65,
                )
            ),
        ),
    )

    # ── Pipeline ───────────────────────────────────────────────────────────────
    # mic → VAD aggregator → GeminiLive (STT → tool call → TTS) → speaker
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )

    # ── Event handlers ─────────────────────────────────────────────────────────
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(
            f"[run_bot] client connected | agent={agent_name} | session={session_id}"
        )
        asyncio.create_task(_lazy_load_resources())
        # Queue LLMRunFrame to kick off the conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[run_bot] client disconnected | session={session_id}")
        await task.cancel()

    logger.info(f"[run_bot] waiting for WebRTC connection | session={session_id}")
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)