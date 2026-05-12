import os
from typing import Any, Dict, Optional

from loguru import logger
from ollama import AsyncClient as OllamaAsyncClient
from google import genai

DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
SUPPORTED_LLM_PROVIDERS = {"ollama", "gemini"}


def normalize_provider(provider: Optional[str]) -> str:
    if not provider:
        return DEFAULT_LLM_PROVIDER

    provider_name = provider.strip().lower()
    if provider_name not in SUPPORTED_LLM_PROVIDERS:
        logger.warning(
            f"Unknown LLM provider '{provider}'. Falling back to {DEFAULT_LLM_PROVIDER}"
        )
        return DEFAULT_LLM_PROVIDER

    return provider_name


async def generate_llm_text(
    provider: Optional[str],
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    provider_name = normalize_provider(provider)
    if provider_name == "ollama":
        return await _generate_ollama(prompt, model=model, system_prompt=system_prompt, options=options)

    if provider_name == "gemini":
        return await _generate_gemini(prompt, model=model, system_prompt=system_prompt, options=options)

    logger.warning(
        f"LLM provider '{provider_name}' is not implemented. Falling back to {DEFAULT_LLM_PROVIDER}"
    )
    return await _generate_ollama(prompt, model=model, system_prompt=system_prompt, options=options)


async def _generate_ollama(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    client = OllamaAsyncClient()
    payload: Dict[str, Any] = {
        "model": model or os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    if options is not None:
        payload["options"] = options

    result = await client.generate(**payload)
    return (result.response or "").strip()


async def _generate_gemini(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini provider")

    client = genai.Client(api_key=api_key)
    async_client = client.aio
    try:
        history = []
        if system_prompt:
            history.append(
                {
                    "role": "system",
                    "content": {
                        "parts": [
                            {"text": system_prompt}
                        ]
                    },
                }
            )

        # Create the chat session with optional system instructions.
        chat = async_client.chats.create(
            model=model or os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            history=history or None,
        )

        response = await chat.send_message(prompt, config=options or None)
        candidate = None
        if response.candidates:
            candidate = response.candidates[0].content

        if candidate is not None:
            text = getattr(candidate, "text", None)
            if text:
                return text.strip()

        text = getattr(response, "text", None)
        return (text or "").strip()

    finally:
        await async_client.aclose()
