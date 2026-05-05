"""Async OpenAI-compatible chat client for PvP bot inference.

Communicates with SGLang's chat completions endpoint.
Streaming with per-chunk timeout and exponential backoff retries.
"""

import asyncio
import logging
import re

import httpx
import openai

from core.models.pvp_models import ChatCompletionConfig, ChatMessage, ChatResult

logger = logging.getLogger(__name__)

_RETRYABLE = (
    TimeoutError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)

_THINK_COMPLETE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think(?:ing)?>.*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove <think>/<thinking> blocks from model output."""
    cleaned = _THINK_COMPLETE.sub("", text)
    for tag in ("</think>", "</thinking>"):
        if tag in cleaned:
            cleaned = cleaned.split(tag)[-1]
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    return cleaned.strip()


async def chat_completion(
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
) -> ChatResult:
    """Stream a chat completion from an OpenAI-compatible endpoint."""
    client = openai.AsyncOpenAI(
        base_url=config.base_url.rstrip("/"),
        api_key=config.api_key,
        timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0),
        max_retries=0,
    )
    try:
        return await _with_retries(client, config, messages)
    finally:
        await client.close()


async def _with_retries(
    client: openai.AsyncOpenAI,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
) -> ChatResult:
    """Execute streaming chat with exponential backoff on transient failures."""
    last_exc: BaseException | None = None
    attempts = config.max_retries + 1

    for attempt in range(attempts):
        try:
            return await _stream(client, config, messages)

        except _RETRYABLE as exc:
            last_exc = exc
            if attempt < attempts - 1:
                wait = min(2**attempt, 32)
                logger.warning(
                    "Chat attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1, attempts, type(exc).__name__, wait,
                )
                await asyncio.sleep(wait)

        except openai.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < attempts - 1:
                last_exc = exc
                await asyncio.sleep(min(2**attempt, 32))
                continue
            raise

    raise RuntimeError(f"Chat failed after {attempts} attempts: {last_exc}")


async def _stream(
    client: openai.AsyncOpenAI,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
) -> ChatResult:
    """Execute a single streaming request and return parsed result."""
    messages_dicts = [msg.model_dump() for msg in messages]

    stream = await client.chat.completions.create(
        model=config.model,
        messages=messages_dicts,
        stream=True,
        stream_options={"include_usage": True},
        temperature=config.temperature,
        seed=config.seed,
        extra_body={"max_new_tokens": config.max_new_tokens},
    )

    parts: list[str] = []
    usage: dict[str, int] | None = None

    try:
        chunk_iter = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    chunk_iter.__anext__(), timeout=config.chunk_timeout
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise TimeoutError(f"No chunk received for {config.chunk_timeout}s")

            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                usage = chunk.usage.model_dump()
    finally:
        try:
            await asyncio.wait_for(stream.response.aclose(), timeout=5.0)
        except Exception:
            pass

    raw_content = "".join(parts).strip() or None
    content = strip_think_tags(raw_content) if raw_content else None
    return ChatResult(content=content, usage=usage)
