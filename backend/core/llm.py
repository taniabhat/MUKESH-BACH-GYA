import asyncio
import json
import re
import time
import threading
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic import ValidationError

from config import get_settings
from core.logging import get_logger


settings = get_settings()

logger = get_logger("core.llm")


# -------------------------------------------------------------------
# Circuit Breaker
# -------------------------------------------------------------------


class CircuitBreaker:
    """Simple circuit breaker for LLM calls.

    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("circuit_breaker.half_open")
            return self._state

    def record_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    logger.info("circuit_breaker.closed")
            else:
                self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                logger.warning(
                    "circuit_breaker.open",
                    failures=self._failure_count,
                    recovery_in=self.recovery_timeout
                )

    def allow_request(self) -> bool:
        current_state = self.state
        if current_state == self.CLOSED:
            return True
        if current_state == self.HALF_OPEN:
            return True
        return False


llm_circuit = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0
)


# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------


class LLMError(Exception):
    pass


class StructuredOutputError(Exception):
    pass


# -------------------------------------------------------------------
# Model Routing
# -------------------------------------------------------------------


def get_model(task: str) -> str:
    model_map = {
        "research":  settings.GPT_OSS_120B,
        "writing":   settings.GPT_OSS_120B,
        "humanize":  settings.GPT_OSS_120B,
        "reasoning": settings.GPT_OSS_120B,
        "code":      settings.GPT_OSS_120B,
        "fast":      settings.GPT_OSS_120B,
    }
    return model_map.get(task, settings.GPT_OSS_120B)


# -------------------------------------------------------------------
# Message Builders
# -------------------------------------------------------------------


def build_system_message(content: str) -> dict:
    return {
        "role": "system",
        "content": content
    }


def build_user_message(content: str) -> dict:
    return {
        "role": "user",
        "content": content
    }


# -------------------------------------------------------------------
# Internal HTTP Client
# -------------------------------------------------------------------


async def _post(
    endpoint: str,
    payload: dict
) -> dict:
    if not llm_circuit.allow_request():
        raise LLMError("Circuit breaker OPEN — LLM calls temporarily blocked")

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    start = time.monotonic()

    async with httpx.AsyncClient(
        timeout=settings.REQUEST_TIMEOUT_SECONDS
    ) as client:

        response = await client.post(
            f"{settings.OPENROUTER_BASE_URL}{endpoint}",
            headers=headers,
            json=payload
        )

    latency_ms = round((time.monotonic() - start) * 1000)

    if response.status_code >= 400:
        llm_circuit.record_failure()
        logger.error(
            "llm.request_failed",
            endpoint=endpoint,
            status=response.status_code,
            latency_ms=latency_ms,
            model=payload.get("model")
        )
        raise LLMError(
            f"LLM request failed: "
            f"{response.status_code} - {response.text}"
        )

    llm_circuit.record_success()

    logger.debug(
        "llm.request_ok",
        endpoint=endpoint,
        status=response.status_code,
        latency_ms=latency_ms,
        model=payload.get("model")
    )

    return response.json()


# -------------------------------------------------------------------
# Chat Completion
# -------------------------------------------------------------------


async def chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> str:

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None

    for attempt in range(settings.MAX_RETRIES):

        try:
            logger.debug(
                "llm.chat.attempt",
                model=model,
                attempt=attempt + 1,
                max_tokens=max_tokens
            )

            data = await _post(
                "/chat/completions",
                payload
            )

            content = (
                data["choices"][0]
                ["message"]
                ["content"]
            ) or (
                data["choices"][0]
                ["message"]
                .get("reasoning", "")
            ) or ""

            # Strip <think>...</think> blocks emitted by reasoning models
            content = re.sub(
                r"<think>[\s\S]*?</think>",
                "",
                content
            ).strip()

            usage = data.get("usage", {})
            logger.info(
                "llm.chat.success",
                model=model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                attempt=attempt + 1
            )

            return content

        except Exception as error:
            last_error = error
            logger.warning(
                "llm.chat.retry",
                model=model,
                attempt=attempt + 1,
                error=str(error)
            )

            await asyncio.sleep(2 ** attempt)

    logger.error(
        "llm.chat.failed",
        model=model,
        error=str(last_error)
    )

    raise LLMError(
        f"Chat request failed after retries: {last_error}"
    )


async def stream_chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096
):
    
    if not llm_circuit.allow_request():
        yield "data: [CIRCUIT_BREAKER_OPEN]\n\n"
        return

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
        try:
            async with client.stream(
                "POST",
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status_code >= 400:
                    llm_circuit.record_failure()
                    yield f"data: [ERROR {response.status_code}]\n\n"
                    return
                
                llm_circuit.record_success()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            data_json = json.loads(data_str)
                            content = data_json["choices"][0].get("delta", {}).get("content")
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
        except Exception as e:
            yield f"data: [ERROR {str(e)}]\n\n"

# -------------------------------------------------------------------
# Standard Completion
# -------------------------------------------------------------------


async def complete(
    prompt: str,
    model: str,
    max_tokens: int = 2048
) -> str:

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens
    }

    last_error = None

    for attempt in range(settings.MAX_RETRIES):

        try:
            logger.debug(
                "llm.complete.attempt",
                model=model,
                attempt=attempt + 1
            )

            data = await _post(
                "/completions",
                payload
            )

            logger.info(
                "llm.complete.success",
                model=model,
                attempt=attempt + 1
            )

            return (
                data["choices"][0]
                ["text"]
            )

        except Exception as error:
            last_error = error
            logger.warning(
                "llm.complete.retry",
                model=model,
                attempt=attempt + 1,
                error=str(error)
            )

            await asyncio.sleep(2 ** attempt)

    logger.error(
        "llm.complete.failed",
        model=model,
        error=str(last_error)
    )

    raise LLMError(
        f"Completion request failed after retries: {last_error}"
    )


# -------------------------------------------------------------------
# Structured Generation
# -------------------------------------------------------------------


async def structured_chat(
    messages: list[dict],
    model: str,
    output_schema: type[BaseModel]
) -> BaseModel:

    schema_json = json.dumps(
        output_schema.model_json_schema(),
        indent=2
    )

    system_instruction = f"""
Return ONLY valid JSON.

The JSON must strictly follow this schema:

{schema_json}

Do not include markdown.
Do not include explanations.
Do not wrap the JSON in code blocks.
"""

    enhanced_messages = [
        build_system_message(system_instruction),
        *messages
    ]

    last_error = None

    for attempt in range(settings.MAX_RETRIES):

        try:
            logger.debug(
                "llm.structured.attempt",
                model=model,
                schema=output_schema.__name__,
                attempt=attempt + 1
            )

            response = await chat(
                messages=enhanced_messages,
                model=model,
                temperature=0.2
            )

            # Strip markdown code blocks
            cleaned = re.sub(r"^```(?:json)?\s*", "", response, flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            # Remove trailing commas which break json.loads
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

            parsed = json.loads(cleaned)

            result = output_schema.model_validate(parsed)

            logger.info(
                "llm.structured.success",
                model=model,
                schema=output_schema.__name__,
                attempt=attempt + 1
            )

            return result

        except (
            json.JSONDecodeError,
            ValidationError
        ) as error:

            last_error = error

            logger.warning(
                "llm.structured.validation_failed",
                model=model,
                schema=output_schema.__name__,
                attempt=attempt + 1,
                error=str(error)
            )

            enhanced_messages.append(
                build_user_message(
                    f"""
Your previous response failed validation.

Validation Error:
{str(error)}

Return ONLY corrected JSON.
"""
                )
            )

            await asyncio.sleep(2 ** attempt)

    logger.error(
        "llm.structured.failed",
        model=model,
        schema=output_schema.__name__,
        error=str(last_error)
    )

    raise StructuredOutputError(
        f"Structured output generation failed: {last_error}"
    )


# -------------------------------------------------------------------
# Warmup
# -------------------------------------------------------------------


async def warmup_llm() -> None:
    try:
        await chat(
            messages=[
                build_user_message("ping")
            ],
            model=get_model("fast"),
            max_tokens=8
        )

    except Exception as error:
        raise LLMError(
            f"LLM warmup failed: {error}"
        )
