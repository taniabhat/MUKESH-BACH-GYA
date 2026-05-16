"""
Shared utilities for the Research Discovery Module.

Provides:
- get_logger()        — structured logger with configurable level
- get_http_client()   — async httpx client factory with retries
- api_retry()         — tenacity retry decorator for API calls
- TokenBucket         — async rate-limiter for Semantic Scholar
- Timer               — elapsed time helper
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Returns a module-level logger.
    Log level is controlled by the LOG_LEVEL environment variable
    (default: INFO). Configured once on first call.
    """
    import os
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_http_client(
    timeout: int = 30,
    headers: Optional[dict] = None,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Async context manager that yields a configured httpx.AsyncClient.

    Usage:
        async with get_http_client(timeout=20) as client:
            resp = await client.get(url)

    Automatically closes the client on exit.
    """
    default_headers = {
        "Accept": "application/json",
        "User-Agent": "ResearchDiscoveryBot/1.0",
    }
    if headers:
        default_headers.update(headers)

    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(
        headers=default_headers,
        timeout=httpx.Timeout(timeout),
        transport=transport,
        follow_redirects=True,
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Retry decorator for external API calls
# ---------------------------------------------------------------------------

def api_retry(max_attempts: int = 3, wait_min: float = 1.0, wait_max: float = 10.0):
    """
    Tenacity-based retry decorator for HTTP API calls.

    Retries on:
    - httpx.HTTPStatusError (5xx server errors)
    - httpx.ConnectError
    - httpx.TimeoutException

    Usage:
        @api_retry(max_attempts=3)
        async def my_api_call():
            ...
    """
    return retry(
        retry=retry_if_exception_type((
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.TimeoutException,
        )),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=wait_min, max=wait_max),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Token Bucket — async rate limiter
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Async token bucket rate limiter.

    Used by Semantic Scholar adapter to stay within 2 req/sec.

    Args:
        capacity:    max burst tokens
        refill_rate: tokens added per second
    """

    def __init__(self, capacity: float = 10.0, refill_rate: float = 2.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Wait until `tokens` are available, then consume them.
        Blocks the caller if rate limit is exceeded.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self.capacity,
                    self._tokens + elapsed * self.refill_rate,
                )
                self._last_refill = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

            # Not enough tokens — wait a bit then retry
            wait_time = (tokens - self._tokens) / self.refill_rate
            await asyncio.sleep(max(0.05, wait_time))


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """
    Simple wall-clock timer.

    Usage:
        t = Timer()
        ...do work...
        print(t.elapsed())  # → e.g. 3.42
    """

    def __init__(self):
        self._start = time.monotonic()

    def elapsed(self) -> float:
        """Returns elapsed seconds, rounded to 2 decimal places."""
        return round(time.monotonic() - self._start, 2)

    def reset(self) -> None:
        self._start = time.monotonic()