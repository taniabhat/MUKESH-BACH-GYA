"""
Infrastructure runtime for Research Discovery Platform.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from research_discovery.config.settings import (
    settings,
)


# ---------------------------------------------------------------------------
# Logging Runtime
# ---------------------------------------------------------------------------

class LoggingRuntime:
    """
    Centralized logging configuration.
    """

    _initialized = False

    @classmethod
    def initialize(
        cls,
    ) -> None:

        if cls._initialized:
            return

        level = getattr(
            logging,
            settings.log_level.upper(),
            logging.INFO,
        )

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | "
                "%(levelname)-8s | "
                "%(name)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler = logging.StreamHandler(
            sys.stdout
        )

        handler.setFormatter(
            formatter
        )

        root = logging.getLogger()

        root.setLevel(level)

        root.handlers.clear()

        root.addHandler(handler)

        cls._initialized = True


def get_logger(
    name: str,
) -> logging.Logger:

    LoggingRuntime.initialize()

    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Runtime Context
# ---------------------------------------------------------------------------

@dataclass
class RuntimeContext:

    started_at: float

    request_id: Optional[str] = None

    metadata: dict = None


# ---------------------------------------------------------------------------
# Retry Policy
# ---------------------------------------------------------------------------

def _retryable_exception(
    exc: Exception,
) -> bool:

    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
        ),
    ):
        return True

    if isinstance(
        exc,
        httpx.HTTPStatusError,
    ):

        if exc.response is None:
            return True

        return (
            exc.response.status_code
            >= 500
        )

    return False


def api_retry(
    max_attempts: int = (
        settings.http.max_retries
    ),
):

    return retry(
        retry=retry_if_exception(
            _retryable_exception
        ),
        stop=stop_after_attempt(
            max_attempts
        ),
        wait=wait_exponential(
            min=1,
            max=10,
        ),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# HTTP Runtime
# ---------------------------------------------------------------------------

class HTTPClientProxy:
    """Proxy for making per-request overrides on a shared httpx client."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        headers: Optional[dict] = None,
        timeout: Optional[httpx.Timeout] = None,
    ):
        self._client = client
        self._headers = headers or {}
        self._timeout = timeout

    async def get(self, url: str, **kwargs) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers.update(self._headers)
        
        timeout = kwargs.pop("timeout", self._timeout)
        if timeout is None:
            timeout = self._client.timeout
            
        return await self._client.get(
            url,
            headers=headers,
            timeout=timeout,
            **kwargs,
        )

    async def post(self, url: str, **kwargs) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers.update(self._headers)
        
        timeout = kwargs.pop("timeout", self._timeout)
        if timeout is None:
            timeout = self._client.timeout
            
        return await self._client.post(
            url,
            headers=headers,
            timeout=timeout,
            **kwargs,
        )


class HTTPRuntime:
    """
    Shared async HTTP runtime.
    """

    _client: Optional[
        httpx.AsyncClient
    ] = None

    @classmethod
    async def startup(
        cls,
    ) -> None:

        if cls._client:
            return

        transport = (
            httpx.AsyncHTTPTransport(
                retries=2
            )
        )

        cls._client = (
            httpx.AsyncClient(
                timeout=httpx.Timeout(
                    settings.http.timeout
                ),
                follow_redirects=True,
                headers={
                    "Accept": (
                        "application/json"
                    ),
                    "User-Agent": (
                        settings.http.user_agent
                    ),
                },
                limits=httpx.Limits(
                    max_connections=(
                        settings.http.max_connections
                    ),
                    max_keepalive_connections=20,
                ),
                transport=transport,
            )
        )

    @classmethod
    async def shutdown(
        cls,
    ) -> None:

        if not cls._client:
            return

        await cls._client.aclose()

        cls._client = None

    @classmethod
    @asynccontextmanager
    async def client(
        cls,
        headers: Optional[
            dict
        ] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[
        HTTPClientProxy,
        None,
    ]:

        if cls._client is None:

            await cls.startup()

        temp_timeout = (
            httpx.Timeout(timeout)
            if timeout is not None
            else None
        )

        yield HTTPClientProxy(
            client=cls._client,
            headers=headers,
            timeout=temp_timeout,
        )


@asynccontextmanager
async def get_http_client(
    headers: Optional[
        dict
    ] = None,
    timeout: Optional[float] = None,
):

    async with HTTPRuntime.client(
        headers=headers,
        timeout=timeout,
    ) as client:

        yield client


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Async token bucket limiter.
    """

    def __init__(
        self,
        capacity: float = 10.0,
        refill_rate: float = 2.0,
    ):

        self.capacity = capacity

        self.refill_rate = refill_rate

        self._tokens = float(
            capacity
        )

        self._last_refill = (
            time.monotonic()
        )

        self._lock = asyncio.Lock()

    async def acquire(
        self,
        tokens: float = 1.0,
    ) -> None:

        while True:

            async with self._lock:

                self._refill()

                if (
                    self._tokens
                    >= tokens
                ):

                    self._tokens -= tokens

                    return

                missing = (
                    tokens
                    - self._tokens
                )

            wait_time = (
                missing
                / self.refill_rate
            )

            await asyncio.sleep(
                max(0.05, wait_time)
            )

    def _refill(
        self,
    ) -> None:

        now = time.monotonic()

        elapsed = (
            now
            - self._last_refill
        )

        self._tokens = min(
            self.capacity,
            self._tokens
            + (
                elapsed
                * self.refill_rate
            ),
        )

        self._last_refill = now


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

class Timer:
    """
    Runtime instrumentation timer.
    """

    def __init__(
        self,
    ):

        self._started_at = (
            time.monotonic()
        )

    def elapsed(
        self,
    ) -> float:

        return round(
            time.monotonic()
            - self._started_at,
            4,
        )

    def reset(
        self,
    ) -> None:

        self._started_at = (
            time.monotonic()
        )