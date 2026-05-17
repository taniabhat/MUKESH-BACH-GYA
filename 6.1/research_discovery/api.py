"""
FastAPI gateway for Research Discovery Platform.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    Field,
)

from research_discovery.config.settings import (
    settings,
)
from research_discovery.core.pipeline import (
    ResearchDiscoveryPipeline,
)
from research_discovery.core.runtime import (
    HTTPRuntime,
    get_logger,
)
from research_discovery.models.paper import (
    DiscoveryResult,
    Paper,
)

logger = get_logger(__name__)

API_VERSION = "v1"


# ---------------------------------------------------------------------------
# Request Context
# ---------------------------------------------------------------------------

@dataclass
class RequestContext:

    request_id: str

    started_at: float


# ---------------------------------------------------------------------------
# API Services
# ---------------------------------------------------------------------------

class DiscoveryAPIService:
    """
    API-facing orchestration layer.
    """

    def __init__(
        self,
    ):

        self.pipeline = (
            ResearchDiscoveryPipeline()
        )

    async def discover(
        self,
        research_idea: str,
    ) -> DiscoveryResult:

        return await self.pipeline.run(
            research_idea
        )

    async def search(
        self,
        query: str,
        sources: list[str],
        per_source: int,
    ) -> list[Paper]:

        providers = [
            provider
            for provider in (
                self.pipeline
                .services
                .retrievers
            )
            if (
                provider.__class__.__name__
                .lower()
                .replace(
                    "adapter",
                    "",
                )
                in sources
            )
        ]

        tasks = [
            provider.search(
                query,
                limit=per_source,
            )
            for provider in providers
        ]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        papers = []

        for result in results:

            if isinstance(
                result,
                Exception,
            ):

                logger.warning(
                    "Search provider failure"
                )

                continue

            papers.extend(
                result.papers
            )

        return papers


# ---------------------------------------------------------------------------
# App Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logger.info(
        "Research Discovery API starting"
    )

    app.state.discovery_service = (
        DiscoveryAPIService()
    )

    yield

    logger.info(
        "Research Discovery API shutting down"
    )
    await HTTPRuntime.shutdown()


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Discovery Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
):

    request_id = str(
        uuid.uuid4()
    )

    request.state.context = (
        RequestContext(
            request_id=request_id,
            started_at=time.time(),
        )
    )

    response = await call_next(
        request
    )

    response.headers[
        "X-Request-ID"
    ] = request_id

    return response


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unhandled API exception"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": (
                "internal_server_error"
            ),
            "detail": str(exc),
        },
    )


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------

def get_discovery_service(
    request: Request,
) -> DiscoveryAPIService:

    return (
        request.app.state.discovery_service
    )


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class DiscoverRequest(
    BaseModel
):

    research_idea: str = Field(
        ...,
        min_length=3,
        max_length=2000,
    )


class APIResponse(
    BaseModel
):

    success: bool = True

    request_id: str

    timestamp: float


class DiscoverResponse(
    APIResponse
):

    result: DiscoveryResult


class SearchRequest(
    BaseModel
):

    query: str

    sources: list[str] = Field(
        default_factory=lambda: [
            "openalex",
            "semantic_scholar",
            "arxiv",
        ]
    )

    per_source: int = 20


class SearchResponse(
    APIResponse
):

    query: str

    papers: list[Paper]

    total: int


class HealthResponse(
    APIResponse
):

    status: str

    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health(
    request: Request,
):

    context = (
        request.state.context
    )

    return HealthResponse(
        request_id=(
            context.request_id
        ),
        timestamp=time.time(),
        status="ok",
        version="1.0.0",
    )


@app.post(
    f"/api/{API_VERSION}/discover",
    response_model=DiscoverResponse,
)
async def discover(
    payload: DiscoverRequest,
    request: Request,
    service: DiscoveryAPIService = Depends(
        get_discovery_service
    ),
):

    context = (
        request.state.context
    )

    result = await service.discover(
        payload.research_idea
    )

    return DiscoverResponse(
        request_id=(
            context.request_id
        ),
        timestamp=time.time(),
        result=result,
    )


@app.post(
    f"/api/{API_VERSION}/search",
    response_model=SearchResponse,
)
async def search(
    payload: SearchRequest,
    request: Request,
    service: DiscoveryAPIService = Depends(
        get_discovery_service
    ),
):

    context = (
        request.state.context
    )

    papers = await service.search(
        query=payload.query,
        sources=payload.sources,
        per_source=payload.per_source,
    )

    return SearchResponse(
        request_id=(
            context.request_id
        ),
        timestamp=time.time(),
        query=payload.query,
        papers=papers,
        total=len(papers),
    )