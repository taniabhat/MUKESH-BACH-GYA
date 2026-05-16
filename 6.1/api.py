"""
FastAPI Application — Research Discovery Module REST API.

Endpoints:
  POST /api/v1/discover          — full pipeline run
  POST /api/v1/expand-queries    — query expansion only
  POST /api/v1/search            — retrieval only (no ranking)
  GET  /api/v1/paper/{doi}       — single paper lookup
  GET  /health                   — liveness check
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from research_discovery.core.pipeline import ResearchDiscoveryPipeline
from research_discovery.core.utils import get_logger
from research_discovery.models.paper import DiscoveryResult, Paper, ResearchQuery
from research_discovery.services.query_expansion.agent import QueryExpansionAgent
from research_discovery.services.retrieval.openalex import OpenAlexAdapter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Discovery Module",
    description=(
        "Academic paper discovery via retrieval + ranking + citation graph expansion. "
        "Combines OpenAlex, Semantic Scholar, and arXiv with LLM query expansion."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DiscoverRequest(BaseModel):
    research_idea: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language description of the research topic",
        example="Using LLMs for automated code review",
    )
    num_expansion_queries: int = Field(default=10, ge=1, le=20)
    results_per_query: int = Field(default=20, ge=5, le=50)
    use_semantic_scholar: bool = True
    use_arxiv: bool = True
    use_crossref_enrichment: bool = True
    use_citation_expansion: bool = True
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    max_final_papers: int = Field(default=150, ge=10, le=300)


class ExpandQueriesRequest(BaseModel):
    research_idea: str = Field(..., min_length=3, max_length=2000)
    num_queries: int = Field(default=10, ge=1, le=20)


class ExpandQueriesResponse(BaseModel):
    original_idea: str
    expanded_queries: list[str]
    count: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    sources: list[str] = Field(default=["openalex"], description="openalex, arxiv, semantic_scholar")
    per_source: int = Field(default=20, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    papers: list[Paper]
    total: int
    sources_queried: list[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness check."""
    return HealthResponse(status="ok", timestamp=time.time())


@app.post(
    "/api/v1/discover",
    response_model=DiscoveryResult,
    tags=["Discovery"],
    summary="Run full research discovery pipeline",
    responses={500: {"model": ErrorResponse}},
)
async def discover(request: DiscoverRequest):
    """
    Full end-to-end pipeline:
    1. LLM query expansion
    2. Parallel retrieval (OpenAlex + S2 + arXiv)
    3. CrossRef metadata enrichment
    4. Deduplication
    5. Embedding generation (BGE-M3)
    6. Hybrid relevance ranking
    7. Citation graph expansion
    8. MMR diversity reranking
    9. Tier classification

    Returns papers organized into:
    - **highly_relevant** (score ≥ 0.75)
    - **relevant_background** (score ≥ 0.55)
    - **adjacent_work** (score ≥ 0.35)
    - **historical_foundations** (score < 0.35)
    """
    try:
        pipeline = ResearchDiscoveryPipeline(
            num_expansion_queries=request.num_expansion_queries,
            results_per_query=request.results_per_query,
            use_semantic_scholar=request.use_semantic_scholar,
            use_arxiv=request.use_arxiv,
            use_crossref_enrichment=request.use_crossref_enrichment,
            use_citation_expansion=request.use_citation_expansion,
            mmr_lambda=request.mmr_lambda,
            max_final_papers=request.max_final_papers,
        )
        result = await pipeline.run(request.research_idea)
        return result
    except Exception as exc:
        logger.error(f"Discovery pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/v1/expand-queries",
    response_model=ExpandQueriesResponse,
    tags=["Query Expansion"],
    summary="Expand a research idea into multiple retrieval queries",
)
async def expand_queries(request: ExpandQueriesRequest):
    """
    Uses LLM (or rule-based fallback) to generate diverse search queries
    from a single research idea.
    """
    try:
        agent = QueryExpansionAgent(num_queries=request.num_queries)
        queries = await agent.expand(request.research_idea)
        return ExpandQueriesResponse(
            original_idea=request.research_idea,
            expanded_queries=queries,
            count=len(queries),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Retrieval"],
    summary="Search academic APIs directly (no ranking)",
)
async def search(request: SearchRequest):
    """
    Direct search without the full pipeline.
    Useful for quick lookup or testing individual API adapters.
    """
    papers: list[Paper] = []
    sources_used: list[str] = []

    tasks = []
    source_labels = []

    if "openalex" in request.sources:
        adapter = OpenAlexAdapter()
        tasks.append(adapter.search(request.query, per_page=request.per_source))
        source_labels.append("openalex")

    if "arxiv" in request.sources:
        from research_discovery.services.retrieval.arxiv import ArxivAdapter
        tasks.append(ArxivAdapter().search(request.query, max_results=request.per_source))
        source_labels.append("arxiv")

    if "semantic_scholar" in request.sources:
        from research_discovery.services.retrieval.semantic_scholar import SemanticScholarAdapter
        tasks.append(SemanticScholarAdapter().search(request.query, limit=request.per_source))
        source_labels.append("semantic_scholar")

    if not tasks:
        raise HTTPException(status_code=400, detail="No valid sources specified")

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for label, result in zip(source_labels, results):
            if isinstance(result, Exception):
                logger.warning(f"Search error from {label}: {result}")
            else:
                papers.extend(result.papers)
                sources_used.append(label)

        return SearchResponse(
            query=request.query,
            papers=papers,
            total=len(papers),
            sources_queried=sources_used,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/api/v1/paper/doi/{doi:path}",
    response_model=Paper,
    tags=["Retrieval"],
    summary="Fetch a single paper by DOI",
)
async def get_paper_by_doi(doi: str):
    """Look up a specific paper by DOI from OpenAlex."""
    try:
        adapter = OpenAlexAdapter()
        paper = await adapter.fetch_by_doi(doi)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found for DOI: {doi}")
        return paper
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Research Discovery Module API starting up...")
    logger.info(f"API docs available at: http://localhost:8000/docs")