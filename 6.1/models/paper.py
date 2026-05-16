"""
Standardized Paper Schema — the canonical data contract for the entire system.
Every API adapter normalizes its output to this schema.
Every downstream module consumes this schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PaperSource(str, Enum):
    OPENALEX = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    CROSSREF = "crossref"
    CITATION_EXPANSION = "citation_expansion"


class PaperTier(str, Enum):
    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT_BACKGROUND = "relevant_background"
    ADJACENT_WORK = "adjacent_work"
    HISTORICAL_FOUNDATIONS = "historical_foundations"


class CitationRelation(str, Enum):
    CITES = "cites"
    REFERENCED_BY = "referenced_by"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ExternalIDs(BaseModel):
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    semantic_scholar: Optional[str] = None
    openalex: Optional[str] = None
    pubmed: Optional[str] = None

    class Config:
        extra = "allow"


class Author(BaseModel):
    name: str
    author_id: Optional[str] = None
    affiliation: Optional[str] = None
    orcid: Optional[str] = None


class RankingFeatures(BaseModel):
    semantic_similarity: float = 0.0
    citation_boost: float = 0.0
    recency_boost: float = 0.0
    venue_score: float = 0.0
    keyword_overlap: float = 0.0
    graph_centrality: float = 0.0  # from citation graph PageRank
    mmr_score: float = 0.0         # after diversity re-ranking


class PaperReference(BaseModel):
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None


class CitationEdge(BaseModel):
    source_paper_id: str
    target_paper_id: str
    relation: CitationRelation


# ---------------------------------------------------------------------------
# Core Paper Model
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    paper_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: PaperSource

    external_ids: ExternalIDs = Field(default_factory=ExternalIDs)
    title: str
    abstract: Optional[str] = None
    authors: list[Author] = Field(default_factory=list)

    year: Optional[int] = None
    venue: Optional[str] = None
    publication_date: Optional[str] = None

    citation_count: int = 0
    reference_count: int = 0

    fields_of_study: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    pdf_url: Optional[str] = None
    landing_page_url: Optional[str] = None
    is_open_access: bool = False
    language: str = "en"

    # Populated after embedding generation
    embedding: list[float] = Field(default_factory=list)

    # Populated after ranking
    similarity_score: float = 0.0
    final_score: float = 0.0
    ranking_features: RankingFeatures = Field(default_factory=RankingFeatures)

    # Citation graph
    references: list[PaperReference] = Field(default_factory=list)
    citations: list[PaperReference] = Field(default_factory=list)

    # Provenance
    retrieved_from_queries: list[str] = Field(default_factory=list)
    tier: Optional[PaperTier] = None
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    # Dedup fingerprint (populated by dedup engine)
    title_fingerprint: Optional[str] = None

    class Config:
        use_enum_values = True

    @model_validator(mode="after")
    def ensure_title(self) -> "Paper":
        if not self.title or not self.title.strip():
            raise ValueError("Paper must have a non-empty title")
        return self

    def get_best_doi(self) -> Optional[str]:
        return self.external_ids.doi

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# System-level models
# ---------------------------------------------------------------------------

class ResearchQuery(BaseModel):
    """The user's original research idea, expanded into retrieval queries."""
    original_idea: str
    expanded_queries: list[str] = Field(default_factory=list)
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class SearchResult(BaseModel):
    """Raw result from a single API call before normalization."""
    source: PaperSource
    query: str
    papers: list[Paper]
    total_found: int = 0
    error: Optional[str] = None


class DiscoveryResult(BaseModel):
    """Final output of the entire pipeline."""
    query: ResearchQuery
    highly_relevant: list[Paper] = Field(default_factory=list)
    relevant_background: list[Paper] = Field(default_factory=list)
    adjacent_work: list[Paper] = Field(default_factory=list)
    historical_foundations: list[Paper] = Field(default_factory=list)
    total_papers: int = 0
    processing_time_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def all_papers(self) -> list[Paper]:
        return (
            self.highly_relevant
            + self.relevant_background
            + self.adjacent_work
            + self.historical_foundations
        )