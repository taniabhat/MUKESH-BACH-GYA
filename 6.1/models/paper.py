"""
Canonical research paper schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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
    HISTORICAL_FOUNDATIONS = (
        "historical_foundations"
    )


class CitationRelation(str, Enum):
    CITES = "cites"
    REFERENCED_BY = "referenced_by"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ExternalIDs(BaseModel):

    model_config = ConfigDict(
        extra="ignore",
    )

    doi: Optional[str] = None
    arxiv: Optional[str] = None
    semantic_scholar: Optional[str] = None
    openalex: Optional[str] = None
    pubmed: Optional[str] = None

    @field_validator("doi")
    @classmethod
    def normalize_doi(
        cls,
        value: Optional[str],
    ) -> Optional[str]:

        if not value:
            return None

        return value.strip().lower()


class Author(BaseModel):

    model_config = ConfigDict(
        frozen=True,
    )

    name: str
    author_id: Optional[str] = None
    affiliation: Optional[str] = None
    orcid: Optional[str] = None

    @field_validator("name")
    @classmethod
    def clean_name(
        cls,
        value: str,
    ) -> str:

        return value.strip()


class RankingFeatures(BaseModel):

    model_config = ConfigDict(
        frozen=True,
    )

    semantic_similarity: float = 0.0
    citation_boost: float = 0.0
    recency_boost: float = 0.0
    venue_score: float = 0.0
    keyword_overlap: float = 0.0
    graph_centrality: float = 0.0
    mmr_score: float = 0.0


class PaperReference(BaseModel):

    model_config = ConfigDict(
        frozen=True,
    )

    doi: Optional[str] = None
    arxiv: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None


class CitationEdge(BaseModel):

    model_config = ConfigDict(
        frozen=True,
    )

    source_paper_id: str
    target_paper_id: str
    relation: CitationRelation


# ---------------------------------------------------------------------------
# Core Paper Model
# ---------------------------------------------------------------------------

class Paper(BaseModel):

    model_config = ConfigDict(
        use_enum_values=True,
    )

    paper_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    source: PaperSource

    external_ids: ExternalIDs = Field(
        default_factory=ExternalIDs
    )

    title: str
    abstract: Optional[str] = None

    authors: list[Author] = Field(
        default_factory=list
    )

    year: Optional[int] = None
    venue: Optional[str] = None
    publication_date: Optional[str] = None

    citation_count: int = 0
    reference_count: int = 0

    fields_of_study: list[str] = Field(
        default_factory=list
    )

    keywords: list[str] = Field(
        default_factory=list
    )

    pdf_url: Optional[str] = None
    landing_page_url: Optional[str] = None

    is_open_access: bool = False
    language: str = "en"

    # Retrieval / ranking state
    embedding: list[float] = Field(
        default_factory=list
    )

    similarity_score: float = 0.0
    final_score: float = 0.0

    ranking_features: RankingFeatures = (
        Field(
            default_factory=RankingFeatures
        )
    )

    # Citation graph
    references: list[PaperReference] = (
        Field(default_factory=list)
    )

    citations: list[PaperReference] = (
        Field(default_factory=list)
    )

    # Provenance
    retrieved_from_queries: list[str] = (
        Field(default_factory=list)
    )

    tier: Optional[PaperTier] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    title_fingerprint: Optional[str] = None

    @field_validator(
        "title",
        "venue",
        mode="before",
    )
    @classmethod
    def clean_text_fields(
        cls,
        value: Optional[str],
    ) -> Optional[str]:

        if not isinstance(value, str):
            return value

        cleaned = value.strip()

        return cleaned or None

    @field_validator("language")
    @classmethod
    def normalize_language(
        cls,
        value: str,
    ) -> str:

        return value.lower().strip()

    @model_validator(mode="after")
    def validate_title(self) -> "Paper":

        if not self.title:
            raise ValueError(
                "Paper title cannot be empty"
            )

        return self

    def get_best_doi(self) -> Optional[str]:

        return self.external_ids.doi


# ---------------------------------------------------------------------------
# System-level Models
# ---------------------------------------------------------------------------

class ResearchQuery(BaseModel):

    """Original user research intent."""

    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    original_idea: str

    expanded_queries: list[str] = (
        Field(default_factory=list)
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


class SearchResult(BaseModel):

    """Single-provider retrieval result."""

    source: PaperSource
    query: str

    papers: list[Paper]

    total_found: int = 0

    error: Optional[str] = None


class DiscoveryResult(BaseModel):

    """Final aggregated retrieval pipeline output."""

    query: ResearchQuery

    highly_relevant: list[Paper] = (
        Field(default_factory=list)
    )

    relevant_background: list[Paper] = (
        Field(default_factory=list)
    )

    adjacent_work: list[Paper] = (
        Field(default_factory=list)
    )

    historical_foundations: list[Paper] = (
        Field(default_factory=list)
    )

    total_papers: int = 0

    processing_time_seconds: float = 0.0

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def all_papers(self) -> list[Paper]:

        return (
            self.highly_relevant
            + self.relevant_background
            + self.adjacent_work
            + self.historical_foundations
        )