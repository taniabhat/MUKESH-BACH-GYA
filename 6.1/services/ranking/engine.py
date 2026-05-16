"""
Multi-signal research paper ranking engine.
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_logger
from research_discovery.models.paper import (
    Paper,
    PaperTier,
    RankingFeatures,
)
from research_discovery.services.embedding.service import (
    EmbeddingService,
)

logger = get_logger(__name__)

CURRENT_YEAR = datetime.datetime.utcnow().year

DEFAULT_MMR_LAMBDA = 0.7
DEFAULT_MAX_PAPERS = 150

GRAPH_BONUS_WEIGHT = 0.05


# ---------------------------------------------------------------------------
# Venue Lists
# ---------------------------------------------------------------------------

HIGH_IMPACT_VENUES = frozenset({
    "neurips", "nips", "icml", "iclr",
    "acl", "emnlp", "naacl",
    "cvpr", "iccv", "eccv",
    "nature", "science",
    "kdd", "sigmod", "vldb",
})

MEDIUM_IMPACT_VENUES = frozenset({
    "conference",
    "workshop",
    "symposium",
    "journal",
    "arxiv",
})


# ---------------------------------------------------------------------------
# Ranking Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingWeights:

    semantic_similarity: float
    citation_boost: float
    recency_boost: float
    venue_score: float
    keyword_overlap: float


DEFAULT_WEIGHTS = RankingWeights(
    semantic_similarity=(
        settings.ranking.semantic_similarity
    ),
    citation_boost=(
        settings.ranking.citation_boost
    ),
    recency_boost=(
        settings.ranking.recency_boost
    ),
    venue_score=(
        settings.ranking.venue_score
    ),
    keyword_overlap=(
        settings.ranking.keyword_overlap
    ),
)


# ---------------------------------------------------------------------------
# Feature Scoring
# ---------------------------------------------------------------------------

class FeatureScorer:
    """Computes bounded ranking features."""

    @staticmethod
    def citation_score(
        citation_count: int,
    ) -> float:

        if citation_count <= 0:
            return 0.0

        score = (
            math.log(citation_count + 1)
            / math.log(1001)
        )

        return min(score, 1.0)

    @staticmethod
    def recency_score(
        year: Optional[int],
    ) -> float:

        if year is None:
            return 0.3

        age = CURRENT_YEAR - year

        if age <= 0:
            return 1.0

        if age >= 10:
            return 0.0

        return max(
            0.0,
            1.0 - age / 10.0,
        )

    @staticmethod
    def venue_score(
        venue: Optional[str],
    ) -> float:

        if not venue:
            return 0.1

        lowered = venue.lower()

        for candidate in (
            HIGH_IMPACT_VENUES
        ):

            if candidate in lowered:
                return 1.0

        for candidate in (
            MEDIUM_IMPACT_VENUES
        ):

            if candidate in lowered:
                return 0.5

        return 0.2

    @staticmethod
    def keyword_overlap(
        paper: Paper,
        query_tokens: set[str],
    ) -> float:

        if not query_tokens:
            return 0.0

        content = paper.title

        if paper.abstract:
            content += (
                " "
                + paper.abstract
            )

        paper_tokens = (
            FeatureScorer.tokenize(
                content
            )
        )

        overlap = (
            query_tokens
            & paper_tokens
        )

        return (
            len(overlap)
            / len(query_tokens)
        )

    @staticmethod
    def tokenize(
        text: str,
    ) -> set[str]:

        return set(
            re.findall(
                r"\b[a-z]{3,}\b",
                text.lower(),
            )
        )


# ---------------------------------------------------------------------------
# Tier Assignment
# ---------------------------------------------------------------------------

class TierAssigner:
    """Assigns semantic paper tiers."""

    THRESHOLDS = {
        PaperTier.HIGHLY_RELEVANT: 0.75,
        PaperTier.RELEVANT_BACKGROUND: 0.55,
        PaperTier.ADJACENT_WORK: 0.35,
        PaperTier.HISTORICAL_FOUNDATIONS: 0.0,
    }

    @classmethod
    def assign(
        cls,
        score: float,
    ) -> PaperTier:

        for tier, threshold in (
            cls.THRESHOLDS.items()
        ):

            if score >= threshold:
                return tier

        return (
            PaperTier.HISTORICAL_FOUNDATIONS
        )


# ---------------------------------------------------------------------------
# Ranking Engine
# ---------------------------------------------------------------------------

class RankingEngine:
    """
    Multi-signal ranking pipeline.
    """

    def __init__(
        self,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        max_papers: int = DEFAULT_MAX_PAPERS,
        weights: RankingWeights = (
            DEFAULT_WEIGHTS
        ),
    ):

        self.mmr_lambda = mmr_lambda

        self.max_papers = max_papers

        self.weights = weights

        self._embedder = (
            EmbeddingService()
        )

    def rank(
        self,
        papers: list[Paper],
        query: str,
        query_embedding: list[float],
    ) -> list[Paper]:

        if not papers:
            return []

        logger.info(
            "Ranking papers count=%s",
            len(papers),
        )

        query_tokens = (
            FeatureScorer.tokenize(
                query
            )
        )

        for paper in papers:

            features = (
                self._compute_features(
                    paper=paper,
                    query_tokens=(
                        query_tokens
                    ),
                )
            )

            paper.ranking_features = (
                features
            )

            paper.final_score = (
                self._compute_final_score(
                    features
                )
            )

        papers.sort(
            key=lambda paper: (
                paper.final_score
            ),
            reverse=True,
        )

        papers = (
            self._apply_mmr_reranking(
                papers,
                query_embedding,
            )
        )

        for paper in papers:

            paper.tier = (
                TierAssigner.assign(
                    paper.final_score
                )
            )

        logger.info(
            "Ranking complete output=%s",
            len(papers),
        )

        return papers[: self.max_papers]

    def _compute_features(
        self,
        paper: Paper,
        query_tokens: set[str],
    ) -> RankingFeatures:

        return RankingFeatures(
            semantic_similarity=round(
                paper.ranking_features.semantic_similarity,
                4,
            ),
            citation_boost=round(
                FeatureScorer.citation_score(
                    paper.citation_count
                ),
                4,
            ),
            recency_boost=round(
                FeatureScorer.recency_score(
                    paper.year
                ),
                4,
            ),
            venue_score=round(
                FeatureScorer.venue_score(
                    paper.venue
                ),
                4,
            ),
            keyword_overlap=round(
                FeatureScorer.keyword_overlap(
                    paper,
                    query_tokens,
                ),
                4,
            ),
            graph_centrality=round(
                paper.ranking_features.graph_centrality,
                4,
            ),
            mmr_score=round(
                paper.ranking_features.mmr_score,
                4,
            ),
        )

    def _compute_final_score(
        self,
        features: RankingFeatures,
    ) -> float:

        score = (
            self.weights.semantic_similarity
            * features.semantic_similarity
            + self.weights.citation_boost
            * features.citation_boost
            + self.weights.recency_boost
            * features.recency_boost
            + self.weights.venue_score
            * features.venue_score
            + self.weights.keyword_overlap
            * features.keyword_overlap
        )

        if features.graph_centrality > 0:

            score += (
                GRAPH_BONUS_WEIGHT
                * features.graph_centrality
            )

        return round(
            min(score, 1.0),
            6,
        )

    def _apply_mmr_reranking(
        self,
        papers: list[Paper],
        query_embedding: list[float],
    ) -> list[Paper]:

        rerank_pool_size = min(
            len(papers),
            self.max_papers * 2,
        )

        rerank_pool = papers[
            :rerank_pool_size
        ]

        remaining = papers[
            rerank_pool_size:
        ]

        eligible = [
            paper
            for paper in rerank_pool
            if paper.embedding
        ]

        reranked = (
            self._embedder.mmr_rerank(
                papers=eligible,
                query_embedding=(
                    query_embedding
                ),
                top_k=min(
                    self.max_papers,
                    len(eligible),
                ),
                lambda_=self.mmr_lambda,
            )
        )

        final = reranked

        if len(final) < self.max_papers:

            slots = (
                self.max_papers
                - len(final)
            )

            final.extend(
                remaining[:slots]
            )

        return final