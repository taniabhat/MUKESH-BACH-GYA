"""
Ranking Engine — Multi-signal paper relevance ranking.

Combines five signals into a final composite score:
1. Semantic similarity   (60%) — BGE-M3 cosine similarity to query
2. Citation boost        (15%) — log-normalized citation count
3. Recency boost         (10%) — year-based decay toward current year
4. Venue score           (10%) — known high-impact venue bonus
5. Keyword overlap       (5%)  — simple token overlap with query

After initial ranking, applies MMR diversity reranking to prevent
returning 50 nearly-identical papers.

Papers are then assigned to one of four tiers based on final score.
"""

from __future__ import annotations

import datetime
import math
import re
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_logger
from research_discovery.models.paper import Paper, PaperTier
from research_discovery.services.embedding.service import EmbeddingService

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Current year — used for recency scoring
# ---------------------------------------------------------------------------
_CURRENT_YEAR = datetime.datetime.utcnow().year

# ---------------------------------------------------------------------------
# High-impact venue list — case-insensitive substring matching
# ---------------------------------------------------------------------------
_HIGH_IMPACT_VENUES = frozenset({
    # ML / AI
    "neurips", "nips", "icml", "iclr", "aaai", "ijcai",
    # NLP
    "acl", "emnlp", "naacl", "coling", "eacl",
    # CV
    "cvpr", "iccv", "eccv",
    # Systems / Engineering
    "sosp", "osdi", "usenix", "sigcomm", "eurosys",
    # Data
    "sigmod", "vldb", "icde", "kdd", "www", "wsdm",
    # Science
    "nature", "science", "cell", "pnas", "plos",
    # Robotics
    "icra", "iros", "rss",
    # IEEE
    "ieee transactions", "tpami", "tnnls",
})

_MEDIUM_IMPACT_VENUES = frozenset({
    "workshop", "symposium", "conference", "acm", "springer", "arxiv",
    "preprint", "journal", "review",
})


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def _citation_score(citation_count: int) -> float:
    """
    Log-normalized citation score.

    0 citations → 0.0
    100 citations → ~0.67
    1000 citations → ~1.0
    Very high counts are capped at 1.0.

    Uses log base 1000 so that 1000+ citations = max score.
    """
    if citation_count <= 0:
        return 0.0
    score = math.log(citation_count + 1) / math.log(1001)
    return min(score, 1.0)


def _recency_score(year: Optional[int]) -> float:
    """
    Recency score based on publication year.

    Current year → 1.0
    5 years ago → 0.5
    10+ years ago → 0.0

    Unknown year → 0.3 (neutral, slightly penalized)
    """
    if year is None:
        return 0.3

    age = _CURRENT_YEAR - year
    if age <= 0:
        return 1.0
    if age >= 10:
        return 0.0

    # Linear decay from 1.0 (age=0) to 0.0 (age=10)
    return max(0.0, 1.0 - age / 10.0)


def _venue_score(venue: Optional[str]) -> float:
    """
    Venue impact score.

    Top-tier venues (NeurIPS, ICML, ACL, etc.) → 1.0
    Medium venues (workshops, symposia) → 0.5
    Unknown venues → 0.2
    No venue → 0.1
    """
    if not venue:
        return 0.1

    venue_lower = venue.lower()

    for v in _HIGH_IMPACT_VENUES:
        if v in venue_lower:
            return 1.0

    for v in _MEDIUM_IMPACT_VENUES:
        if v in venue_lower:
            return 0.5

    return 0.2


def _keyword_overlap(paper: Paper, query: str) -> float:
    """
    Token overlap between query and paper (title + abstract).

    Counts how many distinct query tokens appear in the paper content.
    Normalized by number of query tokens.
    Returns 0.0–1.0.
    """
    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))

    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    content = paper.title
    if paper.abstract:
        content += " " + paper.abstract

    paper_tokens = tokenize(content)
    overlap = query_tokens & paper_tokens
    return len(overlap) / len(query_tokens)


# ---------------------------------------------------------------------------
# Tier thresholds
# ---------------------------------------------------------------------------

_TIER_THRESHOLDS = {
    PaperTier.HIGHLY_RELEVANT: 0.75,
    PaperTier.RELEVANT_BACKGROUND: 0.55,
    PaperTier.ADJACENT_WORK: 0.35,
    PaperTier.HISTORICAL_FOUNDATIONS: 0.0,
}


def _assign_tier(score: float) -> str:
    """Assign a PaperTier string based on the final composite score."""
    if score >= _TIER_THRESHOLDS[PaperTier.HIGHLY_RELEVANT]:
        return PaperTier.HIGHLY_RELEVANT
    if score >= _TIER_THRESHOLDS[PaperTier.RELEVANT_BACKGROUND]:
        return PaperTier.RELEVANT_BACKGROUND
    if score >= _TIER_THRESHOLDS[PaperTier.ADJACENT_WORK]:
        return PaperTier.ADJACENT_WORK
    return PaperTier.HISTORICAL_FOUNDATIONS


# ---------------------------------------------------------------------------
# Ranking Engine
# ---------------------------------------------------------------------------

class RankingEngine:
    """
    Multi-signal ranking engine for academic papers.

    Pipeline:
    1. Compute composite score per paper (5 signals, weighted)
    2. Sort descending by composite score
    3. Apply MMR diversity reranking on top-K
    4. Assign tier labels
    5. Apply citation graph centrality bonus (if available)
    6. Truncate to max_papers
    """

    def __init__(
        self,
        mmr_lambda: float = 0.7,
        max_papers: int = 150,
        weights: Optional[dict] = None,
    ):
        self.mmr_lambda = mmr_lambda
        self.max_papers = max_papers

        # Load weights from config or use provided override
        cfg = settings.ranking
        self.weights = weights or {
            "semantic_similarity": cfg.semantic_similarity,
            "citation_boost":      cfg.citation_boost,
            "recency_boost":       cfg.recency_boost,
            "venue_score":         cfg.venue_score,
            "keyword_overlap":     cfg.keyword_overlap,
        }

        self._embedder = EmbeddingService()

    def _composite_score(self, paper: Paper, query: str) -> float:
        """
        Compute the weighted composite relevance score for a single paper.

        Each component is independently bounded [0, 1].
        Weighted sum gives a final score in [0, 1].
        """
        sem_sim  = paper.ranking_features.semantic_similarity
        cite     = _citation_score(paper.citation_count)
        recency  = _recency_score(paper.year)
        venue    = _venue_score(paper.venue)
        kw       = _keyword_overlap(paper, query)
        graph    = paper.ranking_features.graph_centrality  # 0 if not set

        # Store individual components for transparency
        paper.ranking_features.citation_boost = round(cite, 4)
        paper.ranking_features.recency_boost = round(recency, 4)
        paper.ranking_features.venue_score = round(venue, 4)
        paper.ranking_features.keyword_overlap = round(kw, 4)

        # Weighted composite
        score = (
            self.weights["semantic_similarity"] * sem_sim
            + self.weights["citation_boost"]      * cite
            + self.weights["recency_boost"]        * recency
            + self.weights["venue_score"]          * venue
            + self.weights["keyword_overlap"]      * kw
        )

        # Small graph centrality bonus (additive, capped)
        if graph > 0:
            score = min(1.0, score + 0.05 * graph)

        return round(score, 6)

    def rank(
        self,
        papers: list[Paper],
        query: str,
        query_embedding: list[float],
    ) -> list[Paper]:
        """
        Full ranking pipeline.

        Args:
            papers:          List of papers with embeddings + similarity scores set
            query:           Original research idea string (for keyword overlap)
            query_embedding: Query embedding vector (for MMR)

        Returns:
            Sorted, tiered, MMR-reranked list of papers (max_papers).
        """
        if not papers:
            return []

        logger.info(f"Ranking {len(papers)} papers...")

        # Step 1 — Compute composite scores
        for paper in papers:
            paper.final_score = self._composite_score(paper, query)

        # Step 2 — Sort by composite score descending
        papers.sort(key=lambda p: p.final_score, reverse=True)

        # Step 3 — MMR diversity reranking on top candidates
        # Only rerank the top pool (2x max_papers) to avoid O(n²) cost
        mmr_pool_size = min(len(papers), self.max_papers * 2)
        pool = papers[:mmr_pool_size]
        remainder = papers[mmr_pool_size:]

        reranked = self._embedder.mmr_rerank(
            pool,
            query_embedding,
            top_k=min(self.max_papers, len(pool)),
            lambda_=self.mmr_lambda,
        )

        # Append remainder (already sorted, below threshold)
        # but only up to max_papers total
        final = reranked
        if len(final) < self.max_papers:
            slots = self.max_papers - len(final)
            final = final + remainder[:slots]

        # Step 4 — Assign tiers
        for paper in final:
            paper.tier = _assign_tier(paper.final_score)

        tier_counts = {}
        for p in final:
            tier_counts[p.tier] = tier_counts.get(p.tier, 0) + 1
        logger.info(
            f"Ranking complete: {len(final)} papers | "
            f"tiers: {tier_counts}"
        )

        return final[:self.max_papers]