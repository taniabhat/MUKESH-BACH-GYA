"""
Citation graph expansion and graph centrality scoring.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Optional

from research_discovery.core.utils import get_logger
from research_discovery.models.paper import (
    CitationEdge,
    CitationRelation,
    Paper,
    PaperReference,
    PaperSource,
    RankingFeatures,
)
from research_discovery.services.retrieval.openalex import (
    OpenAlexAdapter,
)

logger = get_logger(__name__)

DEFAULT_TOP_K = 20
DEFAULT_MAX_EXPANSION = 30
DEFAULT_CONCURRENCY = 5

PAGERANK_DAMPING = 0.85
PAGERANK_ITERATIONS = 30


# ---------------------------------------------------------------------------
# Graph Algorithms
# ---------------------------------------------------------------------------

class PageRankCalculator:
    """Lightweight iterative PageRank."""

    @staticmethod
    def compute(
        edges: list[tuple[str, str]],
        paper_ids: set[str],
        damping: float = PAGERANK_DAMPING,
        iterations: int = PAGERANK_ITERATIONS,
    ) -> dict[str, float]:

        if not paper_ids:
            return {}

        node_count = len(paper_ids)

        ranks = {
            paper_id: 1.0 / node_count
            for paper_id in paper_ids
        }

        outgoing = defaultdict(list)
        incoming = defaultdict(list)

        for source, target in edges:
            outgoing[source].append(target)
            incoming[target].append(source)

        out_degree = {
            paper_id: len(outgoing[paper_id])
            for paper_id in paper_ids
        }

        for _ in range(iterations):

            updated_ranks = {}

            for paper_id in paper_ids:

                incoming_score = sum(
                    ranks[source]
                    / (out_degree[source] or 1)
                    for source in incoming[paper_id]
                    if source in ranks
                )

                updated_ranks[paper_id] = (
                    (1 - damping) / node_count
                    + damping * incoming_score
                )

            ranks = updated_ranks

        return PageRankCalculator._normalize(
            ranks
        )

    @staticmethod
    def _normalize(
        ranks: dict[str, float],
    ) -> dict[str, float]:

        if not ranks:
            return {}

        max_rank = max(ranks.values())

        if max_rank == 0:
            return ranks

        return {
            paper_id: score / max_rank
            for paper_id, score in ranks.items()
        }


# ---------------------------------------------------------------------------
# Citation Expansion
# ---------------------------------------------------------------------------

class CitationGraphExpander:
    """
    Expands papers through citation traversal.
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        max_expansion_per_paper: int = DEFAULT_MAX_EXPANSION,
        concurrency_limit: int = DEFAULT_CONCURRENCY,
    ):

        self.top_k = top_k

        self.max_expansion_per_paper = (
            max_expansion_per_paper
        )

        self.concurrency_limit = (
            concurrency_limit
        )

        self._openalex = OpenAlexAdapter()

    async def expand(
        self,
        papers: list[Paper],
    ) -> tuple[list[Paper], list[CitationEdge]]:

        seed_papers = papers[: self.top_k]

        logger.info(
            "Citation expansion starting seed_count=%s",
            len(seed_papers),
        )

        semaphore = asyncio.Semaphore(
            self.concurrency_limit
        )

        async def expand_seed(
            paper: Paper,
        ) -> tuple[list[Paper], list[CitationEdge]]:

            async with semaphore:
                return await self._expand_seed_paper(
                    paper
                )

        tasks = [
            expand_seed(paper)
            for paper in seed_papers
        ]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        expanded_papers = []
        edges = []

        for result in results:

            if not isinstance(result, tuple):
                continue

            papers_batch, edge_batch = result

            expanded_papers.extend(papers_batch)
            edges.extend(edge_batch)

        combined_papers = self._deduplicate_papers(
            papers + expanded_papers
        )

        logger.info(
            "Citation expansion complete "
            "original=%s expanded=%s edges=%s",
            len(papers),
            len(combined_papers),
            len(edges),
        )

        return combined_papers, edges

    async def _expand_seed_paper(
        self,
        paper: Paper,
    ) -> tuple[list[Paper], list[CitationEdge]]:

        openalex_id = (
            paper.external_ids.openalex
        )

        if not openalex_id:
            return [], []

        citation_task = (
            self._openalex.fetch_citations(
                openalex_id
            )
        )

        reference_task = (
            self._openalex.fetch_references(
                openalex_id
            )
        )

        citation_results, reference_results = (
            await asyncio.gather(
                citation_task,
                reference_task,
                return_exceptions=True,
            )
        )

        expanded_papers = []
        edges = []

        if isinstance(citation_results, list):

            citation_results = (
                citation_results[
                    : self.max_expansion_per_paper
                ]
            )

            expanded_papers.extend(
                citation_results
            )

            for cited_by_paper in citation_results:

                edges.append(
                    CitationEdge(
                        source_paper_id=(
                            cited_by_paper.paper_id
                        ),
                        target_paper_id=(
                            paper.paper_id
                        ),
                        relation=(
                            CitationRelation.CITES
                        ),
                    )
                )

        if isinstance(reference_results, list):

            reference_results = (
                reference_results[
                    : self.max_expansion_per_paper
                ]
            )

            expanded_papers.extend(
                reference_results
            )

            for referenced_paper in (
                reference_results
            ):

                edges.append(
                    CitationEdge(
                        source_paper_id=(
                            paper.paper_id
                        ),
                        target_paper_id=(
                            referenced_paper.paper_id
                        ),
                        relation=(
                            CitationRelation.CITES
                        ),
                    )
                )

        return expanded_papers, edges

    def apply_graph_scores(
        self,
        papers: list[Paper],
        edges: list[CitationEdge],
    ) -> list[Paper]:

        paper_map = {
            paper.paper_id: paper
            for paper in papers
        }

        edge_pairs = [
            (
                edge.source_paper_id,
                edge.target_paper_id,
            )
            for edge in edges
            if (
                edge.source_paper_id
                in paper_map
                and edge.target_paper_id
                in paper_map
            )
        ]

        scores = PageRankCalculator.compute(
            edges=edge_pairs,
            paper_ids=set(paper_map.keys()),
        )

        for paper_id, score in scores.items():

            paper = paper_map.get(paper_id)

            if not paper:
                continue

            paper.ranking_features = (
                RankingFeatures(
                    semantic_similarity=(
                        paper.ranking_features.semantic_similarity
                    ),
                    citation_boost=(
                        paper.ranking_features.citation_boost
                    ),
                    recency_boost=(
                        paper.ranking_features.recency_boost
                    ),
                    venue_score=(
                        paper.ranking_features.venue_score
                    ),
                    keyword_overlap=(
                        paper.ranking_features.keyword_overlap
                    ),
                    graph_centrality=round(
                        score,
                        4,
                    ),
                    mmr_score=(
                        paper.ranking_features.mmr_score
                    ),
                )
            )

        logger.info(
            "PageRank completed nodes=%s",
            len(scores),
        )

        return papers

    @staticmethod
    def _deduplicate_papers(
        papers: list[Paper],
    ) -> list[Paper]:

        seen = set()

        deduplicated = []

        for paper in papers:

            identity = (
                paper.external_ids.doi
                or paper.external_ids.openalex
                or paper.title.lower()
            )

            if identity in seen:
                continue

            seen.add(identity)

            deduplicated.append(paper)

        return deduplicated