"""
Citation Graph Expansion.

Takes the top-K papers from initial ranking and expands the corpus
by fetching their references and citing papers.

Then scores graph centrality using a simple PageRank implementation
to identify seminal / bridge papers.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_logger
from research_discovery.models.paper import CitationEdge, CitationRelation, Paper, PaperSource
from research_discovery.services.retrieval.openalex import OpenAlexAdapter
from research_discovery.services.retrieval.semantic_scholar import SemanticScholarAdapter

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight PageRank
# ---------------------------------------------------------------------------

def _pagerank(
    edges: list[tuple[str, str]],
    paper_ids: set[str],
    damping: float = 0.85,
    iterations: int = 30,
) -> dict[str, float]:
    """
    Simple iterative PageRank.

    Args:
        edges: list of (source_id, target_id) — source cites target
        paper_ids: all node IDs
        damping: damping factor (standard 0.85)
        iterations: convergence iterations
    """
    n = len(paper_ids)
    if n == 0:
        return {}

    # Initialize
    rank: dict[str, float] = {pid: 1.0 / n for pid in paper_ids}

    # Build adjacency (out-edges)
    out_edges: dict[str, list[str]] = defaultdict(list)
    in_edges: dict[str, list[str]] = defaultdict(list)
    for src, tgt in edges:
        out_edges[src].append(tgt)
        in_edges[tgt].append(src)

    out_degree = {pid: len(out_edges[pid]) for pid in paper_ids}

    for _ in range(iterations):
        new_rank: dict[str, float] = {}
        for pid in paper_ids:
            incoming_sum = sum(
                rank[src] / (out_degree[src] or 1)
                for src in in_edges[pid]
                if src in rank
            )
            new_rank[pid] = (1 - damping) / n + damping * incoming_sum
        rank = new_rank

    # Normalize to [0, 1]
    max_rank = max(rank.values()) if rank else 1.0
    return {pid: r / max_rank for pid, r in rank.items()}


# ---------------------------------------------------------------------------
# Citation Graph Builder
# ---------------------------------------------------------------------------

class CitationGraphExpander:
    """
    Expands the initial paper set by following citation links.
    Uses OpenAlex as the primary citation graph source.
    """

    def __init__(self, top_k: int = 20, max_expansion_per_paper: int = 30):
        self.top_k = top_k
        self.max_expansion_per_paper = max_expansion_per_paper
        self._openalex = OpenAlexAdapter()

    async def _expand_paper(self, paper: Paper) -> list[Paper]:
        """Fetch citations and references for a single paper."""
        openalex_id = paper.external_ids.openalex
        if not openalex_id:
            return []

        tasks = [
            self._openalex.fetch_citations(openalex_id),
            self._openalex.fetch_references(openalex_id),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        expanded: list[Paper] = []
        for r in results:
            if isinstance(r, list):
                expanded.extend(r[:self.max_expansion_per_paper])

        return expanded

    async def expand(self, papers: list[Paper]) -> tuple[list[Paper], list[CitationEdge]]:
        """
        Takes top-K ranked papers, expands via citation graph.

        Returns:
            (all_papers, citation_edges)
            all_papers = original + expansion papers
            citation_edges = graph edges for PageRank
        """
        seed_papers = papers[: self.top_k]
        logger.info(f"Citation expansion: starting from {len(seed_papers)} seed papers")

        # Fan out — fetch citations/references for all seed papers concurrently
        # Use semaphore to avoid hammering the API
        semaphore = asyncio.Semaphore(5)

        async def _expand_with_sem(paper: Paper) -> list[Paper]:
            async with semaphore:
                return await self._expand_paper(paper)

        tasks = [_expand_with_sem(p) for p in seed_papers]
        expansion_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_new: list[Paper] = []
        for r in expansion_results:
            if isinstance(r, list):
                all_new.extend(r)

        logger.info(f"Citation expansion: fetched {len(all_new)} candidate papers")

        # Build citation edges for PageRank
        edges: list[CitationEdge] = []
        existing_ids = {p.paper_id for p in papers}

        for seed, new_papers in zip(seed_papers, expansion_results):
            if not isinstance(new_papers, list):
                continue
            for new_p in new_papers:
                if new_p.source == PaperSource.CITATION_EXPANSION:
                    # seed → new_p means seed cites new_p
                    edges.append(CitationEdge(
                        source_paper_id=seed.paper_id,
                        target_paper_id=new_p.paper_id,
                        relation=CitationRelation.CITES,
                    ))

        combined = papers + all_new
        logger.info(
            f"Citation expansion: {len(papers)} → {len(combined)} papers, "
            f"{len(edges)} citation edges"
        )
        return combined, edges

    def apply_graph_scores(
        self,
        papers: list[Paper],
        edges: list[CitationEdge],
    ) -> list[Paper]:
        """
        Run PageRank on the citation graph and store centrality
        in paper.ranking_features.graph_centrality.
        """
        paper_map = {p.paper_id: p for p in papers}
        paper_ids = set(paper_map.keys())

        edge_tuples = [
            (e.source_paper_id, e.target_paper_id)
            for e in edges
            if e.source_paper_id in paper_ids and e.target_paper_id in paper_ids
        ]

        pr = _pagerank(edge_tuples, paper_ids)

        for pid, score in pr.items():
            if pid in paper_map:
                paper_map[pid].ranking_features.graph_centrality = round(score, 4)

        logger.info(f"PageRank computed for {len(pr)} papers")
        return papers