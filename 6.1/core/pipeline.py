"""
Research Discovery Pipeline Orchestrator.

Ties all modules together into the full async pipeline:

  User Research Idea
       ↓
  Query Expansion Agent
       ↓
  Parallel Retrieval (OpenAlex + S2 + arXiv + CrossRef enrichment)
       ↓
  Deduplication
       ↓
  Embedding Generation
       ↓
  First-Level Ranking
       ↓
  Citation Graph Expansion
       ↓
  Re-embed + Second-Level Ranking (with MMR)
       ↓
  Final Research Corpus (tiered)
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_logger, Timer
from research_discovery.models.paper import (
    DiscoveryResult,
    Paper,
    PaperTier,
    ResearchQuery,
)
from research_discovery.services.citation_graph.expander import CitationGraphExpander
from research_discovery.services.dedup.engine import DeduplicationEngine
from research_discovery.services.embedding.service import EmbeddingService
from research_discovery.services.query_expansion.agent import QueryExpansionAgent
from research_discovery.services.ranking.engine import RankingEngine
from research_discovery.services.retrieval.arxiv import ArxivAdapter
from research_discovery.services.retrieval.crossref import CrossRefAdapter
from research_discovery.services.retrieval.openalex import OpenAlexAdapter
from research_discovery.services.retrieval.semantic_scholar import SemanticScholarAdapter

logger = get_logger(__name__)


class ResearchDiscoveryPipeline:
    """
    Full end-to-end research discovery pipeline.

    Usage:
        pipeline = ResearchDiscoveryPipeline()
        result = await pipeline.run("Using LLMs for automated code review")
    """

    def __init__(
        self,
        num_expansion_queries: int = 10,
        results_per_query: int = 20,
        use_semantic_scholar: bool = True,
        use_arxiv: bool = True,
        use_crossref_enrichment: bool = True,
        use_citation_expansion: bool = True,
        mmr_lambda: float = 0.7,
        max_final_papers: int = 150,
    ):
        self.num_expansion_queries = num_expansion_queries
        self.results_per_query = results_per_query
        self.use_semantic_scholar = use_semantic_scholar
        self.use_arxiv = use_arxiv
        self.use_crossref_enrichment = use_crossref_enrichment
        self.use_citation_expansion = use_citation_expansion
        self.max_final_papers = max_final_papers

        # Initialize all services
        self._query_expander = QueryExpansionAgent(num_queries=num_expansion_queries)
        self._openalex = OpenAlexAdapter()
        self._semantic_scholar = SemanticScholarAdapter()
        self._arxiv = ArxivAdapter()
        self._crossref = CrossRefAdapter()
        self._dedup = DeduplicationEngine(
            fuzzy_threshold=settings.retrieval.fuzzy_dedup_threshold
        )
        self._embedder = EmbeddingService()
        self._ranker = RankingEngine(mmr_lambda=mmr_lambda, max_papers=max_final_papers)
        self._citation_expander = CitationGraphExpander(
            top_k=settings.retrieval.citation_expansion_top_k
        )

    # -----------------------------------------------------------------------
    # Step 1: Query Expansion
    # -----------------------------------------------------------------------

    async def _expand_queries(self, idea: str) -> ResearchQuery:
        queries = await self._query_expander.expand(idea)
        return ResearchQuery(original_idea=idea, expanded_queries=queries)

    # -----------------------------------------------------------------------
    # Step 2: Parallel Retrieval
    # -----------------------------------------------------------------------

    async def _retrieve_for_query(self, query: str) -> list[Paper]:
        """Run all enabled APIs in parallel for a single query."""
        tasks = [self._openalex.search(query, per_page=self.results_per_query)]

        if self.use_semantic_scholar:
            tasks.append(self._semantic_scholar.search(query, limit=self.results_per_query))

        if self.use_arxiv:
            tasks.append(self._arxiv.search(query, max_results=self.results_per_query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        papers: list[Paper] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Retrieval task failed: {r}")
            else:
                papers.extend(r.papers)

        return papers

    async def _retrieve_all(self, queries: list[str]) -> list[Paper]:
        """
        Fan out across all expanded queries, concurrently.
        Limit parallelism to avoid API abuse.
        """
        semaphore = asyncio.Semaphore(4)  # max 4 concurrent query groups

        async def _bounded(q: str) -> list[Paper]:
            async with semaphore:
                return await self._retrieve_for_query(q)

        results = await asyncio.gather(*[_bounded(q) for q in queries], return_exceptions=True)

        all_papers: list[Paper] = []
        for r in results:
            if isinstance(r, list):
                all_papers.extend(r)
            elif isinstance(r, Exception):
                logger.warning(f"Query retrieval error: {r}")

        logger.info(f"Raw retrieval: {len(all_papers)} total papers before dedup")
        return all_papers

    # -----------------------------------------------------------------------
    # Step 3: CrossRef Enrichment
    # -----------------------------------------------------------------------

    async def _enrich(self, papers: list[Paper]) -> list[Paper]:
        if not self.use_crossref_enrichment:
            return papers
        return await self._crossref.enrich_papers(papers)

    # -----------------------------------------------------------------------
    # Full Pipeline
    # -----------------------------------------------------------------------

    async def run(self, research_idea: str) -> DiscoveryResult:
        """
        Execute the full discovery pipeline.

        Args:
            research_idea: Natural language description of the research topic

        Returns:
            DiscoveryResult with tiered paper lists
        """
        timer = Timer()
        logger.info(f"=== Research Discovery Pipeline START ===")
        logger.info(f"Idea: '{research_idea}'")

        try:
            # Step 1: Query Expansion
            logger.info("Step 1/7: Query Expansion")
            research_query = await self._expand_queries(research_idea)
            logger.info(f"  → {len(research_query.expanded_queries)} queries generated")

            # Step 2: Parallel Retrieval
            logger.info("Step 2/7: Parallel Retrieval")
            raw_papers = await self._retrieve_all(research_query.expanded_queries)

            # Step 3: CrossRef Enrichment
            logger.info("Step 3/7: CrossRef Enrichment")
            raw_papers = await self._enrich(raw_papers)

            # Step 4: Deduplication
            logger.info("Step 4/7: Deduplication")
            deduped = self._dedup.deduplicate(raw_papers)

            # Step 5: Embedding Generation
            logger.info("Step 5/7: Embedding Generation")
            query_embedding = self._embedder.embed_query(research_idea)
            embedded_papers = self._embedder.embed_papers(deduped)
            papers_with_scores = self._embedder.compute_similarity(
                embedded_papers, query_embedding
            )

            # Step 6: First-level Ranking
            logger.info("Step 6/7: First-Level Ranking")
            ranked = self._ranker.rank(papers_with_scores, research_idea, query_embedding)

            # Step 7: Citation Graph Expansion (optional)
            if self.use_citation_expansion:
                logger.info("Step 7/7: Citation Graph Expansion")
                expanded, edges = await self._citation_expander.expand(ranked)

                # Dedup again after expansion
                expanded = self._dedup.deduplicate(expanded)

                # Embed new papers only
                new_papers = [p for p in expanded if not p.embedding]
                if new_papers:
                    self._embedder.embed_papers(new_papers)
                    self._embedder.compute_similarity(new_papers, query_embedding)

                # Apply graph centrality scores
                expanded = self._citation_expander.apply_graph_scores(expanded, edges)

                # Second-level ranking with all papers
                final_papers = self._ranker.rank(expanded, research_idea, query_embedding)
            else:
                logger.info("Step 7/7: Citation expansion disabled — using first-level ranking")
                final_papers = ranked

            # Partition into tiers
            tiers: dict[str, list[Paper]] = {
                PaperTier.HIGHLY_RELEVANT: [],
                PaperTier.RELEVANT_BACKGROUND: [],
                PaperTier.ADJACENT_WORK: [],
                PaperTier.HISTORICAL_FOUNDATIONS: [],
            }
            for paper in final_papers:
                tier_key = paper.tier or PaperTier.ADJACENT_WORK
                tiers[tier_key].append(paper)

            elapsed = timer.elapsed()
            logger.info(
                f"=== Pipeline DONE in {elapsed}s | "
                f"Total papers: {len(final_papers)} ==="
            )

            return DiscoveryResult(
                query=research_query,
                highly_relevant=tiers[PaperTier.HIGHLY_RELEVANT],
                relevant_background=tiers[PaperTier.RELEVANT_BACKGROUND],
                adjacent_work=tiers[PaperTier.ADJACENT_WORK],
                historical_foundations=tiers[PaperTier.HISTORICAL_FOUNDATIONS],
                total_papers=len(final_papers),
                processing_time_seconds=elapsed,
                metadata={
                    "num_expansion_queries": len(research_query.expanded_queries),
                    "raw_retrieved": len(raw_papers),
                    "after_dedup": len(deduped),
                    "after_citation_expansion": len(final_papers),
                    "embedding_dimension": self._embedder.dimension,
                },
            )

        except Exception as exc:
            elapsed = timer.elapsed()
            logger.error(f"Pipeline failed after {elapsed}s: {exc}")
            logger.debug(traceback.format_exc())
            raise


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

async def discover(research_idea: str, **kwargs) -> DiscoveryResult:
    """Top-level convenience function."""
    pipeline = ResearchDiscoveryPipeline(**kwargs)
    return await pipeline.run(research_idea)