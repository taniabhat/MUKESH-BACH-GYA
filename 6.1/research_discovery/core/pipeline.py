"""
Research discovery orchestration runtime.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from research_discovery.config.settings import settings
from research_discovery.core.runtime import (
    Timer,
    get_logger,
)
from research_discovery.models.paper import (
    DiscoveryResult,
    Paper,
    PaperTier,
    ResearchQuery,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Runtime Context
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:

    research_idea: str

    research_query: ResearchQuery | None = None

    query_embedding: list[float] = field(
        default_factory=list
    )

    raw_papers: list[Paper] = field(
        default_factory=list
    )

    deduplicated_papers: list[Paper] = field(
        default_factory=list
    )

    ranked_papers: list[Paper] = field(
        default_factory=list
    )

    final_papers: list[Paper] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Provider Contracts
# ---------------------------------------------------------------------------

class RetrievalProvider(
    Protocol
):

    async def search(
        self,
        query: str,
        limit: int,
    ):
        ...


# ---------------------------------------------------------------------------
# Service Container
# ---------------------------------------------------------------------------

class ServiceContainer:

    def __init__(self):

        from research_discovery.services.query_expansion.agent import (
            QueryExpansionAgent,
        )

        from research_discovery.services.embedding.service import (
            EmbeddingService,
        )

        from research_discovery.providers.factory import (
            ProviderFactory,
        )

        from research_discovery.services.ranking.engine import (
            RankingEngine,
        )

        from research_discovery.services.dedup.engine import (
            DeduplicationEngine,
        )

        from research_discovery.services.citation_graph.expander import (
            CitationGraphExpander,
        )

        from research_discovery.services.retrieval.openalex import (
            OpenAlexAdapter,
        )

        from research_discovery.services.retrieval.semantic_scholar import (
            SemanticScholarAdapter,
        )

        from research_discovery.services.retrieval.arxiv import (
            ArxivAdapter,
        )

        from research_discovery.services.retrieval.crossref import (
            CrossRefAdapter,
        )

        self.query_expander = (
            QueryExpansionAgent(
                num_queries=(
                    settings.retrieval.num_expansion_queries
                )
            )
        )

        self.embedder = (
            EmbeddingService()
        )

        self.ranker = (
            RankingEngine(
                mmr_lambda=(
                    settings.retrieval.mmr_lambda
                ),
                max_papers=(
                    settings.retrieval.final_corpus_max
                ),
            )
        )

        self.deduplicator = (
            DeduplicationEngine(
                fuzzy_threshold=(
                    settings.retrieval.fuzzy_dedup_threshold
                )
            )
        )

        self.citation_expander = (
            CitationGraphExpander(
                top_k=(
                    settings.retrieval.citation_expansion_top_k
                )
            )
        )

        self.crossref = (
            CrossRefAdapter()
        )

        self.retrievers: list[
            RetrievalProvider
        ] = [
            OpenAlexAdapter(),
            SemanticScholarAdapter(),
            ArxivAdapter(),
        ]


# ---------------------------------------------------------------------------
# Pipeline Stages
# ---------------------------------------------------------------------------

class QueryExpansionStage:

    def __init__(
        self,
        services: ServiceContainer,
    ):

        self.services = services

    async def run(
        self,
        context: PipelineContext,
    ) -> None:

        queries = (
            await self.services.query_expander.expand(
                context.research_idea
            )
        )

        context.research_query = (
            ResearchQuery(
                original_idea=(
                    context.research_idea
                ),
                expanded_queries=queries,
            )
        )


class RetrievalStage:

    def __init__(
        self,
        services: ServiceContainer,
    ):

        self.services = services

    async def run(
        self,
        context: PipelineContext,
    ) -> None:

        semaphore = asyncio.Semaphore(4)

        async def retrieve_query(
            query: str,
        ) -> list[Paper]:

            async with semaphore:

                tasks = [
                    provider.search(
                        query,
                        limit=(
                            settings.retrieval.results_per_query
                        ),
                    )
                    for provider in (
                        self.services.retrievers
                    )
                ]

                results = (
                    await asyncio.gather(
                        *tasks,
                        return_exceptions=True,
                    )
                )

                papers = []

                for result in results:

                    if isinstance(
                        result,
                        Exception,
                    ):

                        logger.warning(
                            "Retrieval provider failed"
                        )

                        continue

                    papers.extend(
                        result.papers
                    )

                return papers

        query_results = (
            await asyncio.gather(
                *[
                    retrieve_query(query)
                    for query in (
                        context.research_query.expanded_queries
                    )
                ]
            )
        )

        for papers in query_results:
            context.raw_papers.extend(
                papers
            )


class EnrichmentStage:

    def __init__(
        self,
        services: ServiceContainer,
    ):

        self.services = services

    async def run(
        self,
        context: PipelineContext,
    ) -> None:

        context.raw_papers = (
            await self.services.crossref.enrich_papers(
                context.raw_papers
            )
        )


class DeduplicationStage:

    def __init__(
        self,
        services: ServiceContainer,
    ):

        self.services = services

    async def run(
        self,
        context: PipelineContext,
    ) -> None:

        context.deduplicated_papers = (
            self.services.deduplicator.deduplicate(
                context.raw_papers
            )
        )


class EmbeddingStage:

    def __init__(
        self,
        services: ServiceContainer,
    ):

        self.services = services

    async def run(
        self,
        context: PipelineContext,
    ) -> None:

        context.query_embedding = (
            self.services.embedder.embed_query(
                context.research_idea
            )
        )

        self.services.embedder.embed_papers(
            context.deduplicated_papers
        )

        self.services.embedder.compute_similarity(
            context.deduplicated_papers,
            context.query_embedding,
        )


class RankingStage:

    def __init__(
        self,
        services: ServiceContainer,
    ):

        self.services = services

    async def run(
        self,
        context: PipelineContext,
    ) -> None:

        context.ranked_papers = (
            self.services.ranker.rank(
                context.deduplicated_papers,
                context.research_idea,
                context.query_embedding,
            )
        )


class CitationExpansionStage:

    def __init__(
        self,
        services: ServiceContainer,
    ):

        self.services = services

    async def run(
        self,
        context: PipelineContext,
    ) -> None:

        expanded, edges = (
            await self.services.citation_expander.expand(
                context.ranked_papers
            )
        )

        expanded = (
            self.services.deduplicator.deduplicate(
                expanded
            )
        )

        self.services.embedder.embed_papers(
            expanded
        )

        self.services.embedder.compute_similarity(
            expanded,
            context.query_embedding,
        )

        expanded = (
            self.services.citation_expander.apply_graph_scores(
                expanded,
                edges,
            )
        )

        context.final_papers = (
            self.services.ranker.rank(
                expanded,
                context.research_idea,
                context.query_embedding,
            )
        )


# ---------------------------------------------------------------------------
# Pipeline Runtime
# ---------------------------------------------------------------------------

class ResearchDiscoveryPipeline:

    def __init__(self):

        self.services = (
            ServiceContainer()
        )

        self.stages = [
            QueryExpansionStage(
                self.services
            ),
            RetrievalStage(
                self.services
            ),
            EnrichmentStage(
                self.services
            ),
            DeduplicationStage(
                self.services
            ),
            EmbeddingStage(
                self.services
            ),
            RankingStage(
                self.services
            ),
            CitationExpansionStage(
                self.services
            ),
        ]

    async def run(
        self,
        research_idea: str,
    ) -> DiscoveryResult:

        timer = Timer()

        context = PipelineContext(
            research_idea=research_idea
        )

        logger.info(
            "Pipeline start idea='%s'",
            research_idea,
        )

        for index, stage in enumerate(
            self.stages,
            start=1,
        ):

            logger.info(
                "Running stage %s/%s: %s",
                index,
                len(self.stages),
                stage.__class__.__name__,
            )

            await stage.run(context)

        elapsed = timer.elapsed()

        final_papers = (
            context.final_papers
            or context.ranked_papers
        )

        tiers = self._partition_tiers(
            final_papers
        )

        logger.info(
            "Pipeline completed papers=%s elapsed=%ss",
            len(final_papers),
            elapsed,
        )

        return DiscoveryResult(
            query=context.research_query,
            highly_relevant=tiers[
                PaperTier.HIGHLY_RELEVANT
            ],
            relevant_background=tiers[
                PaperTier.RELEVANT_BACKGROUND
            ],
            adjacent_work=tiers[
                PaperTier.ADJACENT_WORK
            ],
            historical_foundations=tiers[
                PaperTier.HISTORICAL_FOUNDATIONS
            ],
            total_papers=len(
                final_papers
            ),
            processing_time_seconds=elapsed,
            metadata={
                "raw_retrieved": len(
                    context.raw_papers
                ),
                "after_dedup": len(
                    context.deduplicated_papers
                ),
                "embedding_dimension": (
                    self.services.embedder.dimension
                ),
            },
        )

    @staticmethod
    def _partition_tiers(
        papers: list[Paper],
    ) -> dict:

        tiers = {
            PaperTier.HIGHLY_RELEVANT: [],
            PaperTier.RELEVANT_BACKGROUND: [],
            PaperTier.ADJACENT_WORK: [],
            PaperTier.HISTORICAL_FOUNDATIONS: [],
        }

        for paper in papers:

            tier = (
                paper.tier
                or PaperTier.ADJACENT_WORK
            )

            tiers[tier].append(
                paper
            )

        return tiers


async def discover(
    research_idea: str,
) -> DiscoveryResult:

    pipeline = (
        ResearchDiscoveryPipeline()
    )

    return await pipeline.run(
        research_idea
    )