"""
Platform test suite for Research Discovery.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Protocol

from research_discovery.models.paper import (
    DiscoveryResult,
    ExternalIDs,
    Paper,
    PaperSource,
    PaperTier,
    ResearchQuery,
)
from research_discovery.services.dedup.engine import (
    DeduplicationEngine,
)
from research_discovery.services.embedding.service import (
    EmbeddingService,
)
from research_discovery.services.ranking.engine import (
    RankingEngine,
)
from research_discovery.services.storage.json_store import (
    JSONStorageService,
)

# ---------------------------------------------------------------------------
# Fixture Builders
# ---------------------------------------------------------------------------

class PaperFactory:
    """Reusable paper fixture builder."""

    counter = 0

    @classmethod
    def create(
        cls,
        title: str = "Test Paper",
        doi: str | None = None,
        arxiv: str | None = None,
        citation_count: int = 100,
        year: int = 2023,
        venue: str = "NeurIPS",
        abstract: str = (
            "Research abstract."
        ),
        source: PaperSource = (
            PaperSource.OPENALEX
        ),
    ) -> Paper:

        cls.counter += 1

        return Paper(
            source=source,
            title=title,
            abstract=abstract,
            year=year,
            venue=venue,
            citation_count=(
                citation_count
            ),
            external_ids=ExternalIDs(
                doi=doi,
                arxiv=arxiv,
            ),
        )


# ---------------------------------------------------------------------------
# Contract Tests
# ---------------------------------------------------------------------------

class RetrievalProviderContract(
    Protocol
):

    async def search(
        self,
        query: str,
        limit: int,
    ):
        ...


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestPaperModel(
    unittest.IsolatedAsyncioTestCase
):

    def test_paper_creation(
        self,
    ):

        paper = (
            PaperFactory.create()
        )

        self.assertEqual(
            paper.title,
            "Test Paper",
        )

        self.assertIsNotNone(
            paper.paper_id
        )

    def test_discovery_result(
        self,
    ):

        result = DiscoveryResult(
            query=ResearchQuery(
                original_idea="test"
            ),
            highly_relevant=[
                PaperFactory.create()
            ],
        )

        self.assertEqual(
            len(
                result.all_papers()
            ),
            1,
        )


# ---------------------------------------------------------------------------
# Deduplication Tests
# ---------------------------------------------------------------------------

class TestDeduplicationEngine(
    unittest.IsolatedAsyncioTestCase
):

    def setUp(
        self,
    ):

        self.engine = (
            DeduplicationEngine(
                fuzzy_threshold=0.94
            )
        )

    def test_deduplicates_doi(
        self,
    ):

        papers = [
            PaperFactory.create(
                title="Paper A",
                doi="10.1/test",
            ),
            PaperFactory.create(
                title="Paper B",
                doi="10.1/test",
            ),
        ]

        result = (
            self.engine.deduplicate(
                papers
            )
        )

        self.assertEqual(
            len(result),
            1,
        )

    def test_preserves_distinct_papers(
        self,
    ):

        papers = [
            PaperFactory.create(
                title="Transformer"
            ),
            PaperFactory.create(
                title="RLHF"
            ),
        ]

        result = (
            self.engine.deduplicate(
                papers
            )
        )

        self.assertEqual(
            len(result),
            2,
        )


# ---------------------------------------------------------------------------
# Embedding Tests
# ---------------------------------------------------------------------------

class MockEmbeddingProvider:
    async def embed(self, texts):
        import numpy as np
        return [np.random.rand(10).tolist() for _ in texts]

class TestEmbeddingService(
    unittest.IsolatedAsyncioTestCase
):

    def setUp(
        self,
    ):

        self.embedder = (
            EmbeddingService()
        )
        self.embedder.provider = MockEmbeddingProvider()

    async def test_embed_query(
        self,
    ):

        embedding = (
            await self.embedder.embed_query(
                "transformer attention"
            )
        )

        self.assertGreater(
            len(embedding),
            0,
        )

    async def test_compute_similarity(
        self,
    ):

        papers = [
            PaperFactory.create(
                title=(
                    "Transformer "
                    "Attention"
                )
            )
        ]

        await self.embedder.embed_papers(
            papers
        )

        query_embedding = (
            await self.embedder.embed_query(
                "transformer"
            )
        )

        result = (
            self.embedder.compute_similarity(
                papers,
                query_embedding,
            )
        )

        self.assertGreaterEqual(
            result[0]
            .similarity_score,
            0.0,
        )


# ---------------------------------------------------------------------------
# Ranking Tests
# ---------------------------------------------------------------------------

class TestRankingEngine(
    unittest.IsolatedAsyncioTestCase
):

    def setUp(
        self,
    ):

        self.ranker = (
            RankingEngine()
        )

        self.embedder = (
            EmbeddingService()
        )
        self.embedder.provider = MockEmbeddingProvider()

    async def test_rank_assigns_scores(
        self,
    ):

        papers = [
            PaperFactory.create(
                title=(
                    "Transformer "
                    "Architecture"
                ),
                citation_count=500,
            ),
            PaperFactory.create(
                title=(
                    "RLHF Reward "
                    "Modeling"
                ),
                citation_count=50,
            ),
        ]

        await self.embedder.embed_papers(
            papers
        )

        query_embedding = (
            await self.embedder.embed_query(
                "transformer"
            )
        )

        self.embedder.compute_similarity(
            papers,
            query_embedding,
        )

        ranked = (
            self.ranker.rank(
                papers,
                "transformer",
                query_embedding,
            )
        )

        self.assertEqual(
            len(ranked),
            2,
        )

        for paper in ranked:

            self.assertGreater(
                paper.final_score,
                0.0,
            )

            self.assertIsNotNone(
                paper.tier
            )


# ---------------------------------------------------------------------------
# Storage Tests
# ---------------------------------------------------------------------------

class TestStorageService(
    unittest.IsolatedAsyncioTestCase
):

    def setUp(
        self,
    ):

        self.temp_dir = (
            tempfile.mkdtemp()
        )

        self.storage = (
            JSONStorageService(
                storage_dir=Path(
                    self.temp_dir
                )
            )
        )

    def test_save_and_load(
        self,
    ):

        result = DiscoveryResult(
            query=ResearchQuery(
                original_idea="storage"
            ),
            highly_relevant=[
                PaperFactory.create()
            ],
            total_papers=1,
        )

        path = (
            self.storage.save_result(
                result
            )
        )

        loaded = (
            self.storage.load_result(
                path
            )
        )

        self.assertIsNotNone(
            loaded
        )

        self.assertEqual(
            loaded.total_papers,
            1,
        )

    def test_csv_export(
        self,
    ):

        result = DiscoveryResult(
            query=ResearchQuery(
                original_idea="csv"
            ),
            highly_relevant=[
                PaperFactory.create()
            ],
            total_papers=1,
        )

        output = (
            Path(self.temp_dir)
            / "export.csv"
        )

        self.storage.export_csv(
            result,
            str(output),
        )

        self.assertTrue(
            output.exists()
        )


# ---------------------------------------------------------------------------
# Runtime Smoke Tests
# ---------------------------------------------------------------------------

class TestRuntimeSmoke(
    unittest.IsolatedAsyncioTestCase
):

    def test_pipeline_importable(
        self,
    ):

        from research_discovery.core.pipeline import (
            ResearchDiscoveryPipeline,
        )

        pipeline = (
            ResearchDiscoveryPipeline()
        )

        self.assertIsNotNone(
            pipeline
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )