"""
Test suite for Research Discovery Module.
Tests run without real API calls using mocked HTTP responses.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Make sure the package root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_discovery.models.paper import (
    Author,
    CitationEdge,
    CitationRelation,
    DiscoveryResult,
    ExternalIDs,
    Paper,
    PaperSource,
    PaperTier,
    ResearchQuery,
    RankingFeatures,
)
from research_discovery.services.dedup.engine import (
    DeduplicationEngine,
    _normalize_title,
    _title_fingerprint,
)
from research_discovery.services.embedding.service import (
    EmbeddingService,
    _cosine_similarity,
    _tfidf_embeddings,
)
from research_discovery.services.ranking.engine import (
    RankingEngine,
    _citation_score,
    _recency_score,
    _venue_score,
)
from research_discovery.services.citation_graph.expander import _pagerank


def _make_paper(
    title: str = "Test Paper",
    doi: str = None,
    arxiv: str = None,
    year: int = 2023,
    venue: str = "NeurIPS",
    citation_count: int = 100,
    source: PaperSource = PaperSource.OPENALEX,
    abstract: str = "Abstract text here",
) -> Paper:
    return Paper(
        source=source,
        external_ids=ExternalIDs(doi=doi, arxiv=arxiv),
        title=title,
        abstract=abstract,
        year=year,
        venue=venue,
        citation_count=citation_count,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestPaperModel(unittest.TestCase):

    def test_paper_creation(self):
        paper = _make_paper()
        self.assertEqual(paper.title, "Test Paper")
        self.assertIsNotNone(paper.paper_id)
        self.assertIsNotNone(paper.created_at)

    def test_paper_requires_title(self):
        with self.assertRaises(Exception):
            Paper(source=PaperSource.OPENALEX, title="", external_ids=ExternalIDs())

    def test_paper_to_dict(self):
        paper = _make_paper()
        d = paper.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["title"], "Test Paper")

    def test_discovery_result_all_papers(self):
        p1 = _make_paper("Paper 1")
        p2 = _make_paper("Paper 2")
        result = DiscoveryResult(
            query=ResearchQuery(original_idea="test"),
            highly_relevant=[p1],
            relevant_background=[p2],
        )
        all_p = result.all_papers()
        self.assertEqual(len(all_p), 2)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplicationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DeduplicationEngine(fuzzy_threshold=0.94)

    def test_normalize_title(self):
        t1 = _normalize_title("A Study of the Transformer Architecture")
        t2 = _normalize_title("study transformer architecture")
        # Both should produce similar normalized forms
        self.assertIn("transformer", t1)
        self.assertIn("transformer", t2)

    def test_doi_dedup(self):
        p1 = _make_paper("Paper A", doi="10.1234/test")
        p2 = _make_paper("Paper A Variant", doi="10.1234/test")
        result = self.engine.deduplicate([p1, p2])
        self.assertEqual(len(result), 1)

    def test_doi_case_normalization(self):
        p1 = _make_paper("Paper A", doi="10.1234/TEST")
        p2 = _make_paper("Paper A Variant", doi="10.1234/test")
        result = self.engine.deduplicate([p1, p2])
        self.assertEqual(len(result), 1)

    def test_doi_url_stripping(self):
        p1 = _make_paper("Paper A", doi="https://doi.org/10.1234/test")
        p2 = _make_paper("Paper A", doi="10.1234/test")
        result = self.engine.deduplicate([p1, p2])
        self.assertEqual(len(result), 1)

    def test_title_fingerprint_dedup(self):
        p1 = _make_paper("Attention Is All You Need")
        p2 = _make_paper("Attention Is All You Need")
        result = self.engine.deduplicate([p1, p2])
        self.assertEqual(len(result), 1)

    def test_different_papers_preserved(self):
        p1 = _make_paper("Attention Is All You Need")
        p2 = _make_paper("BERT: Pre-training Deep Bidirectional Transformers")
        p3 = _make_paper("GPT-4 Technical Report")
        result = self.engine.deduplicate([p1, p2, p3])
        self.assertEqual(len(result), 3)

    def test_merge_preserves_queries(self):
        p1 = _make_paper("Paper A", doi="10.1/test")
        p1.retrieved_from_queries = ["query 1"]
        p2 = _make_paper("Paper A", doi="10.1/test")
        p2.retrieved_from_queries = ["query 2"]
        result = self.engine.deduplicate([p1, p2])
        self.assertEqual(len(result), 1)
        self.assertIn("query 1", result[0].retrieved_from_queries)
        self.assertIn("query 2", result[0].retrieved_from_queries)

    def test_merge_prefers_higher_citations(self):
        p1 = _make_paper("Paper A", doi="10.1/test", citation_count=50)
        p2 = _make_paper("Paper A", doi="10.1/test", citation_count=500)
        result = self.engine.deduplicate([p1, p2])
        self.assertEqual(result[0].citation_count, 500)

    def test_empty_input(self):
        result = self.engine.deduplicate([])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

class TestEmbeddingService(unittest.TestCase):

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0, places=5)

    def test_cosine_similarity_dimension_mismatch(self):
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        self.assertEqual(_cosine_similarity(a, b), 0.0)

    def test_tfidf_embeddings_shape(self):
        texts = ["attention mechanism transformer", "reinforcement learning reward"]
        embeddings = _tfidf_embeddings(texts)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), len(embeddings[1]))

    def test_tfidf_embeddings_normalized(self):
        import math
        texts = ["hello world test"]
        embs = _tfidf_embeddings(texts)
        norm = math.sqrt(sum(x * x for x in embs[0]))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_tfidf_similar_texts_higher_similarity(self):
        # Use a larger corpus so IDF scores don't zero out shared terms
        texts = [
            "transformer attention mechanism neural network",
            "attention mechanism transformer self neural sequence",
            "completely unrelated food recipe cooking ingredients",
            "baking chocolate cake flour butter sugar",
            "restaurant kitchen chef meal dinner",
            "machine learning gradient descent optimization loss",
        ]
        embs = _tfidf_embeddings(texts)
        # texts[0] and texts[1] share many tokens; texts[0] and texts[2] share none
        sim_related = _cosine_similarity(embs[0], embs[1])
        sim_unrelated = _cosine_similarity(embs[0], embs[2])
        self.assertGreaterEqual(sim_related, sim_unrelated)


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------

class TestRankingFunctions(unittest.TestCase):

    def test_citation_score_zero(self):
        self.assertEqual(_citation_score(0), 0.0)

    def test_citation_score_high(self):
        # 1000 citations → ~1.0
        self.assertGreater(_citation_score(1000), 0.9)

    def test_citation_score_bounded(self):
        self.assertLessEqual(_citation_score(999999), 1.0)
        self.assertGreaterEqual(_citation_score(0), 0.0)

    def test_recency_score_current_year(self):
        import datetime
        current_year = datetime.datetime.utcnow().year
        self.assertAlmostEqual(_recency_score(current_year), 1.0, places=5)

    def test_recency_score_old_paper(self):
        self.assertEqual(_recency_score(2000), 0.0)

    def test_recency_score_unknown(self):
        score = _recency_score(None)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_venue_score_high_impact(self):
        self.assertEqual(_venue_score("NeurIPS 2023"), 1.0)
        self.assertEqual(_venue_score("ICML"), 1.0)
        self.assertEqual(_venue_score("ACL 2022"), 1.0)

    def test_venue_score_unknown(self):
        score = _venue_score("Some Random Workshop")
        self.assertLess(score, 1.0)

    def test_venue_score_none(self):
        score = _venue_score(None)
        self.assertGreaterEqual(score, 0.0)


class TestRankingEngine(unittest.TestCase):

    def _make_papers_with_embeddings(self, n: int = 5) -> list[Paper]:
        papers = []
        for i in range(n):
            p = _make_paper(
                title=f"Test Paper {i}",
                doi=f"10.1/{i}",
                citation_count=i * 100,
            )
            # Simple orthogonal-ish embeddings
            emb = [0.0] * 10
            emb[i % 10] = 1.0
            p.embedding = emb
            p.ranking_features.semantic_similarity = 0.5 + i * 0.05
            papers.append(p)
        return papers

    def test_rank_returns_all_papers(self):
        engine = RankingEngine()
        papers = self._make_papers_with_embeddings(5)
        query_emb = [1.0] + [0.0] * 9
        ranked = engine.rank(papers, "test query", query_emb)
        self.assertEqual(len(ranked), 5)

    def test_rank_assigns_tiers(self):
        engine = RankingEngine()
        papers = self._make_papers_with_embeddings(3)
        query_emb = [1.0] + [0.0] * 9
        ranked = engine.rank(papers, "test query", query_emb)
        for p in ranked:
            self.assertIsNotNone(p.tier)

    def test_rank_final_score_populated(self):
        engine = RankingEngine()
        papers = self._make_papers_with_embeddings(3)
        query_emb = [1.0] + [0.0] * 9
        ranked = engine.rank(papers, "test query", query_emb)
        for p in ranked:
            self.assertGreater(p.final_score, 0.0)


# ---------------------------------------------------------------------------
# Citation graph / PageRank tests
# ---------------------------------------------------------------------------

class TestPageRank(unittest.TestCase):

    def test_empty_graph(self):
        result = _pagerank([], set())
        self.assertEqual(result, {})

    def test_single_node(self):
        result = _pagerank([], {"p1"})
        self.assertIn("p1", result)

    def test_linear_chain(self):
        # p1 → p2 → p3: p3 should have highest in-degree
        edges = [("p1", "p2"), ("p2", "p3")]
        ids = {"p1", "p2", "p3"}
        result = _pagerank(edges, ids)
        self.assertIn("p3", result)
        # p3 gets citations from p2, which itself has p1 pointing at it
        # so p3 should rank higher than p1 (which gets no incoming)
        self.assertGreaterEqual(result["p3"], result["p1"])

    def test_hub_node(self):
        # p0 is cited by many nodes — should rank highest
        edges = [(f"p{i}", "p0") for i in range(1, 6)]
        ids = {f"p{i}" for i in range(6)}
        result = _pagerank(edges, ids)
        self.assertEqual(max(result, key=result.get), "p0")

    def test_scores_bounded(self):
        edges = [("p1", "p2"), ("p2", "p3"), ("p3", "p1")]
        ids = {"p1", "p2", "p3"}
        result = _pagerank(edges, ids)
        for score in result.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# Query expansion tests
# ---------------------------------------------------------------------------

class TestQueryExpansion(unittest.TestCase):

    def test_rule_based_includes_original(self):
        from research_discovery.services.query_expansion.agent import _rule_based_expand
        result = _rule_based_expand("LLM for code review", n=5)
        self.assertIn("LLM for code review", result)

    def test_rule_based_deduplicates(self):
        from research_discovery.services.query_expansion.agent import _rule_based_expand
        result = _rule_based_expand("transformer attention", n=10)
        self.assertEqual(len(result), len(set(result)))

    def test_rule_based_max_n(self):
        from research_discovery.services.query_expansion.agent import _rule_based_expand
        result = _rule_based_expand("test idea", n=3)
        self.assertLessEqual(len(result), 3)


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------

class TestStorageService(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        from pathlib import Path
        from research_discovery.services.storage.json_store import JSONStorageService
        self.storage = JSONStorageService(storage_dir=Path(self.tmpdir))

    def test_save_and_load(self):
        paper = _make_paper("Storage Test Paper")
        result = DiscoveryResult(
            query=ResearchQuery(original_idea="storage test"),
            highly_relevant=[paper],
            total_papers=1,
        )
        path = self.storage.save_result(result)
        loaded = self.storage.load_result(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.query.original_idea, "storage test")
        self.assertEqual(len(loaded.highly_relevant), 1)

    def test_list_results(self):
        paper = _make_paper("List Test Paper")
        result = DiscoveryResult(
            query=ResearchQuery(original_idea="list test"),
            highly_relevant=[paper],
        )
        self.storage.save_result(result)
        listing = self.storage.list_results()
        self.assertGreater(len(listing), 0)
        self.assertIn("filename", listing[0])

    def test_export_csv(self):
        import os
        paper = _make_paper("CSV Test Paper")
        paper.tier = PaperTier.HIGHLY_RELEVANT
        result = DiscoveryResult(
            query=ResearchQuery(original_idea="csv test"),
            highly_relevant=[paper],
            total_papers=1,
        )
        csv_path = os.path.join(self.tmpdir, "test_export.csv")
        self.storage.export_csv(result, csv_path)
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path) as f:
            content = f.read()
        self.assertIn("CSV Test Paper", content)
        self.assertIn("tier", content)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Research Discovery Module — Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPaperModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDeduplicationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingService))
    suite.addTests(loader.loadTestsFromTestCase(TestRankingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestRankingEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestPageRank))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryExpansion))
    suite.addTests(loader.loadTestsFromTestCase(TestStorageService))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)