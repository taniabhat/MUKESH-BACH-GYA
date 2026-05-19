"""
tests/test_rag.py — Stage 6.3 Multimodal RAG Engine Tests
===========================================================
Tests (grouped by component):

  Intent & Query Analysis (no network needed)
  1.  QueryAnalyzer import
  2.  Intent classification — methodology query
  3.  Intent classification — results query
  4.  Intent classification — dataset query
  5.  Intent classification — limitation query
  6.  Query rewriting — 3 variants generated
  7.  Modality routing — correct collections per intent

  Embedders (model download required on first run)
  8.  TextEmbedder import
  9.  TextEmbedder.embed() — shape (N, 1024)
  10. TextEmbedder.embed_query() — single vector
  11. CodeEmbedder import
  12. detect_code_blocks() — Python code detection
  13. detect_code_blocks() — markdown fenced block
  14. CodeEmbedder.embed() — shape (N, 768)
  15. VisionEmbedder import (skip if open_clip unavailable)

  Fusion & Reranking (no network needed)
  16. RRFFuser import
  17. RRFFuser.fuse() — single arm
  18. RRFFuser.fuse() — multi-arm dedup + score accumulation
  19. RRFFuser citation bonus
  20. ContextCompressor import
  21. ContextCompressor deduplication
  22. ContextCompressor token budget enforcement
  23. ContextCompressor format output

  Qdrant Store (requires Qdrant running)
  24. QdrantStore import + client connect
  25. ensure_collections() — all 5 created
  26. upsert_text_chunks() + search()
  27. upsert_figure() + search()
  28. upsert_table() + search()
  29. upsert_equations() + search()
  30. _table_to_markdown() serialisation

  Hybrid Retriever (requires Qdrant running)
  31. HybridRetriever import
  32. Dense search arm
  33. BM25 build + search
  34. _tokenize() helper

  Citation Graph (requires Neo4j running)
  35. CitationGraph import
  36. create_constraints()
  37. ingest_paper()
  38. find_related()

  Full Pipeline
  39. RAGPipeline import + instantiation
  40. ingest() — full doc ingestion
  41. query() — end-to-end retrieval

Run:
    python tests/test_rag.py                      # all tests
    python tests/test_rag.py --skip-neo4j         # skip Neo4j tests
    python tests/test_rag.py --skip-models        # skip model-download tests
    python tests/test_rag.py --smoke              # infrastructure only (fast)

Exit 0 = all pass.
"""

from __future__ import annotations

import argparse
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PASS  = "\033[92m  [PASS]\033[0m"
FAIL  = "\033[91m  [FAIL]\033[0m"
SKIP  = "\033[93m  [SKIP]\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []
_skip_models = False
_skip_neo4j  = False


def check(name: str, fn, skip_if: bool = False) -> bool:
    if skip_if:
        results.append((name, True, "SKIPPED"))
        print(f"{SKIP} {name}")
        return True
    try:
        detail = fn() or ""
        results.append((name, True, detail))
        print(f"{PASS} {name}" + (f"  — {detail}" if detail else ""))
        return True
    except Exception as exc:
        tb = traceback.format_exc()
        results.append((name, False, str(exc)))
        print(f"{FAIL} {name}")
        print(f"       ↳ {exc}")
        lines = [l for l in tb.splitlines() if "File" in l or "Error" in l]
        for l in lines[-3:]:
            print(f"         {l.strip()}")
        return False


# ── Synthetic helpers ──────────────────────────────────────────────────────────

def _make_fake_doc():
    """Minimal PaperDocument-like object for testing."""
    from src.orchestrator import (PaperDocument, SectionOut,
                                   FigureOut, TableOut, CitationOut)
    pid = str(uuid.uuid4())
    return PaperDocument(
        paper_id=pid,
        title="Test Paper: Attention Is All You Need",
        abstract="We propose the Transformer architecture.",
        sections=[
            SectionOut(
                heading="Introduction",
                body="Attention mechanisms allow modelling of dependencies without regard to distance.",
                chunk_id=str(uuid.uuid4()),
            ),
            SectionOut(
                heading="Methodology",
                body="The model consists of encoder and decoder stacks with self-attention layers.",
                chunk_id=str(uuid.uuid4()),
            ),
        ],
        figures=[
            FigureOut(caption="Figure 1: Transformer architecture", image_path="data/figures/fake.png"),
        ],
        tables=[
            TableOut(
                caption="Table 1: BLEU scores",
                data=[["Model", "EN-DE"], ["Ours", "28.4"], ["Base", "25.1"]],
                chunk_id=str(uuid.uuid4()),
            ),
        ],
        equations=["E = mc^2", "\\text{Attention}(Q,K,V) = \\text{softmax}(QK^T/\\sqrt{d_k})V"],
        citations=[
            CitationOut(ref_id="b0", title="BERT", authors=["Jacob Devlin"], year=2019, doi=None),
        ],
    )


def _make_fake_fused_results(n: int = 5):
    """Generate fake FusedResult objects for testing compression."""
    from src.rag.fusion.rrf_fuser import FusedResult
    results = []
    for i in range(n):
        results.append(FusedResult(
            chunk_id=f"chunk-{i:03d}",
            rrf_score=1.0 / (i + 1),
            payload={
                "body":     f"This is the content of chunk {i}. " * 10,
                "heading":  f"Section {i}",
                "paper_id": "test-paper",
                "modality": "text",
                "page":     i,
                "index":    i,
            },
            sources=["dense", "bm25"] if i % 2 == 0 else ["dense"],
            collection="text_chunks",
        ))
    return results


# ── Intent & Query Analysis tests ─────────────────────────────────────────────

def test_analyzer_import():
    from src.rag.intent.query_analyzer import QueryAnalyzer, QueryAnalysis  # noqa
    return "QueryAnalyzer imported"


def test_intent_methodology():
    from src.rag.intent.query_analyzer import QueryAnalyzer
    a = QueryAnalyzer().analyze("What neural network architecture did they use?")
    assert a.intent == "methodology", f"Expected methodology, got {a.intent}"
    return f"intent={a.intent}, conf={a.confidence}"


def test_intent_results():
    from src.rag.intent.query_analyzer import QueryAnalyzer
    a = QueryAnalyzer().analyze("What BLEU score did the model achieve on the benchmark?")
    assert a.intent == "results", f"Expected results, got {a.intent}"
    return f"intent={a.intent}"


def test_intent_dataset():
    from src.rag.intent.query_analyzer import QueryAnalyzer
    a = QueryAnalyzer().analyze("What training dataset and data splits were used?")
    assert a.intent == "dataset", f"Expected dataset, got {a.intent}"
    return f"intent={a.intent}"


def test_intent_limitation():
    from src.rag.intent.query_analyzer import QueryAnalyzer
    a = QueryAnalyzer().analyze("What are the limitations and future work mentioned?")
    assert a.intent == "limitation", f"Expected limitation, got {a.intent}"
    return f"intent={a.intent}"


def test_query_variants():
    from src.rag.intent.query_analyzer import QueryAnalyzer
    a = QueryAnalyzer().analyze("How does the attention mechanism work?")
    assert len(a.variants) == 3, f"Expected 3 variants, got {len(a.variants)}"
    assert all(isinstance(v, str) and v for v in a.variants)
    return f"3 variants: {a.variants[0][:40]!r}"


def test_modality_routing():
    from src.rag.intent.query_analyzer import QueryAnalyzer
    a_meth    = QueryAnalyzer().analyze("What architecture was used?")
    a_results = QueryAnalyzer().analyze("What benchmark score did they get?")
    assert "text_chunks"  in a_meth.modalities
    assert "table_chunks" in a_results.modalities
    return f"methodology→{a_meth.modalities}, results→{a_results.modalities}"


# ── Embedder tests ─────────────────────────────────────────────────────────────

def test_text_embedder_import():
    from src.rag.embedders.text_embedder import TextEmbedder, TextEmbedding  # noqa
    return "TextEmbedder imported"


def test_text_embedder_shape():
    from src.rag.embedders.text_embedder import TextEmbedder
    emb    = TextEmbedder()
    result = emb.embed(["Attention is all you need.", "Deep learning rocks."])
    assert result.dense.shape == (2, 1024), f"Expected (2, 1024), got {result.dense.shape}"
    assert len(result.sparse) == 2
    return f"dense shape={result.dense.shape}, sparse={len(result.sparse)} vecs"


def test_text_embedder_query():
    from src.rag.embedders.text_embedder import TextEmbedder
    emb    = TextEmbedder()
    result = emb.embed_query("What is the transformer?")
    assert result.dense.shape == (1, 1024)
    return "Single query embedded (1, 1024)"


def test_code_embedder_import():
    from src.rag.embedders.code_embedder import CodeEmbedder, detect_code_blocks  # noqa
    return "CodeEmbedder imported"


def test_detect_code_python():
    from src.rag.embedders.code_embedder import detect_code_blocks
    text = (
        "We implement the forward pass as follows:\n\n"
        "def forward(self, x):\n"
        "    x = self.attention(x)\n"
        "    return self.linear(x)\n\n"
        "This gives us the final output."
    )
    blocks = detect_code_blocks(text)
    assert len(blocks) >= 1, f"Expected code block detected, got {blocks}"
    return f"Detected {len(blocks)} code block(s)"


def test_detect_code_fenced():
    from src.rag.embedders.code_embedder import detect_code_blocks
    text = "Example:\n```python\nfor i in range(10):\n    print(i)\n```\nEnd."
    blocks = detect_code_blocks(text)
    assert len(blocks) == 1
    assert "for i in range" in blocks[0]
    return f"Fenced block extracted: {blocks[0][:40]!r}"


def test_code_embedder_shape():
    from src.rag.embedders.code_embedder import CodeEmbedder
    emb  = CodeEmbedder()
    vecs = emb.embed(["def forward(x): return x", "class Model(nn.Module): pass"])
    assert vecs.shape == (2, 768), f"Expected (2, 768), got {vecs.shape}"
    return f"shape={vecs.shape}"


def test_vision_embedder_import():
    try:
        from src.rag.embedders.vision_embedder import VisionEmbedder  # noqa
        return "VisionEmbedder imported"
    except ImportError as e:
        raise ImportError(f"open_clip or transformers missing: {e}")


# ── Fusion tests ───────────────────────────────────────────────────────────────

def test_rrf_import():
    from src.rag.fusion.rrf_fuser import RRFFuser, FusedResult  # noqa
    return "RRFFuser imported"


def test_rrf_single_arm():
    from src.rag.fusion.rrf_fuser import RRFFuser
    from src.rag.retrievers.hybrid_retriever import RetrievedChunk
    chunks = [
        RetrievedChunk("c1", 0.9, {"body": "a", "modality": "text"}, "dense", "text_chunks"),
        RetrievedChunk("c2", 0.7, {"body": "b", "modality": "text"}, "dense", "text_chunks"),
        RetrievedChunk("c3", 0.5, {"body": "c", "modality": "text"}, "dense", "text_chunks"),
    ]
    fused = RRFFuser(k=60).fuse(chunks)
    assert len(fused) == 3
    assert fused[0].chunk_id == "c1"    # highest rank → highest RRF
    return f"Single arm: {[f.chunk_id for f in fused]}"


def test_rrf_multi_arm():
    from src.rag.fusion.rrf_fuser import RRFFuser
    from src.rag.retrievers.hybrid_retriever import RetrievedChunk
    # c2 appears in both arms → should win via RRF accumulation
    dense = [
        RetrievedChunk("c1", 0.9, {"body": "x"}, "dense", "text_chunks"),
        RetrievedChunk("c2", 0.8, {"body": "y"}, "dense", "text_chunks"),
    ]
    bm25 = [
        RetrievedChunk("c2", 5.0, {"body": "y"}, "bm25",  "text_chunks"),
        RetrievedChunk("c3", 4.0, {"body": "z"}, "bm25",  "text_chunks"),
    ]
    fused = RRFFuser(k=60).fuse(dense + bm25)
    ids   = [f.chunk_id for f in fused]
    assert "c2" in ids
    # c2 appears in 2 arms → should rank higher than c3 (1 arm)
    assert ids.index("c2") < ids.index("c3"), f"c2 should beat c3: {ids}"
    return f"Multi-arm: {ids}, c2 sources={fused[ids.index('c2')].sources}"


def test_rrf_citation_bonus():
    from src.rag.fusion.rrf_fuser import RRFFuser
    from src.rag.retrievers.hybrid_retriever import RetrievedChunk
    chunks = [
        RetrievedChunk("c1", 0.9, {"body": "a", "paper_id": "paper-A"}, "dense", "text_chunks"),
        RetrievedChunk("c2", 0.8, {"body": "b", "paper_id": "paper-B"}, "dense", "text_chunks"),
    ]
    # c2's paper is in the citation graph — should get bonus
    fused = RRFFuser(k=60).fuse(chunks, citation_paper_ids=["paper-B"])
    ids   = [f.chunk_id for f in fused]
    assert ids[0] == "c2", f"c2 should win with citation bonus, got {ids}"
    return f"Citation bonus correctly elevated c2: {ids}"


def test_compressor_import():
    from src.rag.compression.context_compressor import ContextCompressor  # noqa
    return "ContextCompressor imported"


def test_compressor_dedup():
    from src.rag.compression.context_compressor import ContextCompressor, _text_overlap
    from src.rag.fusion.rrf_fuser import FusedResult
    # Two nearly identical chunks
    text = "The transformer model uses multi-head self-attention to encode sequences."
    results = [
        FusedResult("c1", 0.9, {"body": text, "modality": "text", "page": 0, "index": 0}, ["dense"], "text_chunks"),
        FusedResult("c2", 0.8, {"body": text + " This is the same information.", "modality": "text", "page": 0, "index": 1}, ["bm25"], "text_chunks"),
        FusedResult("c3", 0.7, {"body": "Completely different: the dataset was COCO.", "modality": "text", "page": 1, "index": 0}, ["dense"], "text_chunks"),
    ]
    cc  = ContextCompressor(sim_threshold=0.5)
    ctx = cc.compress(results)
    assert ctx.dropped >= 1, f"Expected at least 1 duplicate dropped, got {ctx.dropped}"
    return f"Dedup: dropped={ctx.dropped}, kept={len(ctx.chunks)}"


def test_compressor_budget():
    from src.rag.compression.context_compressor import ContextCompressor
    results = _make_fake_fused_results(20)
    cc      = ContextCompressor(token_budget=100)    # very tight budget
    ctx     = cc.compress(results)
    assert ctx.token_count <= 120, f"Token count exceeded budget: {ctx.token_count}"
    assert len(ctx.chunks) < 20, "Should have truncated some chunks"
    return f"Budget enforced: {ctx.token_count} tokens, {len(ctx.chunks)} chunks kept"


def test_compressor_format():
    from src.rag.compression.context_compressor import ContextCompressor
    results = _make_fake_fused_results(3)
    ctx     = ContextCompressor(token_budget=4096).compress(results, query="test query")
    assert isinstance(ctx.text, str) and len(ctx.text) > 0
    assert "test query" in ctx.text
    assert "Chunk 1" in ctx.text
    return f"Format OK: {len(ctx.text)} chars"


# ── Qdrant store tests ─────────────────────────────────────────────────────────

def test_qdrant_import():
    from src.rag.embedders.qdrant_store import QdrantStore  # noqa
    return "QdrantStore imported"


def test_qdrant_connect():
    from src.rag.embedders.qdrant_store import QdrantStore
    store = QdrantStore()
    colls = store.client.get_collections().collections
    return f"Connected — {len(colls)} existing collections"


def test_qdrant_collections():
    from src.rag.embedders.qdrant_store import QdrantStore, COLLECTIONS
    store = QdrantStore()
    store.ensure_collections()
    names = {c.name for c in store.client.get_collections().collections}
    for expected in COLLECTIONS:
        assert expected in names, f"Collection {expected!r} missing"
    return f"All {len(COLLECTIONS)} collections present: {sorted(names)}"


def test_qdrant_text_upsert_search():
    import numpy as np
    from src.rag.embedders.qdrant_store import QdrantStore
    from src.chunkers.semantic_chunker import Chunk

    store  = QdrantStore()
    pid    = f"test-{uuid.uuid4().hex[:8]}"
    chunks = [Chunk(chunk_id=str(uuid.uuid4()), heading="Intro", body="test body text", index=0)]
    vecs   = np.random.rand(1, 1024).astype("float32")
    store.upsert_text_chunks(chunks, vecs, paper_id=pid)

    hits = store.search("text_chunks", vecs[0], top_k=1)
    assert len(hits) >= 1
    assert hits[0]["payload"]["paper_id"] == pid
    return f"Upsert + search OK, score={hits[0]['score']:.4f}"


def test_qdrant_figure_upsert():
    import numpy as np
    from src.rag.embedders.qdrant_store import QdrantStore
    from src.orchestrator import FigureOut

    store = QdrantStore()
    pid   = f"test-{uuid.uuid4().hex[:8]}"
    fig   = FigureOut(caption="Test figure caption", image_path="data/figures/test.png")
    vec   = np.random.rand(768).astype("float32")
    store.upsert_figure(fig, vec, paper_id=pid, fig_index=0)
    hits  = store.search("figure_chunks", vec, top_k=1)
    assert len(hits) >= 1
    return f"Figure upsert+search OK"


def test_qdrant_table_upsert():
    import numpy as np
    from src.rag.embedders.qdrant_store import QdrantStore
    from src.extractors.table_extractor import TableRecord

    store = QdrantStore()
    pid   = f"test-{uuid.uuid4().hex[:8]}"
    tbl   = TableRecord(caption="Test table", data=[["A","B"],["1","2"]], page=1, accuracy=90.0)
    vec   = np.random.rand(1024).astype("float32")
    store.upsert_table(tbl, vec, paper_id=pid, tbl_index=0)
    hits  = store.search("table_chunks", vec, top_k=1)
    assert len(hits) >= 1
    return f"Table upsert+search OK"


def test_qdrant_equation_upsert():
    import numpy as np
    from src.rag.embedders.qdrant_store import QdrantStore

    store = QdrantStore()
    pid   = f"test-{uuid.uuid4().hex[:8]}"
    eqs   = ["E = mc^2", "F = ma"]
    vecs  = np.random.rand(2, 1024).astype("float32")
    n     = store.upsert_equations(eqs, vecs, paper_id=pid)
    assert n == 2
    hits  = store.search("equation_chunks", vecs[0], top_k=1)
    assert len(hits) >= 1
    return f"Equations upsert+search OK ({n} equations)"


def test_table_to_markdown():
    from src.rag.embedders.qdrant_store import _table_to_markdown
    data = [["Model", "Acc"], ["Ours", "94.2"], ["Base", "88.1"]]
    md   = _table_to_markdown(data)
    assert "| Model | Acc |" in md
    assert "| Ours | 94.2 |" in md
    assert "---" in md
    return f"Markdown: {md[:60]!r}"


# ── Hybrid retriever tests ─────────────────────────────────────────────────────

def test_hybrid_import():
    from src.rag.retrievers.hybrid_retriever import HybridRetriever  # noqa
    return "HybridRetriever imported"


def test_hybrid_dense_arm():
    import numpy as np
    from src.rag.embedders.qdrant_store  import QdrantStore
    from src.rag.embedders.text_embedder import TextEmbedder
    from src.rag.retrievers.hybrid_retriever import HybridRetriever

    store     = QdrantStore()
    text_emb  = TextEmbedder()
    retriever = HybridRetriever(store, text_emb)
    qvec      = np.random.rand(1024).astype("float32")
    results   = retriever._dense_search(qvec, ["text_chunks"], top_k=5, metadata_filter=None)
    assert isinstance(results, list)
    return f"Dense arm returned {len(results)} results"


def test_bm25_build_search():
    import numpy as np
    from src.rag.embedders.qdrant_store  import QdrantStore
    from src.rag.embedders.text_embedder import TextEmbedder
    from src.rag.retrievers.hybrid_retriever import HybridRetriever

    store     = QdrantStore()
    text_emb  = TextEmbedder()
    retriever = HybridRetriever(store, text_emb)
    retriever.refresh_bm25()
    results   = retriever._bm25_search("attention transformer", top_k=5)
    assert isinstance(results, list)
    return f"BM25 returned {len(results)} results"


def test_tokenize():
    from src.rag.retrievers.hybrid_retriever import _tokenize
    tokens = _tokenize("The Transformer uses self-attention (multi-head).")
    assert "transformer" in tokens
    assert "multi" in tokens
    assert "head" in tokens
    return f"Tokens: {tokens}"


# ── Citation graph tests ───────────────────────────────────────────────────────

def test_citation_import():
    from src.rag.retrievers.citation_graph import CitationGraph  # noqa
    return "CitationGraph imported"


def test_citation_constraints():
    from src.rag.retrievers.citation_graph import CitationGraph
    graph = CitationGraph()
    graph.create_constraints()
    return "Constraints created"


def test_citation_ingest():
    from src.rag.retrievers.citation_graph import CitationGraph
    doc   = _make_fake_doc()
    graph = CitationGraph()
    graph.ingest_paper(doc)
    return f"Paper {doc.paper_id[:8]} ingested with {len(doc.citations)} citations"


def test_citation_find_related():
    from src.rag.retrievers.citation_graph import CitationGraph
    doc   = _make_fake_doc()
    graph = CitationGraph()
    graph.ingest_paper(doc)
    related = graph.find_related([doc.paper_id], hops=2)
    assert isinstance(related, list)
    return f"Found {len(related)} related papers"


# ── Full pipeline tests ────────────────────────────────────────────────────────

def test_rag_pipeline_import():
    from src.rag.rag_pipeline import RAGPipeline, RAGResult  # noqa
    return "RAGPipeline imported"


def test_rag_ingest():
    from src.rag.rag_pipeline import RAGPipeline
    doc      = _make_fake_doc()
    pipeline = RAGPipeline(use_citation_graph=not _skip_neo4j, use_vision=False)
    summary  = pipeline.ingest(doc)
    assert isinstance(summary, dict)
    assert "text_chunks" in summary
    return f"Ingestion summary: {summary}"


def test_rag_query():
    from src.rag.rag_pipeline import RAGPipeline, RAGResult
    # Ingest first so there's something to retrieve
    doc      = _make_fake_doc()
    pipeline = RAGPipeline(use_citation_graph=not _skip_neo4j, use_vision=False)
    pipeline.ingest(doc)

    result = pipeline.query("What attention mechanism was used?")
    assert isinstance(result, RAGResult)
    assert result.analysis.intent in {"methodology", "definition", "general"}
    assert isinstance(result.context.text, str)
    assert result.context.token_count >= 0
    return (
        f"intent={result.analysis.intent}  "
        f"reranked={result.reranked_count}  "
        f"tokens={result.context.token_count}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global _skip_models, _skip_neo4j

    parser = argparse.ArgumentParser(description="Stage 6.3 RAG engine tests")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip tests that download model weights")
    parser.add_argument("--skip-neo4j",  action="store_true",
                        help="Skip Neo4j citation graph tests")
    parser.add_argument("--smoke",       action="store_true",
                        help="Infrastructure-only smoke test (fast)")
    args = parser.parse_args()

    _skip_models = args.skip_models or args.smoke
    _skip_neo4j  = args.skip_neo4j  or args.smoke

    print(f"\n{BOLD}╔══════════════════════════════════════════════════╗")
    print(      "║   Document Intelligence Engine — Stage 6.3      ║")
    print(      "║          Multimodal RAG Engine Tests             ║")
    print(      f"╚══════════════════════════════════════════════════╝{RESET}")
    if _skip_models:
        print(f"  {SKIP} --skip-models active: embedding tests skipped")
    if _skip_neo4j:
        print(f"  {SKIP} --skip-neo4j active: citation graph tests skipped")
    print()

    print(f"{BOLD}── Intent & Query Analysis ────────────────────────{RESET}")
    check("QueryAnalyzer import",           test_analyzer_import)
    check("Intent: methodology",            test_intent_methodology)
    check("Intent: results",                test_intent_results)
    check("Intent: dataset",                test_intent_dataset)
    check("Intent: limitation",             test_intent_limitation)
    check("Query rewriting (3 variants)",   test_query_variants)
    check("Modality routing",               test_modality_routing)

    print(f"\n{BOLD}── Embedders ──────────────────────────────────────{RESET}")
    check("TextEmbedder import",            test_text_embedder_import)
    check("TextEmbedder shape (N, 1024)",   test_text_embedder_shape,   skip_if=_skip_models)
    check("TextEmbedder query (1, 1024)",   test_text_embedder_query,   skip_if=_skip_models)
    check("CodeEmbedder import",            test_code_embedder_import)
    check("detect_code_blocks() Python",    test_detect_code_python)
    check("detect_code_blocks() fenced",    test_detect_code_fenced)
    check("CodeEmbedder shape (N, 768)",    test_code_embedder_shape,   skip_if=_skip_models)
    check("VisionEmbedder import",          test_vision_embedder_import)

    print(f"\n{BOLD}── RRF Fusion & Compression ───────────────────────{RESET}")
    check("RRFFuser import",                test_rrf_import)
    check("RRF single arm",                 test_rrf_single_arm)
    check("RRF multi-arm dedup",            test_rrf_multi_arm)
    check("RRF citation bonus",             test_rrf_citation_bonus)
    check("ContextCompressor import",       test_compressor_import)
    check("Compressor deduplication",       test_compressor_dedup)
    check("Compressor token budget",        test_compressor_budget)
    check("Compressor format output",       test_compressor_format)

    print(f"\n{BOLD}── Qdrant Store ───────────────────────────────────{RESET}")
    check("QdrantStore import",             test_qdrant_import)
    check("Qdrant connect",                 test_qdrant_connect)
    check("ensure_collections() (5/5)",     test_qdrant_collections)
    check("text_chunks upsert + search",    test_qdrant_text_upsert_search)
    check("figure_chunks upsert + search",  test_qdrant_figure_upsert)
    check("table_chunks upsert + search",   test_qdrant_table_upsert)
    check("equation_chunks upsert",         test_qdrant_equation_upsert)
    check("_table_to_markdown()",           test_table_to_markdown)

    print(f"\n{BOLD}── Hybrid Retriever ───────────────────────────────{RESET}")
    check("HybridRetriever import",         test_hybrid_import)
    check("Dense search arm",               test_hybrid_dense_arm)
    check("BM25 build + search",            test_bm25_build_search)
    check("_tokenize() helper",             test_tokenize)

    print(f"\n{BOLD}── Citation Graph (Neo4j) ─────────────────────────{RESET}")
    check("CitationGraph import",           test_citation_import)
    check("create_constraints()",           test_citation_constraints,  skip_if=_skip_neo4j)
    check("ingest_paper()",                 test_citation_ingest,       skip_if=_skip_neo4j)
    check("find_related()",                 test_citation_find_related, skip_if=_skip_neo4j)

    print(f"\n{BOLD}── Full RAG Pipeline ──────────────────────────────{RESET}")
    check("RAGPipeline import",             test_rag_pipeline_import)
    check("ingest() full document",         test_rag_ingest,            skip_if=_skip_models)
    check("query() end-to-end",             test_rag_query,             skip_if=_skip_models)

    # ── Summary ────────────────────────────────────────────────────────────────
    total   = len(results)
    passed  = sum(1 for _, ok, d in results if ok and d != "SKIPPED")
    skipped = sum(1 for _, ok, d in results if d == "SKIPPED")
    failed  = total - passed - skipped

    print(f"\n{BOLD}── Results ────────────────────────────────────────{RESET}")
    print(f"  {passed} passed  |  {skipped} skipped  |  {failed} failed  (of {total})")

    if failed:
        print(f"\n{BOLD}  Failing checks:{RESET}")
        for name, ok, detail in results:
            if not ok:
                print(f"    • {name}: {detail}")
        sys.exit(1)
    else:
        print(f"\n  {BOLD}✓ All checks passed — Stage 6.3 RAG Engine is solid.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
