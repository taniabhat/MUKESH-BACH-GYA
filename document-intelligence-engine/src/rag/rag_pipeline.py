"""
src/rag/rag_pipeline.py
=========================
Multimodal RAG Engine — top-level pipeline.

Two entry points:

  1. ingest(doc: PaperDocument)
     Takes a parsed document from 6.2 and:
       - Embeds all text chunks with BGE-M3
       - Embeds all figures with SigLIP/CLIP
       - Embeds all tables (as markdown) with BGE-M3
       - Detects and embeds code blocks with CodeBERT
       - Embeds all equations with BGE-M3
       - Upserts everything into the correct Qdrant collections
       - Ingests citation edges into Neo4j

  2. query(q: str) → RAGResult
     Takes a user question and runs the full retrieval pipeline:
       Intent → Rewrite → Hybrid Retrieve → RRF Fuse →
       Citation Expand → BGE Rerank → Compress → Return

Usage:
    from pathlib import Path
    from src.orchestrator import Orchestrator
    from src.rag.rag_pipeline import RAGPipeline

    # --- Ingest ---
    doc      = Orchestrator().process(Path("data/input/paper.pdf"))
    pipeline = RAGPipeline()
    pipeline.ingest(doc)

    # --- Query ---
    result = pipeline.query("What attention mechanism did they use?")
    print(result.context.text)
    print(result.intent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from src.rag.compression.context_compressor import CompressedContext, ContextCompressor
from src.rag.embedders.code_embedder        import CodeEmbedder, detect_code_blocks
from src.rag.embedders.qdrant_store         import QdrantStore, _table_to_markdown
from src.rag.embedders.text_embedder        import TextEmbedder
from src.rag.embedders.vision_embedder      import VisionEmbedder
from src.rag.fusion.rrf_fuser               import RRFFuser
from src.rag.intent.query_analyzer          import QueryAnalyzer, QueryAnalysis
from src.rag.rerankers.bge_reranker         import BGEReranker
from src.rag.retrievers.citation_graph      import CitationGraph
from src.rag.retrievers.hybrid_retriever    import HybridRetriever


@dataclass
class RAGResult:
    query:        str
    analysis:     QueryAnalysis
    context:      CompressedContext
    cited_papers: list[dict]          = field(default_factory=list)
    reranked_count: int               = 0


class RAGPipeline:
    """
    Full Multimodal RAG pipeline.

    Parameters
    ----------
    top_retrieve : int
        How many candidates to retrieve per arm (before reranking).
    top_rerank : int
        Final top-k after cross-encoder reranking.
    token_budget : int
        Max tokens in compressed context.
    use_citation_graph : bool
        Whether to query Neo4j for citation expansion.
        Set False if Neo4j isn't running.
    use_vision : bool
        Whether to use SigLIP for figure retrieval.
        Set False for CPU-only environments.
    """

    def __init__(
        self,
        top_retrieve:        int  = 20,
        top_rerank:          int  = 10,
        token_budget:        int  = 8192,
        use_citation_graph:  bool = True,
        use_vision:          bool = True,
    ) -> None:
        self.top_retrieve  = top_retrieve
        self.top_rerank    = top_rerank

        logger.info("Initialising RAG Pipeline components …")

        # Core components
        self.store     = QdrantStore()
        self.text_emb  = TextEmbedder()
        self.code_emb  = CodeEmbedder()
        self.vision_emb: Optional[VisionEmbedder] = None

        if use_vision:
            try:
                self.vision_emb = VisionEmbedder()
            except Exception as exc:
                logger.warning(f"VisionEmbedder failed to load: {exc}")

        self.citation_graph: Optional[CitationGraph] = None
        if use_citation_graph:
            try:
                self.citation_graph = CitationGraph()
            except Exception as exc:
                logger.warning(f"CitationGraph failed to connect: {exc}")

        self.analyzer   = QueryAnalyzer()
        self.retriever  = HybridRetriever(
            store=self.store,
            text_embedder=self.text_emb,
            vision_embedder=self.vision_emb,
        )
        self.fuser      = RRFFuser(k=60, top_n=50)
        self.reranker   = BGEReranker()
        self.compressor = ContextCompressor(token_budget=token_budget)

        # Ensure Qdrant collections exist
        self.store.ensure_collections()
        logger.info("RAG Pipeline ready.")

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, doc) -> dict:
        """
        Embed and store a PaperDocument across all modalities.

        Returns a summary dict of how many items were indexed per modality.
        """
        logger.info(f"Ingesting paper: {doc.paper_id[:8]} — {doc.title[:60]!r}")
        summary: dict[str, int] = {}

        # ── Text chunks ────────────────────────────────────────────────────────
        if doc.sections:
            texts = [s.body for s in doc.sections]
            emb   = self.text_emb.embed(texts)
            n     = self.store.upsert_text_chunks(doc.sections, emb.dense, doc.paper_id)
            summary["text_chunks"] = n

        # ── Figures ────────────────────────────────────────────────────────────
        fig_count = 0
        if doc.figures and self.vision_emb:
            for i, fig in enumerate(doc.figures):
                try:
                    from pathlib import Path
                    vec = self.vision_emb.embed_image(Path(fig.image_path))
                    self.store.upsert_figure(fig, vec, doc.paper_id, i)
                    fig_count += 1
                except Exception as exc:
                    logger.warning(f"Figure {i} embed failed: {exc}")
        elif doc.figures and not self.vision_emb:
            # Fallback: embed caption text with BGE-M3
            captions = [f.caption or "figure" for f in doc.figures]
            vecs     = self.text_emb.embed(captions).dense
            for i, (fig, vec) in enumerate(zip(doc.figures, vecs)):
                # Resize to 768 by truncating / zero-padding
                v768 = np.zeros(768, dtype=np.float32)
                v768[:min(768, len(vec))] = vec[:768]
                self.store.upsert_figure(fig, v768, doc.paper_id, i)
                fig_count += 1
        summary["figures"] = fig_count

        # ── Tables ─────────────────────────────────────────────────────────────
        tbl_count = 0
        if doc.tables:
            md_texts = [
                (t.caption + "\n" if t.caption else "") +
                _table_to_markdown(t.data)
                for t in doc.tables
            ]
            vecs = self.text_emb.embed(md_texts).dense
            for i, (tbl, vec) in enumerate(zip(doc.tables, vecs)):
                self.store.upsert_table(tbl, vec, doc.paper_id, i)
                tbl_count += 1
        summary["tables"] = tbl_count

        # ── Code blocks ────────────────────────────────────────────────────────
        code_count = 0
        for sec in doc.sections:
            snippets = detect_code_blocks(sec.body)
            if snippets:
                vecs = self.code_emb.embed(snippets)
                for j, (snippet, vec) in enumerate(zip(snippets, vecs)):
                    self.store.upsert_code(
                        snippet, vec, doc.paper_id,
                        code_index=code_count,
                        context_heading=sec.heading,
                    )
                    code_count += 1
        summary["code_chunks"] = code_count

        # ── Equations ──────────────────────────────────────────────────────────
        eq_count = 0
        if doc.equations:
            vecs = self.text_emb.embed(doc.equations).dense
            eq_count = self.store.upsert_equations(doc.equations, vecs, doc.paper_id)
        summary["equations"] = eq_count

        # ── Citation graph ─────────────────────────────────────────────────────
        if self.citation_graph:
            try:
                self.citation_graph.create_constraints()
                self.citation_graph.ingest_paper(doc)
                summary["citations_indexed"] = len(doc.citations)
            except Exception as exc:
                logger.warning(f"Citation graph ingest failed: {exc}")

        logger.info(f"Ingestion complete: {summary}")
        return summary

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        q: str,
        metadata_filter: Optional[dict] = None,
        top_rerank: Optional[int] = None,
    ) -> RAGResult:
        """
        Run the full RAG retrieval pipeline for a user query.

        Parameters
        ----------
        q : str
            User's natural language question.
        metadata_filter : dict, optional
            e.g. {"paper_id": "abc"} to restrict to one paper.
        top_rerank : int, optional
            Override the default top_rerank setting.
        """
        top_k = top_rerank or self.top_rerank
        logger.info(f"RAG query: {q!r}")

        # Step 1: Intent analysis + query rewriting
        analysis = self.analyzer.analyze(q)
        logger.info(f"Intent: {analysis.intent} (conf={analysis.confidence})")

        # Step 2: Embed all query variants
        all_raw = []
        for variant in [q] + analysis.variants:
            emb = self.text_emb.embed_query(variant)
            qvec = emb.dense[0]
            raw = self.retriever.retrieve(
                query=variant,
                query_vec=qvec,
                collections=analysis.modalities,
                top_k=self.top_retrieve,
                metadata_filter=metadata_filter,
            )
            all_raw.extend(raw)

        # Step 3: Citation graph expansion
        cited_papers: list[dict] = []
        if self.citation_graph:
            seed_ids = list({
                r.payload.get("paper_id", "")
                for r in all_raw
                if r.payload.get("paper_id")
            })
            cited_papers = self.citation_graph.find_related(seed_ids, hops=2)

        # Step 4: RRF fusion
        citation_paper_ids = [p["paper_id"] for p in cited_papers if p.get("paper_id")]
        fused = self.fuser.fuse(all_raw, citation_paper_ids=citation_paper_ids)

        # Step 5: Cross-encoder reranking (top 50 → top k)
        reranked = self.reranker.rerank(q, fused, top_k=top_k)

        # Step 6: Context compression
        context = self.compressor.compress(reranked, query=q)

        return RAGResult(
            query=q,
            analysis=analysis,
            context=context,
            cited_papers=cited_papers,
            reranked_count=len(reranked),
        )
