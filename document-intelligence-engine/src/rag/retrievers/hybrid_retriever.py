"""
src/rag/retrievers/hybrid_retriever.py
========================================
Hybrid retrieval engine — runs Dense + BM25 + Metadata in parallel.

Four retrieval arms (per query):
  1. Dense vector search    — Qdrant cosine similarity per modality
  2. BM25 sparse search     — in-memory keyword index (rank-bm25)
  3. Metadata filter search — year, paper_id, heading filter via Qdrant
  4. Figure/table retrieval — vision embeddings (cross-modal text→image)

Each arm runs independently; results are combined by the RRF fuser.

BM25 index:
  Built lazily on first query from the text payloads stored in Qdrant.
  Refreshed when refresh_bm25() is called (e.g. after new ingestion).

Usage:
    from src.rag.retrievers.hybrid_retriever import HybridRetriever
    from src.rag.embedders.text_embedder import TextEmbedder
    from src.rag.embedders.qdrant_store import QdrantStore

    store     = QdrantStore()
    text_emb  = TextEmbedder()
    retriever = HybridRetriever(store, text_emb)

    results = retriever.retrieve(
        query="What attention mechanism was used?",
        query_vec=text_emb.embed_query("...").dense[0],
        top_k=20,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class RetrievedChunk:
    chunk_id:   str
    score:      float
    payload:    dict
    source:     str    # "dense", "bm25", "metadata", "vision"
    collection: str


class HybridRetriever:
    """
    Runs all four retrieval arms and aggregates raw results.
    Final fusion (RRF) happens in the RRFFuser module.

    Parameters
    ----------
    store : QdrantStore
    text_embedder : TextEmbedder
    vision_embedder : VisionEmbedder, optional
        If provided, enables text→figure cross-modal retrieval.
    bm25_top_k : int
        How many BM25 results to return per arm.
    """

    def __init__(
        self,
        store,
        text_embedder,
        vision_embedder=None,
        bm25_top_k: int = 20,
    ) -> None:
        self.store           = store
        self.text_emb        = text_embedder
        self.vision_emb      = vision_embedder
        self.bm25_top_k      = bm25_top_k
        self._bm25_index     = None
        self._bm25_docs: list[dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        query_vec: np.ndarray,          # (1024,) dense vector from TextEmbedder
        collections: list[str],         # from QueryAnalysis.modalities
        top_k: int = 20,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Run all active retrieval arms and return combined raw results.

        Parameters
        ----------
        query : str
            Original user query (used for BM25).
        query_vec : np.ndarray
            Dense query embedding.
        collections : list[str]
            Which Qdrant collections to search (from intent analysis).
        top_k : int
            How many results to return per arm per collection.
        metadata_filter : dict, optional
            e.g. {"paper_id": "abc"} to restrict to one paper.
        """
        all_results: list[RetrievedChunk] = []

        # Arm 1: Dense vector search
        all_results.extend(
            self._dense_search(query_vec, collections, top_k, metadata_filter)
        )

        # Arm 2: BM25 keyword search
        all_results.extend(
            self._bm25_search(query, top_k)
        )

        # Arm 3: Metadata filter (if filter given — find by exact paper/year)
        if metadata_filter:
            all_results.extend(
                self._metadata_search(query_vec, metadata_filter, top_k)
            )

        # Arm 4: Cross-modal figure retrieval (text → figure)
        if self.vision_emb and "figure_chunks" in collections:
            all_results.extend(
                self._vision_search(query, top_k)
            )

        logger.debug(
            f"HybridRetriever: {len(all_results)} raw candidates "
            f"({_count_by_source(all_results)})"
        )
        return all_results

    def refresh_bm25(self) -> None:
        """Rebuild BM25 index from current Qdrant text_chunks content."""
        logger.info("Refreshing BM25 index …")
        self._bm25_index = None
        self._bm25_docs  = []
        self._build_bm25()

    # ── Arm 1: Dense search ────────────────────────────────────────────────────

    def _dense_search(
        self,
        query_vec: np.ndarray,
        collections: list[str],
        top_k: int,
        metadata_filter: Optional[dict],
    ) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        for coll in collections:
            try:
                hits = self.store.search(
                    collection=coll,
                    query_vec=query_vec,
                    top_k=top_k,
                    filter_payload=metadata_filter,
                )
                for h in hits:
                    results.append(RetrievedChunk(
                        chunk_id=h["payload"].get("chunk_id", str(h["id"])),
                        score=h["score"],
                        payload=h["payload"],
                        source="dense",
                        collection=coll,
                    ))
            except Exception as exc:
                logger.warning(f"Dense search on {coll} failed: {exc}")
        return results

    # ── Arm 2: BM25 search ─────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25_index is None:
            self._build_bm25()
        if not self._bm25_docs:
            return []

        try:
            from rank_bm25 import BM25Okapi
            tokens = _tokenize(query)
            scores = self._bm25_index.get_scores(tokens)

            # Get top-k indices
            top_idx = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]

            results: list[RetrievedChunk] = []
            for idx in top_idx:
                if scores[idx] <= 0:
                    continue
                doc = self._bm25_docs[idx]
                results.append(RetrievedChunk(
                    chunk_id=doc.get("chunk_id", str(idx)),
                    score=float(scores[idx]),
                    payload=doc,
                    source="bm25",
                    collection="text_chunks",
                ))
            return results
        except Exception as exc:
            logger.warning(f"BM25 search failed: {exc}")
            return []

    def _build_bm25(self) -> None:
        """Load all text payloads from Qdrant and build BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
            from qdrant_client.models import ScrollRequest

            # Scroll through all text_chunks
            all_docs: list[dict] = []
            offset = None
            while True:
                result, next_offset = self.store.client.scroll(
                    collection_name="text_chunks",
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in result:
                    all_docs.append(point.payload)
                if next_offset is None:
                    break
                offset = next_offset

            if not all_docs:
                logger.debug("BM25: no documents in text_chunks yet")
                return

            self._bm25_docs  = all_docs
            corpus           = [_tokenize(d.get("body", "")) for d in all_docs]
            self._bm25_index = BM25Okapi(corpus)
            logger.info(f"BM25 index built: {len(all_docs)} documents")

        except Exception as exc:
            logger.warning(f"BM25 index build failed: {exc}")

    # ── Arm 3: Metadata filter search ─────────────────────────────────────────

    def _metadata_search(
        self,
        query_vec: np.ndarray,
        metadata_filter: dict,
        top_k: int,
    ) -> list[RetrievedChunk]:
        try:
            hits = self.store.search(
                collection="text_chunks",
                query_vec=query_vec,
                top_k=top_k,
                filter_payload=metadata_filter,
            )
            return [
                RetrievedChunk(
                    chunk_id=h["payload"].get("chunk_id", str(h["id"])),
                    score=h["score"],
                    payload=h["payload"],
                    source="metadata",
                    collection="text_chunks",
                )
                for h in hits
            ]
        except Exception as exc:
            logger.warning(f"Metadata search failed: {exc}")
            return []

    # ── Arm 4: Cross-modal vision search ──────────────────────────────────────

    def _vision_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        try:
            query_vec = self.vision_emb.embed_text(query)
            hits      = self.store.search(
                collection="figure_chunks",
                query_vec=query_vec,
                top_k=top_k,
            )
            return [
                RetrievedChunk(
                    chunk_id=h["payload"].get("chunk_id", str(h["id"])),
                    score=h["score"],
                    payload=h["payload"],
                    source="vision",
                    collection="figure_chunks",
                )
                for h in hits
            ]
        except Exception as exc:
            logger.warning(f"Vision search failed: {exc}")
            return []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _count_by_source(results: list[RetrievedChunk]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.source] = counts.get(r.source, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in counts.items())
