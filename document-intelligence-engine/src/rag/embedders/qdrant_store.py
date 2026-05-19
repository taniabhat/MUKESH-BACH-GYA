"""
src/rag/embedders/qdrant_store.py
===================================
Qdrant collection manager — creates collections and indexes documents.

Collections (one per modality):
  text_chunks     — BGE-M3 dense 1024-dim  + sparse
  figure_chunks   — SigLIP/CLIP 768-dim
  table_chunks    — BGE-M3 dense 1024-dim  (tables serialised to markdown)
  code_chunks     — CodeBERT 768-dim
  equation_chunks — BGE-M3 dense 1024-dim  (LaTeX text)

Each point stored in Qdrant has:
  id      — integer (auto-incremented)
  vector  — dense float vector
  payload — {paper_id, chunk_id, text/caption/code, page, modality, ...}

Usage:
    from src.rag.embedders.qdrant_store import QdrantStore
    store = QdrantStore()
    store.ensure_collections()
    store.upsert_text_chunks(chunks, embeddings, paper_id="abc")
    store.upsert_figure(fig_record, vec, paper_id="abc")
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger

QDRANT_URL  = os.getenv("QDRANT_URL",  "http://localhost:6333")
QDRANT_KEY  = os.getenv("QDRANT_API_KEY", None)   # None = no auth (local)

# Collection specs: name → (vector_dim, distance)
COLLECTIONS: dict[str, tuple[int, str]] = {
    "text_chunks":     (1024, "Cosine"),
    "figure_chunks":   (768,  "Cosine"),
    "table_chunks":    (1024, "Cosine"),
    "code_chunks":     (768,  "Cosine"),
    "equation_chunks": (1024, "Cosine"),
}

_client = None


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        logger.info(f"Connecting to Qdrant at {QDRANT_URL} …")
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
        logger.info("Qdrant client connected.")
    return _client


def _stable_id(paper_id: str, chunk_id: str) -> int:
    """Convert (paper_id, chunk_id) to a stable integer point ID."""
    h = hashlib.sha256(f"{paper_id}::{chunk_id}".encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 53)   # JS-safe integer range


def _table_to_markdown(data: list[list[str]]) -> str:
    """Serialise a 2D table to markdown for text embedding."""
    if not data:
        return ""
    header = "| " + " | ".join(str(c) for c in data[0]) + " |"
    sep    = "| " + " | ".join("---" for _ in data[0]) + " |"
    rows   = "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |"
        for row in data[1:]
    )
    return "\n".join([header, sep, rows])


class QdrantStore:
    """
    High-level interface for all Qdrant collection operations.
    """

    def __init__(self) -> None:
        self._client = None   # lazily initialised

    @property
    def client(self):
        if self._client is None:
            self._client = _get_client()
        return self._client

    # ── Collection setup ───────────────────────────────────────────────────────

    def ensure_collections(self, recreate: bool = False) -> None:
        """
        Create all five collections if they don't exist.
        If recreate=True, drop and recreate (useful for dev resets).
        """
        from qdrant_client.models import Distance, VectorParams

        distance_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT}

        existing = {c.name for c in self.client.get_collections().collections}

        for name, (dim, dist) in COLLECTIONS.items():
            if recreate and name in existing:
                logger.warning(f"Dropping collection: {name}")
                self.client.delete_collection(name)
                existing.discard(name)

            if name not in existing:
                logger.info(f"Creating collection: {name} (dim={dim}, dist={dist})")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=distance_map[dist],
                    ),
                )
            else:
                logger.debug(f"Collection exists: {name}")

        logger.info("All Qdrant collections ready.")

    # ── Upsert helpers ─────────────────────────────────────────────────────────

    def upsert_text_chunks(
        self,
        chunks: list,        # list of Chunk from semantic_chunker
        embeddings: np.ndarray,
        paper_id: str,
    ) -> int:
        """Upsert text chunks into text_chunks collection."""
        from qdrant_client.models import PointStruct
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            pid = _stable_id(paper_id, chunk.chunk_id)
            points.append(PointStruct(
                id=pid,
                vector=vec.tolist(),
                payload={
                    "paper_id":  paper_id,
                    "chunk_id":  chunk.chunk_id,
                    "heading":   chunk.heading,
                    "body":      chunk.body,
                    "index":     chunk.index,
                    "modality":  "text",
                },
            ))
        self.client.upsert("text_chunks", points=points)
        logger.info(f"Upserted {len(points)} text chunks for paper {paper_id[:8]}")
        return len(points)

    def upsert_figure(
        self,
        fig,               # FigureRecord
        vec: np.ndarray,   # (768,)
        paper_id: str,
        fig_index: int,
    ) -> None:
        """Upsert a single figure into figure_chunks collection."""
        from qdrant_client.models import PointStruct
        chunk_id = f"{paper_id}::fig::{fig_index}"
        pid      = _stable_id(paper_id, chunk_id)
        self.client.upsert("figure_chunks", points=[PointStruct(
            id=pid,
            vector=vec.tolist(),
            payload={
                "paper_id":   paper_id,
                "chunk_id":   chunk_id,
                "caption":    fig.caption,
                "image_path": fig.image_path,
                #"page":       #fig.page,
                "modality":   "figure",
            },
        )])

    def upsert_table(
        self,
        tbl,               # TableRecord
        vec: np.ndarray,   # (1024,) — BGE-M3 on markdown
        paper_id: str,
        tbl_index: int,
    ) -> None:
        """Upsert a single table into table_chunks collection."""
        from qdrant_client.models import PointStruct
        chunk_id = f"{paper_id}::tbl::{tbl_index}"
        pid      = _stable_id(paper_id, chunk_id)
        md_text  = _table_to_markdown(tbl.data)
        self.client.upsert("table_chunks", points=[PointStruct(
            id=pid,
            vector=vec.tolist(),
            payload={
                "paper_id":  paper_id,
                "chunk_id":  chunk_id,
                "caption":   tbl.caption,
                "data":      tbl.data,
                "markdown":  md_text,
                "page":      tbl.page,
                "modality":  "table",
            },
        )])

    def upsert_code(
        self,
        snippet: str,
        vec: np.ndarray,   # (768,)
        paper_id: str,
        code_index: int,
        context_heading: str = "",
    ) -> None:
        """Upsert a code snippet into code_chunks collection."""
        from qdrant_client.models import PointStruct
        chunk_id = f"{paper_id}::code::{code_index}"
        pid      = _stable_id(paper_id, chunk_id)
        self.client.upsert("code_chunks", points=[PointStruct(
            id=pid,
            vector=vec.tolist(),
            payload={
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "code":     snippet,
                "heading":  context_heading,
                "modality": "code",
            },
        )])

    def upsert_equations(
        self,
        equations: list[str],
        embeddings: np.ndarray,
        paper_id: str,
    ) -> int:
        """Upsert equations into equation_chunks collection."""
        from qdrant_client.models import PointStruct
        points = []
        for i, (eq, vec) in enumerate(zip(equations, embeddings)):
            chunk_id = f"{paper_id}::eq::{i}"
            pid      = _stable_id(paper_id, chunk_id)
            points.append(PointStruct(
                id=pid,
                vector=vec.tolist(),
                payload={
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "equation": eq,
                    "modality": "equation",
                },
            ))
        if points:
            self.client.upsert("equation_chunks", points=points)
        return len(points)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        query_vec: np.ndarray,
        top_k: int = 10,
        filter_payload: Optional[dict] = None,
    ) -> list[dict]:
        """
        Dense vector search on any collection.
        Returns list of {id, score, payload} dicts.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filter_payload:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_payload.items()
            ]
            qdrant_filter = Filter(must=conditions)

        hits = self.client.search(
            collection_name=collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {"id": h.id, "score": h.score, "payload": h.payload}
            for h in hits
        ]

    def collection_count(self, name: str) -> int:
        """Return number of points in a collection."""
        return self.client.count(name).count
