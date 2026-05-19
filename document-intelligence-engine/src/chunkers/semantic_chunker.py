"""
src/chunkers/semantic_chunker.py
=================================
Semantic chunking of parsed document sections.

What this module does:
  Takes the flat list of GrobidSection / OcrSection objects produced
  by Stage 2 and splits them into semantically coherent, size-bounded
  chunks that are suitable for embedding and retrieval.

Chunking strategy — three-pass pipeline:
  Pass 1 — Hard split on section boundaries.
    Each section heading marks a new chunk.  A section whose body
    exceeds MAX_CHUNK_CHARS is split further in Pass 2.

  Pass 2 — Paragraph-level split for oversized sections.
    Body text is split on double-newline (paragraph boundary).
    Paragraphs are greedily merged into chunks ≤ MAX_CHUNK_CHARS,
    with MIN_CHUNK_CHARS overlap carried forward for context continuity.

  Pass 3 — Sentence-level split for oversized paragraphs.
    Any paragraph still exceeding MAX_CHUNK_CHARS is split on
    sentence boundaries (". ", "? ", "! ") with the same greedy merge.

Each chunk is assigned:
  - A deterministic UUID derived from (paper_id + chunk_index)
    so re-processing the same paper always produces the same IDs.
  - The section heading it belongs to.
  - Its sequential index within the document.

Output per chunk:
  {
    "chunk_id": "uuid4-string",
    "heading":  "Methodology",
    "body":     "...",
    "index":    3
  }

Usage:
    from src.chunkers.semantic_chunker import SemanticChunker
    chunker = SemanticChunker()
    chunks  = chunker.chunk(sections, paper_id="abc123")
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# ── Tunables ───────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 1500   # hard ceiling per chunk
MIN_CHUNK_CHARS = 200    # don't emit chunks smaller than this (merge forward)
OVERLAP_CHARS   = 150    # carry-forward overlap between successive chunks


# ── Input protocol — accepts GrobidSection OR OcrSection ──────────────────────
@runtime_checkable
class HasHeadingAndBody(Protocol):
    heading: str
    body:    str


# ── Output dataclass ───────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str
    heading:  str
    body:     str
    index:    int


# ── Chunker ────────────────────────────────────────────────────────────────────
class SemanticChunker:
    """
    Three-pass semantic chunker for document sections.

    Parameters
    ----------
    max_chars : int
        Maximum characters per chunk (hard ceiling).
    min_chars : int
        Minimum characters to emit as a standalone chunk.
        Smaller chunks are merged with the previous one.
    overlap_chars : int
        Number of trailing characters from the previous chunk
        to prepend to the next for context continuity.
    """

    def __init__(
        self,
        max_chars:     int = MAX_CHUNK_CHARS,
        min_chars:     int = MIN_CHUNK_CHARS,
        overlap_chars: int = OVERLAP_CHARS,
    ) -> None:
        self.max_chars     = max_chars
        self.min_chars     = min_chars
        self.overlap_chars = overlap_chars

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(
        self,
        sections: list,           # list[GrobidSection | OcrSection]
        paper_id: str = "",
    ) -> list[Chunk]:
        """
        Chunk a list of sections into size-bounded Chunk objects.

        Parameters
        ----------
        sections : list
            Parsed sections from Stage 2 (GrobidSection or OcrSection).
        paper_id : str
            Used to generate deterministic chunk UUIDs.

        Returns
        -------
        list[Chunk]
        """
        raw_chunks: list[tuple[str, str]] = []   # (heading, body)

        for section in sections:
            if not isinstance(section, HasHeadingAndBody):
                continue
            heading = section.heading or ""
            body    = section.body    or ""

            if not body.strip():
                continue

            if len(body) <= self.max_chars:
                raw_chunks.append((heading, body))
            else:
                # Pass 2 + 3: split oversized body
                for sub_body in self._split_body(body):
                    raw_chunks.append((heading, sub_body))

        # Merge tiny trailing chunks into previous
        merged = self._merge_small(raw_chunks)

        # Assign deterministic UUIDs and indices
        chunks: list[Chunk] = []
        for idx, (heading, body) in enumerate(merged):
            chunk_id = self._make_uuid(paper_id, idx)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                heading=heading,
                body=body,
                index=idx,
            ))

        return chunks

    # ── Pass 2: paragraph-level split ─────────────────────────────────────────

    def _split_body(self, body: str) -> list[str]:
        """Split body on paragraphs, then sentences if still too large."""
        paragraphs = re.split(r"\n\n+", body)
        pieces: list[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.max_chars:
                pieces.append(para)
            else:
                # Pass 3: sentence-level
                pieces.extend(self._split_sentences(para))

        return self._greedy_merge(pieces)

    # ── Pass 3: sentence-level split ──────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries."""
        # Split after ". ", "? ", "! " — keep delimiter with the left side
        sentences = re.split(r"(?<=[.?!])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    # ── Greedy merge ──────────────────────────────────────────────────────────

    def _greedy_merge(self, pieces: list[str]) -> list[str]:
        """
        Greedily merge pieces into chunks ≤ max_chars,
        carrying OVERLAP_CHARS forward for context continuity.
        """
        if not pieces:
            return []

        chunks:  list[str] = []
        current: str       = ""

        for piece in pieces:
            candidate = (current + "\n\n" + piece).strip() if current else piece

            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # Carry overlap from end of previous chunk
                    overlap  = current[-self.overlap_chars:] if len(current) > self.overlap_chars else current
                    current  = (overlap + "\n\n" + piece).strip()
                else:
                    # Single piece exceeds max — emit it as-is (hard truncation avoided)
                    chunks.append(piece[:self.max_chars])
                    current = ""

        if current:
            chunks.append(current)

        return chunks

    # ── Merge small chunks ─────────────────────────────────────────────────────

    def _merge_small(
        self, raw: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Merge chunks below min_chars into the preceding chunk."""
        if not raw:
            return []

        merged: list[tuple[str, str]] = []

        for heading, body in raw:
            if (
                merged
                and len(body) < self.min_chars
                and len(merged[-1][1]) + len(body) <= self.max_chars
            ):
                prev_h, prev_b = merged[-1]
                merged[-1] = (prev_h, (prev_b + "\n\n" + body).strip())
            else:
                merged.append((heading, body))

        return merged

    # ── UUID generation ────────────────────────────────────────────────────────

    @staticmethod
    def _make_uuid(paper_id: str, index: int) -> str:
        """
        Deterministic UUID5 derived from paper_id + chunk index.
        Same paper always produces the same chunk IDs — safe to re-process.
        """
        namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
        name      = f"{paper_id}::{index}"
        return str(uuid.uuid5(namespace, name))
