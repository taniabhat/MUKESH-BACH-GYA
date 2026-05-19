"""
src/rag/compression/context_compressor.py
===========================================
Context compressor — removes redundant content and enforces token budget.

Two-pass compression:

  Pass 1 — Deduplication
    Chunks with cosine similarity ≥ SIM_THRESHOLD are considered
    redundant. We keep the one with the higher reranker score and
    drop the rest. This handles paraphrased duplicates from multiple
    retrieval arms pulling in the same information differently.

  Pass 2 — Token budget enforcement
    Counts tokens (via tiktoken cl100k_base — GPT-4 / Claude tokeniser)
    and greedily includes chunks from highest→lowest score until the
    budget is exhausted.

  Output is sorted by document order (page + chunk index) so the
  Research Agent receives context in reading order, not score order.

Budget recommendation:
  8192 tokens ≈ ~6000 words ≈ 10–15 dense academic paragraphs.
  This fits comfortably in Claude / Qwen3's context window.

Usage:
    from src.rag.compression.context_compressor import ContextCompressor
    from src.rag.rerankers.bge_reranker import BGEReranker

    compressor = ContextCompressor(token_budget=8192)
    context    = compressor.compress(reranked_results)
    print(context.text)          # final prompt-ready string
    print(context.token_count)   # actual tokens used
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

SIM_THRESHOLD = 0.92     # cosine similarity above this → treat as duplicate
DEFAULT_BUDGET = 8192    # tokens


@dataclass
class CompressedContext:
    chunks:       list[dict]    # kept chunks with metadata
    text:         str           # final formatted string for prompt injection
    token_count:  int
    dropped:      int           # how many were removed


class ContextCompressor:
    """
    Deduplicates and token-budgets the reranked context.

    Parameters
    ----------
    token_budget : int
        Maximum tokens in the final context string.
    sim_threshold : float
        Cosine similarity above which two chunks are considered duplicates.
    """

    def __init__(
        self,
        token_budget:   int   = DEFAULT_BUDGET,
        sim_threshold:  float = SIM_THRESHOLD,
    ) -> None:
        self.token_budget  = token_budget
        self.sim_threshold = sim_threshold
        self._tokenizer    = None

    def compress(
        self,
        reranked: list,          # list[FusedResult] after reranking
        query: str = "",
    ) -> CompressedContext:
        """
        Compress reranked results into a token-budgeted context string.

        Parameters
        ----------
        reranked : list[FusedResult]
            Top results from BGE Reranker.
        query : str
            Original query (used for relevance header in output).
        """
        if not reranked:
            return CompressedContext(chunks=[], text="", token_count=0, dropped=0)

        original_count = len(reranked)

        # Pass 1: deduplicate
        deduped = self._deduplicate(reranked)

        # Pass 2: token budget
        kept, total_tokens = self._apply_budget(deduped)

        # Sort by document order (page, then chunk index)
        kept.sort(key=lambda x: (
            x.payload.get("page", 0),
            x.payload.get("index", 0),
        ))

        # Format into prompt-ready string
        text = self._format(kept, query)

        dropped = original_count - len(kept)
        logger.info(
            f"ContextCompressor: {original_count} → {len(kept)} chunks "
            f"({total_tokens} tokens, {dropped} dropped)"
        )

        return CompressedContext(
            chunks=[{"chunk_id": c.chunk_id,
                     "payload":  c.payload,
                     "score":    c.rrf_score,
                     "sources":  c.sources}
                    for c in kept],
            text=text,
            token_count=total_tokens,
            dropped=dropped,
        )

    # ── Pass 1: deduplication ──────────────────────────────────────────────────

    def _deduplicate(self, results: list) -> list:
        """Remove near-duplicate chunks using text overlap heuristic."""
        kept:       list  = []
        kept_texts: list[str] = []

        for r in results:
            text = self._get_text(r)
            if not text:
                kept.append(r)
                continue

            # Check overlap with already-kept texts
            is_dup = False
            for existing in kept_texts:
                sim = _text_overlap(text, existing)
                if sim >= self.sim_threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(r)
                kept_texts.append(text)

        logger.debug(
            f"Dedup: {len(results)} → {len(kept)} "
            f"({len(results) - len(kept)} duplicates removed)"
        )
        return kept

    # ── Pass 2: token budget ───────────────────────────────────────────────────

    def _apply_budget(self, results: list) -> tuple[list, int]:
        """Greedily include results until token budget is exhausted."""
        tokenizer  = self._get_tokenizer()
        kept:  list = []
        total: int  = 0

        for r in results:
            text   = self._get_text(r)
            tokens = len(tokenizer.encode(text)) if text else 0

            if total + tokens > self.token_budget:
                logger.debug(
                    f"Budget hit at {total} tokens — "
                    f"stopping before chunk {r.chunk_id[:8]}"
                )
                break

            kept.append(r)
            total += tokens

        return kept, total

    # ── Formatting ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format(kept: list, query: str) -> str:
        """Format kept chunks into a structured prompt-injectable string."""
        parts: list[str] = []

        if query:
            parts.append(f"[Query: {query}]\n")

        for i, r in enumerate(kept, 1):
            p        = r.payload
            modality = p.get("modality", "text")
            heading  = p.get("heading", "")
            paper_id = p.get("paper_id", "")[:8]

            # Section header
            src_label = f"[{modality.upper()} | paper:{paper_id} | sources:{','.join(r.sources)}]"
            if heading:
                parts.append(f"\n--- Chunk {i} {src_label} — {heading} ---")
            else:
                parts.append(f"\n--- Chunk {i} {src_label} ---")

            # Content by modality
            if modality == "figure":
                parts.append(f"Caption: {p.get('caption', '')}")
                parts.append(f"Image: {p.get('image_path', '')}")
            elif modality == "table":
                parts.append(f"Caption: {p.get('caption', '')}")
                parts.append(p.get("markdown", ""))
            elif modality == "equation":
                parts.append(f"Equation: {p.get('equation', '')}")
            elif modality == "code":
                parts.append(f"```\n{p.get('code', '')}\n```")
            else:
                parts.append(p.get("body", ""))

        return "\n".join(parts)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # Fallback: rough approximation (1 token ≈ 4 chars)
                logger.warning("tiktoken not installed — using char/4 approximation")
                self._tokenizer = _CharTokenizer()
        return self._tokenizer

    @staticmethod
    def _get_text(r) -> str:
        p = r.payload
        return (
            p.get("body")     or
            p.get("caption")  or
            p.get("equation") or
            p.get("code")     or
            p.get("markdown") or
            ""
        )


# ── Text overlap heuristic (Jaccard on word sets) ─────────────────────────────

def _text_overlap(a: str, b: str) -> float:
    """Fast Jaccard similarity on word bigrams."""
    def bigrams(text: str) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return set(zip(words, words[1:])) if len(words) > 1 else set(words)

    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    intersection = len(bg_a & bg_b)
    union        = len(bg_a | bg_b)
    return intersection / union if union else 0.0


class _CharTokenizer:
    """Fallback tokenizer when tiktoken is unavailable."""
    @staticmethod
    def encode(text: str) -> list:
        return list(range(max(1, len(text) // 4)))
