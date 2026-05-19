"""
src/rag/rerankers/bge_reranker.py
====================================
BGE Reranker v2 — cross-encoder reranking (top 50 → top 10).

What is cross-encoder reranking?
  Bi-encoder retrieval (BGE-M3) encodes query and document SEPARATELY,
  which is fast but loses query-document interaction.

  A cross-encoder (BGE Reranker v2) processes the query and document
  TOGETHER in one forward pass, giving it full attention over both.
  This is much slower but significantly more accurate.

  The standard pattern is:
    Retrieve top-50 cheaply (bi-encoder) → Rerank to top-10 (cross-encoder)

Model:      BAAI/bge-reranker-v2-m3
Input:      (query, passage) pair
Output:     relevance score (logit → sigmoid → 0…1)

Usage:
    from src.rag.rerankers.bge_reranker import BGEReranker
    from src.rag.fusion.rrf_fuser import FusedResult

    reranker = BGEReranker()
    top10    = reranker.rerank(query, fused_results, top_k=10)
"""

from __future__ import annotations

from loguru import logger

_model_cache: dict = {}


def _get_reranker():
    if "model" not in _model_cache:
        from FlagEmbedding import FlagReranker
        logger.info("Loading BGE Reranker v2 (BAAI/bge-reranker-v2-m3) …")
        _model_cache["model"] = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=False,
        )
        logger.info("BGE Reranker v2 loaded.")
    return _model_cache["model"]


class BGEReranker:
    """
    Cross-encoder reranker using BGE Reranker v2.

    Parameters
    ----------
    batch_size : int
        Pairs per forward pass. Lower = less RAM.
    """

    def __init__(self, batch_size: int = 16) -> None:
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list,        # list[FusedResult]
        top_k: int = 10,
    ) -> list:
        """
        Rerank candidates using cross-encoder relevance scoring.

        Parameters
        ----------
        query : str
            Original user query.
        candidates : list[FusedResult]
            Up to 50 candidates from RRF fusion.
        top_k : int
            How many to return after reranking.

        Returns
        -------
        list[FusedResult] (reranked, trimmed to top_k)
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            logger.debug("Reranker: fewer candidates than top_k — skipping")
            return candidates[:top_k]

        # Build (query, passage) pairs
        # Extract text from payload — try common field names
        pairs = []
        for c in candidates:
            text = (
                c.payload.get("body")      or
                c.payload.get("caption")   or
                c.payload.get("equation")  or
                c.payload.get("code")      or
                c.payload.get("markdown")  or
                ""
            )
            pairs.append([query, text[:512]])   # truncate for speed

        try:
            reranker = _get_reranker()
            scores   = reranker.compute_score(pairs, batch_size=self.batch_size)

            # Attach scores and sort
            scored = list(zip(scores, candidates))
            scored.sort(key=lambda x: x[0], reverse=True)

            result = [c for _, c in scored[:top_k]]
            logger.info(
                f"BGEReranker: {len(candidates)} → {len(result)} "
                f"(top score={scored[0][0]:.3f})"
            )
            return result

        except Exception as exc:
            logger.warning(f"Reranker failed ({exc}), returning RRF order")
            return candidates[:top_k]
