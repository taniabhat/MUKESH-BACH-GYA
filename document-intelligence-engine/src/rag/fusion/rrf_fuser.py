"""
src/rag/fusion/rrf_fuser.py
=============================
Reciprocal Rank Fusion (RRF) — combines results from all retrieval arms.

What is RRF?
  RRF is a score-free rank fusion algorithm. Instead of relying on
  raw similarity scores (which are on incompatible scales across arms),
  it uses RANK positions. The formula for each document d is:

    RRF(d) = Σ  1 / (k + rank_i(d))
             i

  where rank_i(d) is the rank of d in retrieval arm i, and
  k=60 is a smoothing constant (standard value from the 2009 paper).

Why RRF over score normalisation?
  - Dense cosine scores, BM25 scores, and graph hop-counts are
    on completely different scales — normalising them is fragile.
  - RRF is parameter-free (only k matters), robust, and consistently
    outperforms weighted sum fusion in benchmarks.
  - Documents appearing in MULTIPLE arms get a strong bonus —
    this is exactly what we want: multi-evidence corroboration.

Usage:
    from src.rag.fusion.rrf_fuser import RRFFuser
    from src.rag.retrievers.hybrid_retriever import RetrievedChunk

    fuser   = RRFFuser(k=60, top_n=50)
    fused   = fuser.fuse(all_chunks)   # list[RetrievedChunk] from all arms
    top_10  = fused[:10]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class FusedResult:
    chunk_id:   str
    rrf_score:  float
    payload:    dict
    sources:    list[str]      # which arms contributed ("dense", "bm25", ...)
    collection: str


class RRFFuser:
    """
    Reciprocal Rank Fusion over heterogeneous retrieval results.

    Parameters
    ----------
    k : int
        RRF smoothing constant (default 60, from original paper).
    top_n : int
        Return at most top_n fused results.
    """

    def __init__(self, k: int = 60, top_n: int = 50) -> None:
        self.k     = k
        self.top_n = top_n

    def fuse(
        self,
        results: list,          # list[RetrievedChunk]
        citation_paper_ids: Optional[list[str]] = None,
    ) -> list[FusedResult]:
        """
        Fuse multi-arm retrieval results with RRF.

        Parameters
        ----------
        results : list[RetrievedChunk]
            All raw results from all retrieval arms combined.
        citation_paper_ids : list[str], optional
            Paper IDs discovered via citation graph traversal.
            These receive a citation-graph bonus score.

        Returns
        -------
        list[FusedResult]
            Sorted by rrf_score descending, at most top_n items.
        """
        if not results:
            return []

        # Group results by (source, collection) to get per-arm ranked lists
        arms: dict[str, list] = {}
        for r in results:
            arm_key = f"{r.source}::{r.collection}"
            arms.setdefault(arm_key, []).append(r)

        # Sort each arm by its own score (descending)
        for arm_key in arms:
            arms[arm_key].sort(key=lambda x: x.score, reverse=True)

        # Accumulate RRF scores
        rrf_scores:  dict[str, float]      = {}
        chunk_data:  dict[str, dict]       = {}   # chunk_id → payload
        chunk_src:   dict[str, set[str]]   = {}   # chunk_id → set of sources
        chunk_coll:  dict[str, str]        = {}   # chunk_id → collection

        for arm_key, ranked in arms.items():
            for rank, item in enumerate(ranked, start=1):
                cid = item.chunk_id
                rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + 1.0 / (self.k + rank)
                chunk_data[cid]  = item.payload
                chunk_src.setdefault(cid, set()).add(item.source)
                chunk_coll[cid]  = item.collection

        # Citation graph bonus: papers found via graph get +1/(k+1) boost
        if citation_paper_ids:
            for cid, payload in chunk_data.items():
                if payload.get("paper_id") in citation_paper_ids:
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k + 1)

        # Sort and package
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        fused: list[FusedResult] = []
        for cid in sorted_ids[: self.top_n]:
            fused.append(FusedResult(
                chunk_id=cid,
                rrf_score=round(rrf_scores[cid], 6),
                payload=chunk_data[cid],
                sources=sorted(chunk_src[cid]),
                collection=chunk_coll[cid],
            ))

        logger.debug(
            f"RRF fused {len(results)} raw → {len(fused)} unique "
            f"(top score={fused[0].rrf_score:.4f} if fused else n/a)"
        )
        return fused
