"""
src/rag/embedders/text_embedder.py
====================================
Cloud-based BGE-M3 text embedder (1024-dim) via Hugging Face Inference API.

Used for:  text_chunks, table_chunks, equation_chunks
Model:     BAAI/bge-m3  — multilingual, long-context (up to 8192 tokens)
API:       https://api-inference.huggingface.co (free tier, no local storage)

The Hugging Face Inference API only returns dense vectors (1024-dim).
Sparse vectors are returned as empty dicts — hybrid BM25 arms that rely on
sparse weights will silently degrade to dense-only, which is still excellent.

Usage:
    from src.rag.embedders.text_embedder import TextEmbedder
    emb = TextEmbedder()
    result = emb.embed(["Attention is all you need", "Deep learning methods"])
    print(result.dense.shape)    # (2, 1024)
    print(result.sparse)         # [{}, {}]  — empty (API limitation)

Environment:
    HF_TOKEN=hf_...   (required — add to your .env file)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
from loguru import logger


@dataclass
class TextEmbedding:
    dense:  np.ndarray               # shape (N, 1024)
    sparse: list[dict[int, float]]   # one sparse vector per input (empty for API mode)
    texts:  list[str]


class TextEmbedder:
    """
    Cloud-based Text Embedder using Hugging Face Inference API (BAAI/bge-m3).
    Requires 0 GB of local storage.

    Fully drop-in compatible with the local FlagEmbedding version:
      - .embed(texts)        → TextEmbedding
      - .embed_query(query)  → TextEmbedding
      - .dim()               → 1024

    Parameters
    ----------
    batch_size : int
        Max texts per API call. The free HF tier handles ~32 short texts.
        Use a smaller value (8–16) for long academic paper chunks.
    max_retries : int
        Number of retries on rate-limit (HTTP 429) or model-loading (HTTP 503).
    retry_delay : float
        Seconds to wait between retries (doubles on each attempt).
    """

    _API_URL = (
        "https://api-inference.huggingface.co"
        "/pipeline/feature-extraction/BAAI/bge-m3"
    )

    def __init__(
        self,
        batch_size:  int   = 16,
        max_length:  int   = 512,   # kept for API compatibility; ignored by HF API
        max_retries: int   = 5,
        retry_delay: float = 2.0,
    ) -> None:
        self.batch_size  = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found. "
                "Please add HF_TOKEN=hf_... to your .env file."
            )

        self._headers = {"Authorization": f"Bearer {hf_token}"}
        logger.info("Initialized Cloud BGE-M3 TextEmbedder (HF Inference API)")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call_api(self, texts: list[str]) -> np.ndarray:
        """
        POST a batch of texts to the HF Inference API and return a dense
        numpy array of shape (len(texts), 1024).

        Retries automatically on 429 (rate limit) and 503 (model loading).
        """
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            response = requests.post(
                self._API_URL,
                headers=self._headers,
                json={"inputs": texts},
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                return np.array(data, dtype=np.float32)

            if response.status_code in (429, 503):
                # Rate-limited or model still warming up — wait and retry
                logger.warning(
                    f"HF API returned {response.status_code} "
                    f"(attempt {attempt}/{self.max_retries}). "
                    f"Retrying in {delay:.1f}s …"
                )
                time.sleep(delay)
                delay *= 2   # exponential back-off
                continue

            # Any other status code is a hard error
            raise RuntimeError(
                f"Hugging Face API Error {response.status_code}: {response.text}"
            )

        raise RuntimeError(
            f"HF API failed after {self.max_retries} retries. "
            "Check your token and network connection."
        )

    # ── Public interface (matches original local embedder) ─────────────────────

    def embed(self, texts: list[str]) -> TextEmbedding:
        """
        Embed a list of texts.

        Returns TextEmbedding with:
          .dense  — numpy array (N, 1024)
          .sparse — list of empty dicts (API doesn't return sparse weights)
          .texts  — original input (for traceability)
        """
        if not texts:
            return TextEmbedding(
                dense=np.empty((0, 1024), dtype=np.float32),
                sparse=[],
                texts=[],
            )

        logger.debug(f"TextEmbedder: embedding {len(texts)} texts via HF API …")

        # Split into batches to respect free-tier limits
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs  = self._call_api(batch)
            all_vecs.append(vecs)

        dense = np.vstack(all_vecs)   # (N, 1024)
        logger.debug(f"TextEmbedder: dense shape={dense.shape}")

        # Sparse is unsupported by the HF pipeline endpoint — return empty dicts
        sparse: list[dict[int, float]] = [{} for _ in texts]

        return TextEmbedding(dense=dense, sparse=sparse, texts=texts)

    def embed_query(self, query: str) -> TextEmbedding:
        """Embed a single query string."""
        return self.embed([query])

    @staticmethod
    def dim() -> int:
        return 1024
