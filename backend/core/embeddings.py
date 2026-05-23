"""
core/embeddings.py
──────────────────
Local transformer embeddings via sentence-transformers.
All models are lazy-loaded on first use and kept in memory.
Designed for a host with ≤ 8 GB RAM: one text encoder, one
cross-encoder reranker, one vision encoder.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import get_settings
from core.logging import get_logger

settings = get_settings()
logger = get_logger("core.embeddings")

# ── device ─────────────────────────────────────────────────────────
DEVICE = "cpu"


# ── Text encoder ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_text_encoder() -> SentenceTransformer:
    logger.info("embeddings.text_encoder.loading", model=settings.EMBEDDING_MODEL)
    model = SentenceTransformer(
        settings.EMBEDDING_MODEL,
        device=DEVICE,
    )
    model.eval()
    logger.info("embeddings.text_encoder.ready")
    return model


# ── Cross-encoder reranker ───────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    logger.info("embeddings.reranker.loading", model=settings.RERANKER_MODEL)
    model = CrossEncoder(settings.RERANKER_MODEL, device=DEVICE)
    logger.info("embeddings.reranker.ready")
    return model


# ── Vision encoder ───────────────────────────────────────────────────

_vision_model: "AutoModel | None" = None
_vision_processor: "AutoProcessor | None" = None


def _get_vision_model() -> tuple["AutoModel", "AutoProcessor"]:
    global _vision_model, _vision_processor
    if _vision_model is None:
        from transformers import AutoModel, AutoProcessor
        logger.info("embeddings.vision_encoder.loading", model=settings.IMAGE_EMBED_MODEL)
        _vision_processor = AutoProcessor.from_pretrained(settings.IMAGE_EMBED_MODEL)
        _vision_model = AutoModel.from_pretrained(
            settings.IMAGE_EMBED_MODEL,
            torch_dtype=torch.float32,
        ).to(DEVICE).eval()
        logger.info("embeddings.vision_encoder.ready")
    return _vision_model, _vision_processor

def unload_vision_model() -> None:
    global _vision_model, _vision_processor
    _vision_model = None
    _vision_processor = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ── Public async helpers ─────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts.  Returns a list of float32 vectors."""
    if not texts:
        return []
    clean = [t.strip() or " " for t in texts]
    model = _get_text_encoder()

    def _encode() -> np.ndarray:
        return model.encode(
            clean,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    vectors: np.ndarray = await asyncio.to_thread(_encode)
    return vectors.tolist()


async def embed_single(text: str) -> list[float]:
    result = await embed_texts([text])
    return result[0]


async def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """Score candidates with the cross-encoder and return top_n."""
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c["content"]) for c in candidates]

    def _score() -> list[float]:
        return model.predict(pairs, batch_size=16).tolist()

    scores: list[float] = await asyncio.to_thread(_score)

    for candidate, score in zip(candidates, scores):
        candidate["score"] = float(score)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


async def embed_images(image_paths: list[str]) -> list[list[float]]:
    """Embed images via vision encoder. Heavy — lazy-loaded."""
    from PIL import Image

    model, processor = _get_vision_model()
    embeddings: list[list[float]] = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            vec = (
                outputs.pooler_output.squeeze()
                .cpu()
                .numpy()
                .tolist()
            )
        embeddings.append(vec)

    return embeddings
