"""
src/rag/embedders/vision_embedder.py
======================================
SigLIP / CLIP vision embedder (768-dim) for figure images.

Model:     google/siglip-base-patch16-224  (via OpenCLIP / HuggingFace)
           Falls back to openai/clip-vit-base-patch32 if SigLIP unavailable.
Dim:       768 (SigLIP) / 512 (CLIP fallback)
Storage:   Qdrant collection: figure_chunks

Supports:
  - Image-to-vector:  embed a figure PNG → 768-dim float vector
  - Text-to-vector:   embed a query string → same 768-dim space
                      (enables text → figure cross-modal search)

Usage:
    from src.rag.embedders.vision_embedder import VisionEmbedder
    from PIL import Image

    emb = VisionEmbedder()
    img = Image.open("data/figures/fig1.png")
    vec = emb.embed_image(img)          # np.ndarray (768,)
    qvec = emb.embed_text("attention heatmap diagram")  # same space
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger
from PIL import Image

_model_cache: dict = {}


def _get_siglip():
    if "siglip" not in _model_cache:
        try:
            import open_clip
            logger.info("Loading SigLIP (google/siglip-base-patch16-224) …")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "hf-hub:timm/ViT-B-16-SigLIP"
            )
            tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP")
            model.eval()
            _model_cache["siglip"] = (model, preprocess, tokenizer)
            _model_cache["dim"]    = 768
            logger.info("SigLIP loaded (768-dim).")
        except Exception as exc:
            logger.warning(f"SigLIP load failed ({exc}), falling back to CLIP …")
            _load_clip_fallback()
    return _model_cache


def _load_clip_fallback():
    from transformers import CLIPProcessor, CLIPModel
    logger.info("Loading CLIP (openai/clip-vit-base-patch32) …")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    _model_cache["clip"]      = (model, processor)
    _model_cache["dim"]       = 512
    _model_cache["use_clip"]  = True
    logger.info("CLIP fallback loaded (512-dim).")


class VisionEmbedder:
    """
    Embed figure images and text queries into the same visual-semantic space.

    Parameters
    ----------
    normalize : bool
        L2-normalise output vectors (recommended for cosine similarity).
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def embed_image(
        self, image: Union[Image.Image, Path, str]
    ) -> np.ndarray:
        """
        Embed a PIL Image or image file path.
        Returns np.ndarray of shape (dim,).
        """
        if not isinstance(image, Image.Image):
            image = Image.open(str(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        cache = _get_siglip()

        if cache.get("use_clip"):
            return self._clip_image(image, cache)
        return self._siglip_image(image, cache)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string into the visual-semantic space.
        Enables text → figure cross-modal retrieval.
        Returns np.ndarray of shape (dim,).
        """
        cache = _get_siglip()
        if cache.get("use_clip"):
            return self._clip_text(text, cache)
        return self._siglip_text(text, cache)

    def embed_images_batch(
        self, images: list[Union[Image.Image, Path, str]]
    ) -> np.ndarray:
        """Embed a batch of images. Returns (N, dim) array."""
        if not images:
            return np.empty((0, self.dim()))
        vecs = [self.embed_image(img) for img in images]
        return np.stack(vecs, axis=0)

    @staticmethod
    def dim() -> int:
        cache = _get_siglip()
        return cache.get("dim", 768)

    # ── SigLIP helpers ─────────────────────────────────────────────────────────

    def _siglip_image(self, image: Image.Image, cache: dict) -> np.ndarray:
        import torch
        model, preprocess, _ = cache["siglip"]
        tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(tensor)
        vec = feat.squeeze(0).numpy().astype(np.float32)
        return self._maybe_normalize(vec)

    def _siglip_text(self, text: str, cache: dict) -> np.ndarray:
        import torch
        model, _, tokenizer = cache["siglip"]
        tokens = tokenizer([text])
        with torch.no_grad():
            feat = model.encode_text(tokens)
        vec = feat.squeeze(0).numpy().astype(np.float32)
        return self._maybe_normalize(vec)

    # ── CLIP fallback helpers ──────────────────────────────────────────────────

    def _clip_image(self, image: Image.Image, cache: dict) -> np.ndarray:
        import torch
        model, processor = cache["clip"]
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
        vec = feat.squeeze(0).numpy().astype(np.float32)
        return self._maybe_normalize(vec)

    def _clip_text(self, text: str, cache: dict) -> np.ndarray:
        import torch
        model, processor = cache["clip"]
        inputs = processor(text=[text], return_tensors="pt",
                           padding=True, truncation=True)
        with torch.no_grad():
            feat = model.get_text_features(**inputs)
        vec = feat.squeeze(0).numpy().astype(np.float32)
        return self._maybe_normalize(vec)

    def _maybe_normalize(self, vec: np.ndarray) -> np.ndarray:
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
        return vec
