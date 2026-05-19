"""
src/rag/embedders/code_embedder.py
=====================================
CodeBERT embedder (768-dim) for code snippets.

Model:     microsoft/codebert-base
Dim:       768
Storage:   Qdrant collection: code_chunks

CodeBERT is trained on (code, documentation) pairs across 6 languages
(Python, Java, JavaScript, PHP, Ruby, Go) — ideal for academic ML code.

Code detection:
  We also ship a lightweight _detect_code() heuristic that scans
  section body text for code-like patterns so the pipeline can
  automatically route code snippets to this embedder.

Usage:
    from src.rag.embedders.code_embedder import CodeEmbedder, detect_code_blocks
    emb = CodeEmbedder()
    vec = emb.embed(["def forward(self, x): return self.linear(x)"])
    print(vec.shape)   # (1, 768)

    blocks = detect_code_blocks(section_body)
    # returns list of extracted code strings
"""

from __future__ import annotations

import re

import numpy as np
from loguru import logger

_model_cache: dict = {}

# ── Code detection patterns ────────────────────────────────────────────────────
_CODE_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),                          # markdown fenced
    re.compile(r"(?:def |class |import |from .+ import )"), # Python keywords
    re.compile(r"(?:void |public |private |static |int |float )\w+\s*\("), # Java/C
    re.compile(r"(?:function |const |let |var )\w+"),       # JavaScript
    re.compile(r"(?:Algorithm|Procedure|Input:|Output:)\s"), # pseudocode
    re.compile(r"for\s+\w+\s+in\s+range\("),                # Python loop
    re.compile(r"if\s+__name__\s*=="),                      # Python main guard
]
_MIN_CODE_LINE_RATIO = 0.30   # ≥30% of lines look like code → it's a code block


def detect_code_blocks(text: str) -> list[str]:
    """
    Extract code-like blocks from a section body.
    Returns a list of code strings.
    """
    # First: explicit markdown fenced blocks
    fenced = re.findall(r"```(?:\w+\n)?([\s\S]*?)```", text)
    if fenced:
        return [b.strip() for b in fenced if b.strip()]

    # Second: paragraph-level heuristic
    blocks: list[str] = []
    for paragraph in re.split(r"\n\n+", text):
        lines     = [l for l in paragraph.splitlines() if l.strip()]
        if not lines:
            continue
        code_lines = sum(
            1 for l in lines
            if any(p.search(l) for p in _CODE_PATTERNS)
        )
        if len(lines) > 2 and (code_lines / len(lines)) >= _MIN_CODE_LINE_RATIO:
            blocks.append(paragraph.strip())

    return blocks


def _get_model():
    if "model" not in _model_cache:
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading CodeBERT (microsoft/codebert-base) …")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model     = AutoModel.from_pretrained("microsoft/codebert-base")
        model.eval()
        _model_cache["tokenizer"] = tokenizer
        _model_cache["model"]     = model
        logger.info("CodeBERT loaded (768-dim).")
    return _model_cache["tokenizer"], _model_cache["model"]


class CodeEmbedder:
    """
    Embed code snippets with CodeBERT.

    Parameters
    ----------
    max_length : int
        Token limit (CodeBERT supports up to 512).
    normalize : bool
        L2-normalise output vectors.
    """

    def __init__(self, max_length: int = 512, normalize: bool = True) -> None:
        self.max_length = max_length
        self.normalize  = normalize

    def embed(self, snippets: list[str]) -> np.ndarray:
        """
        Embed a list of code strings.
        Returns np.ndarray of shape (N, 768).
        """
        if not snippets:
            return np.empty((0, 768))

        import torch
        tokenizer, model = _get_model()

        inputs = tokenizer(
            snippets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool over token dimension (more stable than [CLS] for code)
        mask     = inputs["attention_mask"].unsqueeze(-1).float()
        summed   = (outputs.last_hidden_state * mask).sum(dim=1)
        counts   = mask.sum(dim=1).clamp(min=1e-8)
        vecs     = (summed / counts).numpy().astype(np.float32)

        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            vecs  = vecs / norms

        return vecs

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (768,) array."""
        return self.embed([query])[0]

    @staticmethod
    def dim() -> int:
        return 768
