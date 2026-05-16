"""
Embedding Service — Local BGE-M3 embeddings via sentence-transformers.

BGE-M3 is the best open-source embedding model for this use case:
- Multilingual (100+ languages)
- Supports dense, sparse, and hybrid retrieval
- 1024-dimensional dense vectors
- Runs on CPU or GPU

Falls back to TF-IDF if sentence-transformers is not installed
(useful for lightweight testing environments).

NO paid embedding APIs. NO OpenAI. Fully local.
"""

from __future__ import annotations

import math
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_logger
from research_discovery.models.paper import Paper

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.
    Returns 0.0 on dimension mismatch or zero vectors.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# TF-IDF fallback (no ML dependencies needed)
# ---------------------------------------------------------------------------

def _tfidf_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Lightweight TF-IDF embeddings for testing without GPU/ML stack.

    Returns L2-normalized TF-IDF vectors. Dimension = vocabulary size.
    All vectors are unit-length (comparable via dot product = cosine).
    """
    import re
    from collections import Counter

    # Tokenize
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    tokenized = [tokenize(t) for t in texts]
    n_docs = len(tokenized)

    # Build vocabulary
    vocab: list[str] = []
    seen_vocab: set[str] = set()
    for tokens in tokenized:
        for t in tokens:
            if t not in seen_vocab:
                seen_vocab.add(t)
                vocab.append(t)
    vocab_index = {t: i for i, t in enumerate(vocab)}
    vocab_size = len(vocab)

    if vocab_size == 0:
        return [[0.0] for _ in texts]

    # Document frequency
    df = Counter()
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1

    # TF-IDF matrix
    embeddings = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = [0.0] * vocab_size
        for t, count in tf.items():
            if t in vocab_index:
                tf_score = count / total
                idf = math.log((n_docs + 1) / (df[t] + 1)) + 1.0
                vec[vocab_index[t]] = tf_score * idf
        embeddings.append(_l2_normalize(vec))

    return embeddings


# ---------------------------------------------------------------------------
# BGE-M3 Embedding Model (local, GPU-accelerated)
# ---------------------------------------------------------------------------

class _BGEModel:
    """
    Singleton wrapper for the BGE-M3 sentence-transformers model.

    Loaded once and reused across all embedding calls.
    Supports GPU (cuda) and CPU inference.
    """
    _instance: Optional["_BGEModel"] = None
    _model = None

    @classmethod
    def get(cls) -> Optional["_BGEModel"]:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            device = settings.embedding.device
            # Auto-fallback to CPU if CUDA not available
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                logger.warning("CUDA not available — using CPU for BGE-M3")

            logger.info(
                f"Loading BGE-M3 model: {settings.embedding.model_name} on {device}"
            )
            self._model = SentenceTransformer(
                settings.embedding.model_name,
                cache_folder=settings.embedding.cache_dir,
                device=device,
            )
            logger.info("BGE-M3 loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers\n"
                "Falling back to TF-IDF embeddings (reduced quality)."
            )
            self._model = None
        except Exception as exc:
            logger.error(f"BGE-M3 load failed: {exc} — falling back to TF-IDF")
            self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into dense embeddings.
        Returns list of float vectors.
        """
        if self._model is None:
            return _tfidf_embeddings(texts)

        try:
            vectors = self._model.encode(
                texts,
                batch_size=settings.embedding.batch_size,
                normalize_embeddings=True,   # L2-normalized — cosine = dot product
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]
        except Exception as exc:
            logger.error(f"BGE-M3 encode failed: {exc} — falling back to TF-IDF")
            return _tfidf_embeddings(texts)

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension() or settings.embedding.dimension
        return settings.embedding.dimension


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) — diversity reranking
# ---------------------------------------------------------------------------

def _mmr_rerank(
    query_embedding: list[float],
    paper_embeddings: list[list[float]],
    paper_scores: list[float],
    top_k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """
    Maximal Marginal Relevance reranking for diversity.

    Balances relevance to query vs. diversity from already-selected papers.

    Args:
        query_embedding: embedding of the research idea
        paper_embeddings: list of paper embeddings (same order as papers)
        paper_scores: initial relevance scores (same order)
        top_k: number of papers to select
        lambda_: 0 = max diversity, 1 = max relevance (default 0.7)

    Returns:
        List of selected indices in MMR order.
    """
    n = len(paper_embeddings)
    if n == 0:
        return []

    top_k = min(top_k, n)
    selected: list[int] = []
    remaining = list(range(n))

    # Pre-compute query similarities
    query_sims = [_cosine_similarity(paper_embeddings[i], query_embedding) for i in range(n)]

    for _ in range(top_k):
        if not remaining:
            break

        if not selected:
            # First selection: just pick highest relevance
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            # MMR: relevance - redundancy
            best_score = float("-inf")
            best = remaining[0]
            for i in remaining:
                # Max similarity to any already-selected paper
                max_sim_to_selected = max(
                    _cosine_similarity(paper_embeddings[i], paper_embeddings[s])
                    for s in selected
                )
                mmr_score = (
                    lambda_ * query_sims[i]
                    - (1 - lambda_) * max_sim_to_selected
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i

        selected.append(best)
        remaining.remove(best)

    return selected


# ---------------------------------------------------------------------------
# Public EmbeddingService
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Main embedding service for the Research Discovery Module.

    Responsibilities:
    - Embed research query (single text)
    - Batch-embed all papers (title + abstract)
    - Compute cosine similarity between papers and query
    - MMR diversity reranking
    """

    def __init__(self):
        self._model = _BGEModel.get()

    @property
    def dimension(self) -> int:
        return self._model.dimension if self._model else settings.embedding.dimension

    def embed_query(self, query: str) -> list[float]:
        """Embed the research idea into a single vector."""
        result = self._model.encode([query])
        return result[0]

    def embed_papers(self, papers: list[Paper]) -> list[Paper]:
        """
        Batch-embed all papers using title + abstract.
        Modifies papers in-place (sets paper.embedding).
        Skips papers that already have embeddings.
        Returns the same list.
        """
        to_embed_indices = [i for i, p in enumerate(papers) if not p.embedding]
        if not to_embed_indices:
            return papers

        texts = []
        for i in to_embed_indices:
            p = papers[i]
            # Combine title and abstract for richer embedding
            # BGE-M3 handles long text well (up to 8192 tokens)
            text = p.title
            if p.abstract:
                text += f". {p.abstract[:500]}"  # truncate abstract
            texts.append(text)

        logger.info(f"Embedding {len(texts)} papers with BGE-M3...")
        embeddings = self._model.encode(texts)

        for list_idx, paper_idx in enumerate(to_embed_indices):
            papers[paper_idx].embedding = embeddings[list_idx]

        logger.info(f"Embedded {len(texts)} papers (dim={self.dimension})")
        return papers

    def compute_similarity(
        self,
        papers: list[Paper],
        query_embedding: list[float],
    ) -> list[Paper]:
        """
        Compute cosine similarity between each paper and the query.
        Stores result in paper.ranking_features.semantic_similarity.
        Also sets paper.similarity_score for quick access.
        Returns the same list.
        """
        for paper in papers:
            if paper.embedding:
                sim = _cosine_similarity(paper.embedding, query_embedding)
                paper.similarity_score = round(sim, 6)
                paper.ranking_features.semantic_similarity = round(sim, 6)
        return papers

    def mmr_rerank(
        self,
        papers: list[Paper],
        query_embedding: list[float],
        top_k: int,
        lambda_: float = 0.7,
    ) -> list[Paper]:
        """
        Rerank papers using MMR for diversity.
        Returns top_k papers in MMR order.
        """
        embeddings = [p.embedding for p in papers]
        scores = [p.similarity_score for p in papers]

        selected_indices = _mmr_rerank(
            query_embedding, embeddings, scores, top_k=top_k, lambda_=lambda_
        )

        result = []
        for rank, idx in enumerate(selected_indices):
            paper = papers[idx]
            # Store MMR score (position-based decay)
            paper.ranking_features.mmr_score = round(1.0 - (rank / max(len(selected_indices), 1)), 4)
            result.append(paper)

        return result