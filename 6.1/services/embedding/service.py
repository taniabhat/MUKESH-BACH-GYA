"""
Embedding and semantic ranking service.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.runtime import (
    get_logger,
)
from research_discovery.models.paper import (
    Paper,
    RankingFeatures,
)

logger = get_logger(__name__)

MAX_ABSTRACT_CHARS = 500

DEFAULT_MMR_LAMBDA = 0.7

TOKEN_BUCKET_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Vector Math
# ---------------------------------------------------------------------------

class VectorMath:
    """Vector utility operations."""

    @staticmethod
    def cosine_similarity(
        left: list[float],
        right: list[float],
    ) -> float:

        if (
            not left
            or not right
            or len(left) != len(right)
        ):
            return 0.0

        dot_product = sum(
            x * y
            for x, y in zip(left, right)
        )

        left_norm = math.sqrt(
            sum(x * x for x in left)
        )

        right_norm = math.sqrt(
            sum(y * y for y in right)
        )

        if (
            left_norm == 0.0
            or right_norm == 0.0
        ):
            return 0.0

        return dot_product / (
            left_norm * right_norm
        )

    @staticmethod
    def l2_normalize(
        vector: list[float],
    ) -> list[float]:

        norm = math.sqrt(
            sum(x * x for x in vector)
        )

        if norm == 0.0:
            return vector

        return [
            x / norm
            for x in vector
        ]


# ---------------------------------------------------------------------------
# TF-IDF Fallback
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """Lightweight fallback embedder."""

    @staticmethod
    def encode(
        texts: list[str],
    ) -> list[list[float]]:

        import re

        def tokenize(
            text: str,
        ) -> list[str]:

            return re.findall(
                r"\b[a-z]{2,}\b",
                text.lower(),
            )

        tokenized = [
            tokenize(text)
            for text in texts
        ]

        document_count = len(tokenized)

        vocabulary = []
        seen_tokens = set()

        for tokens in tokenized:
            for token in tokens:
                if token not in seen_tokens:
                    seen_tokens.add(token)
                    vocabulary.append(token)

        if not vocabulary:
            return [
                [0.0]
                for _ in texts
            ]

        vocabulary_index = {
            token: index
            for index, token in enumerate(
                vocabulary
            )
        }

        document_frequency = Counter()

        for tokens in tokenized:
            for token in set(tokens):
                document_frequency[token] += 1

        embeddings = []

        for tokens in tokenized:

            term_frequency = Counter(tokens)

            total_terms = len(tokens) or 1

            vector = [
                0.0
            ] * len(vocabulary)

            for token, count in (
                term_frequency.items()
            ):

                if token not in vocabulary_index:
                    continue

                tf_score = (
                    count / total_terms
                )

                idf_score = (
                    math.log(
                        (
                            document_count
                            + 1
                        )
                        / (
                            document_frequency[
                                token
                            ]
                            + 1
                        )
                    )
                    + 1.0
                )

                vector[
                    vocabulary_index[token]
                ] = (
                    tf_score * idf_score
                )

            embeddings.append(
                VectorMath.l2_normalize(
                    vector
                )
            )

        return embeddings


# ---------------------------------------------------------------------------
# Embedding Backend
# ---------------------------------------------------------------------------

class BGEModel:
    """
    Singleton embedding backend.
    """

    _instance: Optional["BGEModel"] = None

    def __init__(self):

        self._model = None

        self._load_model()

    @classmethod
    def get(
        cls,
    ) -> "BGEModel":

        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def _load_model(
        self,
    ) -> None:

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

            import torch

            device = (
                settings.embedding.device
            )

            if (
                device == "cuda"
                and not torch.cuda.is_available()
            ):

                logger.warning(
                    "CUDA unavailable — "
                    "falling back to CPU"
                )

                device = "cpu"

            logger.info(
                "Loading embedding model "
                "model=%s device=%s",
                settings.embedding.model_name,
                device,
            )

            self._model = (
                SentenceTransformer(
                    settings.embedding.model_name,
                    cache_folder=(
                        settings.embedding.cache_dir
                    ),
                    device=device,
                )
            )

            logger.info(
                "Embedding model loaded"
            )

        except ImportError:

            logger.warning(
                "sentence-transformers unavailable "
                "— using TF-IDF fallback"
            )

            self._model = None

        except Exception:

            logger.exception(
                "Embedding model load failed"
            )

            self._model = None

    def encode(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        if self._model is None:
            return TFIDFEmbedder.encode(
                texts
            )

        try:

            vectors = self._model.encode(
                texts,
                batch_size=(
                    settings.embedding.batch_size
                ),
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return [
                vector.tolist()
                for vector in vectors
            ]

        except Exception:

            logger.exception(
                "Embedding generation failed"
            )

            return TFIDFEmbedder.encode(
                texts
            )

    @property
    def dimension(
        self,
    ) -> int:

        if self._model is None:
            return (
                settings.embedding.dimension
            )

        return (
            self._model.get_sentence_embedding_dimension()
            or settings.embedding.dimension
        )


# ---------------------------------------------------------------------------
# MMR Reranking
# ---------------------------------------------------------------------------

class MMRRanker:
    """Maximal Marginal Relevance reranking."""

    @staticmethod
    def rerank(
        query_embedding: list[float],
        paper_embeddings: list[list[float]],
        top_k: int,
        lambda_: float = DEFAULT_MMR_LAMBDA,
    ) -> list[int]:

        if not paper_embeddings:
            return []

        selected = []

        remaining = list(
            range(len(paper_embeddings))
        )

        query_similarities = [
            VectorMath.cosine_similarity(
                embedding,
                query_embedding,
            )
            for embedding in paper_embeddings
        ]

        top_k = min(
            top_k,
            len(paper_embeddings),
        )

        for _ in range(top_k):

            if not remaining:
                break

            if not selected:

                best_index = max(
                    remaining,
                    key=lambda idx: (
                        query_similarities[idx]
                    ),
                )

            else:

                best_score = float("-inf")

                best_index = remaining[0]

                for index in remaining:

                    max_similarity = max(
                        VectorMath.cosine_similarity(
                            paper_embeddings[
                                index
                            ],
                            paper_embeddings[
                                selected_index
                            ],
                        )
                        for selected_index in selected
                    )

                    mmr_score = (
                        lambda_
                        * query_similarities[
                            index
                        ]
                        - (
                            1 - lambda_
                        )
                        * max_similarity
                    )

                    if mmr_score > best_score:

                        best_score = mmr_score

                        best_index = index

            selected.append(best_index)

            remaining.remove(best_index)

        return selected


# ---------------------------------------------------------------------------
# Public Embedding Service
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Semantic embedding and reranking service.
    """

    def __init__(self):

        self._model = BGEModel.get()

    @property
    def dimension(
        self,
    ) -> int:

        return self._model.dimension

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        embeddings = self._model.encode(
            [query]
        )

        return embeddings[0]

    def embed_papers(
        self,
        papers: list[Paper],
    ) -> list[Paper]:

        pending_indices = [
            index
            for index, paper in enumerate(
                papers
            )
            if not paper.embedding
        ]

        if not pending_indices:
            return papers

        texts = [
            self._build_paper_text(
                papers[index]
            )
            for index in pending_indices
        ]

        logger.info(
            "Embedding papers count=%s",
            len(texts),
        )

        embeddings = self._batch_encode(
            texts
        )

        for embedding_index, paper_index in enumerate(
            pending_indices
        ):

            papers[
                paper_index
            ].embedding = (
                embeddings[
                    embedding_index
                ]
            )

        logger.info(
            "Embedding complete count=%s dim=%s",
            len(texts),
            self.dimension,
        )

        return papers

    def compute_similarity(
        self,
        papers: list[Paper],
        query_embedding: list[float],
    ) -> list[Paper]:

        for paper in papers:

            if not paper.embedding:
                continue

            similarity = round(
                VectorMath.cosine_similarity(
                    paper.embedding,
                    query_embedding,
                ),
                6,
            )

            paper.similarity_score = (
                similarity
            )

            paper.ranking_features = (
                RankingFeatures(
                    semantic_similarity=(
                        similarity
                    ),
                    citation_boost=(
                        paper.ranking_features.citation_boost
                    ),
                    recency_boost=(
                        paper.ranking_features.recency_boost
                    ),
                    venue_score=(
                        paper.ranking_features.venue_score
                    ),
                    keyword_overlap=(
                        paper.ranking_features.keyword_overlap
                    ),
                    graph_centrality=(
                        paper.ranking_features.graph_centrality
                    ),
                    mmr_score=(
                        paper.ranking_features.mmr_score
                    ),
                )
            )

        return papers

    def mmr_rerank(
        self,
        papers: list[Paper],
        query_embedding: list[float],
        top_k: int,
        lambda_: float = DEFAULT_MMR_LAMBDA,
    ) -> list[Paper]:

        embeddings = [
            paper.embedding
            for paper in papers
        ]

        selected_indices = (
            MMRRanker.rerank(
                query_embedding=(
                    query_embedding
                ),
                paper_embeddings=(
                    embeddings
                ),
                top_k=top_k,
                lambda_=lambda_,
            )
        )

        reranked = []

        for rank, paper_index in enumerate(
            selected_indices
        ):

            paper = papers[paper_index]

            paper.ranking_features = (
                RankingFeatures(
                    semantic_similarity=(
                        paper.ranking_features.semantic_similarity
                    ),
                    citation_boost=(
                        paper.ranking_features.citation_boost
                    ),
                    recency_boost=(
                        paper.ranking_features.recency_boost
                    ),
                    venue_score=(
                        paper.ranking_features.venue_score
                    ),
                    keyword_overlap=(
                        paper.ranking_features.keyword_overlap
                    ),
                    graph_centrality=(
                        paper.ranking_features.graph_centrality
                    ),
                    mmr_score=round(
                        1.0
                        - (
                            rank
                            / max(
                                len(
                                    selected_indices
                                ),
                                1,
                            )
                        ),
                        4,
                    ),
                )
            )

            reranked.append(paper)

        return reranked

    def _batch_encode(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        all_embeddings = []

        for start in range(
            0,
            len(texts),
            TOKEN_BUCKET_BATCH_SIZE,
        ):

            batch = texts[
                start:
                start
                + TOKEN_BUCKET_BATCH_SIZE
            ]

            batch_embeddings = (
                self._model.encode(batch)
            )

            all_embeddings.extend(
                batch_embeddings
            )

        return all_embeddings

    @staticmethod
    def _build_paper_text(
        paper: Paper,
    ) -> str:

        text = paper.title

        if paper.abstract:

            text += (
                ". "
                + paper.abstract[
                    :MAX_ABSTRACT_CHARS
                ]
            )

        return text