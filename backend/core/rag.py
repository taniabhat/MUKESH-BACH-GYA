import asyncio
import hashlib
import math
import uuid
from collections import defaultdict
from io import BytesIO

import httpx
import numpy as np
import torch
from huggingface_hub import InferenceClient
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams
from transformers import AutoModel
from transformers import AutoProcessor

from config import get_settings
from core.document import DocumentResult
from core.llm import build_system_message
from core.llm import build_user_message
from core.llm import chat
from core.llm import get_model
from core.logging import get_logger


settings = get_settings()

logger = get_logger("core.rag")


# -------------------------------------------------------------------
# HF Inference Client
# -------------------------------------------------------------------


hf_client = InferenceClient(
    provider="hf-inference",
    api_key=settings.HF_API_KEY
)


# -------------------------------------------------------------------
# Lazy Loaded Vision Models
# -------------------------------------------------------------------


_vision_model = None
_vision_processor = None


def get_vision_model():

    global _vision_model
    global _vision_processor

    if (
        _vision_model is None
        or _vision_processor is None
    ):

        _vision_processor = AutoProcessor.from_pretrained(
            settings.IMAGE_EMBED_MODEL
        )

        _vision_model = AutoModel.from_pretrained(
            settings.IMAGE_EMBED_MODEL
        )

    return (
        _vision_model,
        _vision_processor
    )


# -------------------------------------------------------------------
# Qdrant Client
# -------------------------------------------------------------------


qdrant = QdrantClient(
    url=settings.QDRANT_URL
)


# -------------------------------------------------------------------
# Collections
# -------------------------------------------------------------------


def ensure_qdrant_collections() -> None:

    collections = qdrant.get_collections().collections

    existing = {
        collection.name
        for collection in collections
    }

    collection_configs = {
        "text_chunks": 1024,
        "figure_chunks": 768,
        "table_chunks": 1024,
        "code_chunks": 768,
        "equation_chunks": 1024
    }

    for collection_name, dimension in collection_configs.items():

        if collection_name in existing:
            continue

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )


# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------


async def embed_text(
    texts: list[str]
) -> list[list[float]]:

    embeddings = []

    for text in texts:
        clean_text = text if (text and str(text).strip()) else " "

        vector = hf_client.feature_extraction(
            clean_text,
            model=settings.EMBEDDING_MODEL
        )

        if (
            isinstance(vector, list)
            and len(vector) > 0
            and isinstance(vector[0], list)
        ):
            pooled = np.mean(
                np.array(vector),
                axis=0
            ).tolist()

            embeddings.append(pooled)

        else:
            embeddings.append(vector)

    return embeddings


async def embed_images(
    image_paths: list[str]
) -> list[list[float]]:

    model, processor = get_vision_model()

    embeddings = []

    for path in image_paths:

        image = Image.open(
            path
        ).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(**inputs)

            pooled = (
                outputs.last_hidden_state
                .mean(dim=1)
                .squeeze()
                .cpu()
                .numpy()
                .tolist()
            )

        embeddings.append(pooled)

    return embeddings


async def embed_code(
    code_snippets: list[str]
) -> list[list[float]]:

    embeddings = []

    for snippet in code_snippets:

        vector = hf_client.feature_extraction(
            snippet,
            model=settings.CODE_EMBED_MODEL
        )

        if (
            isinstance(vector, list)
            and len(vector) > 0
            and isinstance(vector[0], list)
        ):

            pooled = np.mean(
                np.array(vector),
                axis=0
            ).tolist()

            embeddings.append(pooled)

        else:
            embeddings.append(vector)

    return embeddings


# -------------------------------------------------------------------
# Indexing
# -------------------------------------------------------------------


def build_payload(
    chunk: dict
) -> dict:

    metadata = chunk.get(
        "metadata",
        {}
    )

    return {
        "paper_id": chunk.get("paper_id"),
        "project_id": metadata.get("project_id"),
        "section": chunk.get("section"),
        "chunk_type": chunk.get("chunk_type"),
        "content": chunk.get("content"),
        "metadata": metadata,
        "year": metadata.get("year")
    }


def upsert_points(
    collection_name: str,
    chunks: list[dict],
    vectors: list[list[float]]
) -> None:

    points = []

    for chunk, vector in zip(chunks, vectors):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=build_payload(chunk)
            )
        )

    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )


async def index_paper(
    doc: DocumentResult,
    project_id: str
) -> None:

    logger.info("rag.index_paper.started", paper_id=doc.paper_id, project_id=project_id)
    chunks = chunk_document(doc)

    logger.debug("rag.index_paper.chunked", paper_id=doc.paper_id, chunks=len(chunks))

    text_chunks = []
    figure_chunks = []
    table_chunks = []
    equation_chunks = []

    for chunk in doc_result.chunks:

        chunk_type = chunk["chunk_type"]

        if chunk_type == "text":
            text_chunks.append(chunk)

        elif chunk_type == "figure":
            figure_chunks.append(chunk)

        elif chunk_type == "table":
            table_chunks.append(chunk)

        elif chunk_type == "equation":
            equation_chunks.append(chunk)

    if text_chunks:

        text_vectors = await embed_text([
            chunk["content"]
            for chunk in text_chunks
        ])

        upsert_points(
            "text_chunks",
            text_chunks,
            text_vectors
        )

    if figure_chunks:

        image_vectors = await embed_images([
            chunk["content"]
            for chunk in figure_chunks
        ])

        upsert_points(
            "figure_chunks",
            figure_chunks,
            image_vectors
        )

    if table_chunks:

        table_vectors = await embed_text([
            chunk["content"]
            for chunk in table_chunks
        ])

        upsert_points(
            "table_chunks",
            table_chunks,
            table_vectors
        )

    if equation_chunks:

        equation_vectors = await embed_text([
            chunk["content"]
            for chunk in equation_chunks
        ])

        upsert_points(
            "equation_chunks",
            equation_chunks,
            equation_vectors
        )

    logger.info("rag.index_paper.success", paper_id=doc.paper_id)


# -------------------------------------------------------------------
# Query Rewriting
# -------------------------------------------------------------------


async def rewrite_query(
    query: str
) -> list[str]:

    messages = [
        build_system_message(
            """
Generate 3 query reformulations:

1. paraphrase
2. keyword-focused
3. methodology-focused

Return ONLY one query per line.
"""
        ),
        build_user_message(query)
    ]

    response = await chat(
        messages=messages,
        model=get_model("fast"),
        temperature=0.2,
        max_tokens=256
    )

    rewrites = [
        line.strip()
        for line in response.split("\n")
        if line.strip()
    ]

    return [query] + rewrites[:3]


# -------------------------------------------------------------------
# Hybrid Search
# -------------------------------------------------------------------


def build_filter(
    filters: dict | None
) -> Filter | None:

    if not filters:
        return None

    conditions = []

    for key, value in filters.items():

        conditions.append(
            FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
        )

    return Filter(
        must=conditions
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60
) -> list[dict]:

    scores = defaultdict(float)

    payload_map = {}

    for ranked in ranked_lists:

        for rank, item in enumerate(ranked):

            chunk_id = item["chunk_id"]

            scores[chunk_id] += (
                1 / (k + rank + 1)
            )

            payload_map[chunk_id] = item

    reranked = []

    for chunk_id, score in scores.items():

        item = payload_map[chunk_id]

        item["score"] = score

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return reranked


async def hybrid_search(
    query: str,
    collection: str,
    top_k: int,
    filters: dict | None = None
) -> list[dict]:

    query_vector = (
        await embed_text([query])
    )[0]

    qdrant_filter = build_filter(
        filters
    )

    dense_results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter
    )

    formatted_results = []

    for result in dense_results:

        payload = result.payload

        formatted_results.append({
            "chunk_id": str(result.id),
            "content": payload.get("content"),
            "score": result.score,
            "metadata": payload.get(
                "metadata",
                {}
            ),
            "section": payload.get(
                "section"
            ),
            "chunk_type": payload.get(
                "chunk_type"
            )
        })

    logger.info("rag.retrieve.success", query=query, retrieved_count=len(formatted_results))
    return formatted_results


# -------------------------------------------------------------------
# Reranking
# -------------------------------------------------------------------


async def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 10
) -> list[dict]:

    if not candidates:
        logger.warning("rag.rerank.empty_candidates")
        return []

    candidate_texts = [
        item["content"]
        for item in candidates
    ]

    scores = hf_client.sentence_similarity(
        {
            "source_sentence": query,
            "sentences": candidate_texts
        },
        model=settings.RERANKER_MODEL
    )

    rescored = []

    for item, score in zip(
        candidates,
        scores
    ):

        item["score"] = float(score)

        rescored.append(item)

    rescored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    logger.debug("rag.rerank.success", initial_count=len(candidates), final_count=len(rescored[:top_n]))
    return rescored[:top_n]


# -------------------------------------------------------------------
# Context Compression
# -------------------------------------------------------------------


def content_overlap(
    a: str,
    b: str
) -> float:

    a_words = set(
        a.lower().split()
    )

    b_words = set(
        b.lower().split()
    )

    if not a_words or not b_words:
        return 0.0

    intersection = len(
        a_words & b_words
    )

    union = len(
        a_words | b_words
    )

    return intersection / union


def compress_context(
    chunks: list[dict],
    max_tokens: int = 8000
) -> list[dict]:

    selected = []

    total_tokens = 0

    sorted_chunks = sorted(
        chunks,
        key=lambda x: x["score"],
        reverse=True
    )

    for chunk in sorted_chunks:

        estimated_tokens = (
            len(chunk["content"]) // 4
        )

        if (
            total_tokens
            + estimated_tokens
            > max_tokens
        ):
            continue

        is_duplicate = False

        for existing in selected:

            overlap = content_overlap(
                chunk["content"],
                existing["content"]
            )

            if overlap > 0.8:

                is_duplicate = True
                break

        if is_duplicate:
            continue

        selected.append(chunk)

        total_tokens += estimated_tokens

    return selected


# -------------------------------------------------------------------
# Main Retrieval Pipeline
# -------------------------------------------------------------------


async def retrieve(
    query: str,
    project_id: str | None = None,
    top_k: int = 10,
    collections: list[str] = ["text_chunks"]
) -> list[dict]:

    logger.debug("rag.retrieve.started", query=query, collections=collections, top_k=top_k)
    rewritten_queries = await rewrite_query(
        query
    )

    all_results = []

    filters = {
        "project_id": project_id
    }

    collections = [
        "text_chunks",
        "table_chunks",
        "equation_chunks"
    ]

    for rewritten_query in rewritten_queries:

        for collection in collections:

            results = await hybrid_search(
                query=rewritten_query,
                collection=collection,
                top_k=top_k,
                filters=filters
            )

            all_results.append(results)

    fused_results = reciprocal_rank_fusion(
        all_results
    )

    reranked = await rerank(
        query=query,
        candidates=fused_results,
        top_n=top_k
    )

    compressed = compress_context(
        reranked,
        max_tokens=settings.MAX_CONTEXT_TOKENS
    )

    return compressed