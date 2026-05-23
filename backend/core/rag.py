import asyncio
import hashlib
import math
import random
import uuid
from collections import defaultdict
from io import BytesIO

import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams

from config import get_settings
from core.document import DocumentResult
from core.llm import build_system_message
from core.llm import build_user_message
from core.llm import chat
from core.llm import get_model
from core.logging import get_logger
from core.embeddings import embed_texts, embed_single, rerank as rerank_candidates, embed_images


settings = get_settings()

logger = get_logger("core.rag")


# -------------------------------------------------------------------
# Qdrant Client
# -------------------------------------------------------------------


if settings.QDRANT_URL:
    qdrant = QdrantClient(url=settings.QDRANT_URL)
else:
    qdrant = QdrantClient(path=settings.QDRANT_PATH)


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
        "text_chunks": settings.EMBEDDING_DIM,
        "figure_chunks": 768,
        "table_chunks": settings.EMBEDDING_DIM,
        "code_chunks": settings.EMBEDDING_DIM,
        "equation_chunks": settings.EMBEDDING_DIM
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
    
    chunks = doc.chunks
    for c in chunks:
        if "metadata" not in c:
            c["metadata"] = {}
        c["metadata"]["project_id"] = project_id

    logger.debug("rag.index_paper.chunked", paper_id=doc.paper_id, chunks=len(chunks))

    text_chunks = []
    figure_chunks = []
    table_chunks = []
    equation_chunks = []

    for chunk in chunks:

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

        text_vectors = await embed_texts([
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

        table_vectors = await embed_texts([
            chunk["content"]
            for chunk in table_chunks
        ])

        upsert_points(
            "table_chunks",
            table_chunks,
            table_vectors
        )

    if equation_chunks:

        equation_vectors = await embed_texts([
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

    query_vector = await embed_single(query)

    qdrant_filter = build_filter(
        filters
    )

    dense_results = qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter
    ).points

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

    model = get_reranker(settings.RERANKER_MODEL)
    pairs = [[query, text] for text in candidate_texts]
    
    scores = await asyncio.to_thread(model.predict, pairs)

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

            if overlap > 0.6:

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


DEFAULT_SEARCH_COLLECTIONS = ["text_chunks", "table_chunks", "equation_chunks"]

async def retrieve(
    query: str,
    project_id: str | None = None,
    top_k: int = 10,
    collections: list[str] | None = None
) -> list[dict]:

    if collections is None:
        collections = DEFAULT_SEARCH_COLLECTIONS

    logger.debug("rag.retrieve.started", query=query, collections=collections, top_k=top_k)
    rewritten_queries = await rewrite_query(
        query
    )

    all_results = []

    filters = {
        "project_id": project_id
    }

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

    reranked = await rerank_candidates(
        query=query,
        candidates=fused_results,
        top_n=top_k
    )

    compressed = compress_context(
        reranked,
        max_tokens=settings.MAX_CONTEXT_TOKENS
    )

    return compressed
