"""
Semantic Scholar API Adapter.

Used for:
- Citation counts
- Influential citations
- TLDRs
- Cross-referencing

Rate-limited aggressively via semaphore (5 concurrent requests).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_http_client, get_logger, TokenBucket
from research_discovery.models.paper import (
    Author,
    ExternalIDs,
    Paper,
    PaperReference,
    PaperSource,
    SearchResult,
)

logger = get_logger(__name__)

BASE_URL = settings.api.semantic_scholar_base_url

FIELDS = (
    "paperId,externalIds,title,abstract,authors,year,venue,publicationDate,"
    "citationCount,referenceCount,fieldsOfStudy,s2FieldsOfStudy,isOpenAccess,"
    "openAccessPdf,references,citations,tldr"
)


def _parse_paper(raw: dict, query: str) -> Optional[Paper]:
    try:
        title = raw.get("title", "")
        if not title:
            return None

        ext = raw.get("externalIds") or {}
        external_ids = ExternalIDs(
            doi=ext.get("DOI"),
            arxiv=ext.get("ArXiv"),
            semantic_scholar=raw.get("paperId"),
        )

        authors = []
        for a in raw.get("authors", [])[:20]:
            authors.append(Author(
                name=a.get("name", ""),
                author_id=a.get("authorId"),
            ))

        fields = []
        for f in raw.get("s2FieldsOfStudy", []):
            if f.get("category"):
                fields.append(f["category"])

        refs: list[PaperReference] = []
        for r in (raw.get("references") or [])[:50]:
            refs.append(PaperReference(
                title=r.get("title"),
                doi=(r.get("externalIds") or {}).get("DOI"),
            ))

        oa_pdf = raw.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url")

        return Paper(
            source=PaperSource.SEMANTIC_SCHOLAR,
            external_ids=external_ids,
            title=title,
            abstract=raw.get("abstract"),
            authors=authors,
            year=raw.get("year"),
            venue=raw.get("venue"),
            publication_date=raw.get("publicationDate"),
            citation_count=raw.get("citationCount", 0),
            reference_count=raw.get("referenceCount", 0),
            fields_of_study=fields[:10],
            is_open_access=raw.get("isOpenAccess", False),
            pdf_url=pdf_url,
            references=refs,
            retrieved_from_queries=[query],
        )
    except Exception as exc:
        logger.debug(f"S2 parse error: {exc}")
        return None


class SemanticScholarAdapter:
    """Fetches papers from Semantic Scholar."""

    def __init__(self):
        self._semaphore = asyncio.Semaphore(settings.api.semantic_scholar_rps)
        self._bucket = TokenBucket(capacity=10, refill_rate=2)  # 2 req/sec
        api_key = settings.api.semantic_scholar_api_key
        self._headers = {"x-api-key": api_key} if api_key else {}

    async def _get(self, url: str, params: dict) -> dict:
        await self._bucket.acquire()
        async with self._semaphore:
            async with get_http_client(
                timeout=settings.api.http_timeout,
                headers=self._headers,
            ) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()

    async def search(self, query: str, limit: int = 20) -> SearchResult:
        url = f"{BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": FIELDS,
        }
        papers: list[Paper] = []

        try:
            data = await self._get(url, params)
            for raw in data.get("data", []):
                p = _parse_paper(raw, query)
                if p:
                    papers.append(p)

            total = data.get("total", 0)
            logger.info(f"S2 '{query}': {len(papers)}/{total} papers")
            return SearchResult(
                source=PaperSource.SEMANTIC_SCHOLAR,
                query=query,
                papers=papers,
                total_found=total,
            )

        except Exception as exc:
            logger.error(f"S2 search error for '{query}': {exc}")
            return SearchResult(
                source=PaperSource.SEMANTIC_SCHOLAR,
                query=query,
                papers=[],
                total_found=0,
                error=str(exc),
            )

    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Fetch full details for a single paper (S2 paper ID or DOI)."""
        url = f"{BASE_URL}/paper/{paper_id}"
        params = {"fields": FIELDS}
        try:
            raw = await self._get(url, params)
            return _parse_paper(raw, f"detail:{paper_id}")
        except Exception as exc:
            logger.debug(f"S2 detail fetch failed {paper_id}: {exc}")
            return None