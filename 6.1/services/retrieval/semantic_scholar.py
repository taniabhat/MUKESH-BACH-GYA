"""
Semantic Scholar API Adapter.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import (
    TokenBucket,
    get_http_client,
    get_logger,
)
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

MAX_AUTHORS = 20
MAX_REFERENCES = 50
MAX_FIELDS_OF_STUDY = 10

TOKEN_BUCKET_CAPACITY = 10
TOKEN_BUCKET_REFILL_RATE = 2

PAPER_FIELDS = [
    "paperId",
    "externalIds",
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "publicationDate",
    "citationCount",
    "referenceCount",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "isOpenAccess",
    "openAccessPdf",
    "references",
    "citations",
    "tldr",
]

FIELDS = ",".join(PAPER_FIELDS)


class SemanticScholarAdapter:
    """Fetches papers from Semantic Scholar."""

    def __init__(self):

        self._semaphore = asyncio.Semaphore(
            settings.api.semantic_scholar_rps
        )

        self._bucket = TokenBucket(
            capacity=TOKEN_BUCKET_CAPACITY,
            refill_rate=TOKEN_BUCKET_REFILL_RATE,
        )

        api_key = (
            settings.api.semantic_scholar_api_key
        )

        self._headers = (
            {"x-api-key": api_key}
            if api_key
            else {}
        )

    async def search(
        self,
        query: str,
        limit: int = 20,
    ) -> SearchResult:

        url = f"{BASE_URL}/paper/search"

        params = {
            "query": query,
            "limit": limit,
            "fields": FIELDS,
        }

        try:
            data = await self._get(
                url=url,
                params=params,
            )

            papers = self._parse_results(
                data=data,
                query=query,
            )

            total_found = data.get(
                "total",
                0,
            )

            logger.info(
                "SemanticScholar query='%s' fetched=%s total=%s",
                query,
                len(papers),
                total_found,
            )

            return SearchResult(
                source=PaperSource.SEMANTIC_SCHOLAR,
                query=query,
                papers=papers,
                total_found=total_found,
            )

        except Exception:
            logger.exception(
                "SemanticScholar search failed query='%s'",
                query,
            )

            return SearchResult(
                source=PaperSource.SEMANTIC_SCHOLAR,
                query=query,
                papers=[],
                total_found=0,
            )

    async def get_paper_details(
        self,
        paper_id: str,
    ) -> Optional[Paper]:

        url = f"{BASE_URL}/paper/{paper_id}"

        params = {
            "fields": FIELDS,
        }

        try:
            raw = await self._get(
                url=url,
                params=params,
            )

            return self._parse_paper(
                raw=raw,
                query=f"detail:{paper_id}",
            )

        except Exception:
            logger.exception(
                "SemanticScholar detail fetch failed paper_id='%s'",
                paper_id,
            )

            return None

    async def _get(
        self,
        url: str,
        params: dict,
    ) -> dict:

        await self._bucket.acquire()

        async with self._semaphore:

            async with get_http_client(
                timeout=settings.api.http_timeout,
                headers=self._headers,
            ) as client:

                response = await client.get(
                    url,
                    params=params,
                )

                response.raise_for_status()

                return response.json()

    def _parse_results(
        self,
        data: dict,
        query: str,
    ) -> list[Paper]:

        papers = []

        for raw in data.get("data", []):

            try:
                paper = self._parse_paper(
                    raw=raw,
                    query=query,
                )

                if paper:
                    papers.append(paper)

            except Exception:
                logger.exception(
                    "Failed to parse SemanticScholar paper"
                )

        return papers

    def _parse_paper(
        self,
        raw: dict,
        query: str,
    ) -> Optional[Paper]:

        title = raw.get("title", "").strip()

        if not title:
            return None

        return Paper(
            source=PaperSource.SEMANTIC_SCHOLAR,
            external_ids=self._extract_external_ids(raw),
            title=title,
            abstract=raw.get("abstract"),
            authors=self._extract_authors(raw),
            year=raw.get("year"),
            venue=raw.get("venue"),
            publication_date=raw.get(
                "publicationDate"
            ),
            citation_count=raw.get(
                "citationCount",
                0,
            ),
            reference_count=raw.get(
                "referenceCount",
                0,
            ),
            fields_of_study=self._extract_fields(
                raw
            ),
            is_open_access=raw.get(
                "isOpenAccess",
                False,
            ),
            pdf_url=self._extract_pdf_url(raw),
            references=self._extract_references(
                raw
            ),
            retrieved_from_queries=[query],
        )

    @staticmethod
    def _extract_external_ids(
        raw: dict,
    ) -> ExternalIDs:

        external_ids = (
            raw.get("externalIds")
            or {}
        )

        return ExternalIDs(
            doi=external_ids.get("DOI"),
            arxiv=external_ids.get("ArXiv"),
            semantic_scholar=raw.get(
                "paperId"
            ),
        )

    @staticmethod
    def _extract_authors(
        raw: dict,
    ) -> list[Author]:

        authors = []

        for author in raw.get(
            "authors",
            [],
        )[:MAX_AUTHORS]:

            authors.append(
                Author(
                    name=author.get(
                        "name",
                        "",
                    ),
                    author_id=author.get(
                        "authorId"
                    ),
                )
            )

        return authors

    @staticmethod
    def _extract_fields(
        raw: dict,
    ) -> list[str]:

        fields = []

        for field in raw.get(
            "s2FieldsOfStudy",
            [],
        ):

            category = field.get(
                "category"
            )

            if category:
                fields.append(category)

        return fields[:MAX_FIELDS_OF_STUDY]

    @staticmethod
    def _extract_pdf_url(
        raw: dict,
    ) -> Optional[str]:

        open_access_pdf = (
            raw.get("openAccessPdf")
            or {}
        )

        return open_access_pdf.get("url")

    @staticmethod
    def _extract_references(
        raw: dict,
    ) -> list[PaperReference]:

        references = []

        for reference in raw.get(
            "references",
            [],
        )[:MAX_REFERENCES]:

            external_ids = (
                reference.get("externalIds")
                or {}
            )

            references.append(
                PaperReference(
                    title=reference.get(
                        "title"
                    ),
                    doi=external_ids.get(
                        "DOI"
                    ),
                )
            )

        return references