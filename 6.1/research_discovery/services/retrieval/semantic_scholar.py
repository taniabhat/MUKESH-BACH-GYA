"""
Semantic Scholar retrieval adapter.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from research_discovery.config.settings import (
    settings,
)
from research_discovery.core.runtime import (
    TokenBucket,
    api_retry,
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

BASE_URL = (
    settings.semantic_scholar.base_url
)

MAX_AUTHORS = 20

MAX_REFERENCES = 50

MAX_FIELDS_OF_STUDY = 10

TOKEN_BUCKET_CAPACITY = 2

TOKEN_BUCKET_REFILL_RATE = 1

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
    "url",
    "tldr",
]

FIELDS = ",".join(
    PAPER_FIELDS
)


class SemanticScholarAdapter:
    """
    Semantic Scholar retrieval adapter.
    """

    def __init__(self):

        self._semaphore = (
            asyncio.Semaphore(
                settings
                .semantic_scholar
                .rps_limit
            )
        )

        self._bucket = (
            TokenBucket(
                capacity=(
                    TOKEN_BUCKET_CAPACITY
                ),
                refill_rate=(
                    TOKEN_BUCKET_REFILL_RATE
                ),
            )
        )

        api_key = (
            settings
            .semantic_scholar
            .api_key
        )

        self._headers = (
            {
                "x-api-key": api_key
            }
            if api_key
            else {}
        )

    async def search(
        self,
        query: str,
        limit: int = 20,
    ) -> SearchResult:

        url = (
            f"{BASE_URL}/paper/search"
        )

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

            papers = (
                self._parse_results(
                    data=data,
                    query=query,
                )
            )

            total_found = data.get(
                "total",
                0,
            )

            logger.info(
                "SemanticScholar "
                "retrieved "
                "query='%s' "
                "papers=%s "
                "total=%s",
                query,
                len(papers),
                total_found,
            )

            return SearchResult(
                source=(
                    PaperSource
                    .SEMANTIC_SCHOLAR
                ),
                query=query,
                papers=papers,
                total_found=(
                    total_found
                ),
            )

        except Exception as exc:

            logger.exception(
                "SemanticScholar "
                "search failed "
                "query='%s'",
                query,
            )

            return SearchResult(
                source=(
                    PaperSource
                    .SEMANTIC_SCHOLAR
                ),
                query=query,
                papers=[],
                total_found=0,
                error=str(exc),
            )

    async def get_paper_details(
        self,
        paper_id: str,
    ) -> Optional[Paper]:

        url = (
            f"{BASE_URL}/paper/"
            f"{paper_id}"
        )

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
                query=(
                    f"detail:"
                    f"{paper_id}"
                ),
            )

        except Exception:

            logger.exception(
                "SemanticScholar "
                "detail fetch failed "
                "paper_id='%s'",
                paper_id,
            )

            return None

    async def fetch_references(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[Paper]:

        url = (
            f"{BASE_URL}/paper/"
            f"{paper_id}/references"
        )

        params = {
            "fields": FIELDS,
            "limit": limit,
        }

        try:

            data = await self._get(
                url=url,
                params=params,
            )

            papers = []

            for item in data.get(
                "data",
                [],
            ):

                cited_paper = (
                    item.get(
                        "citedPaper"
                    )
                )

                if not cited_paper:
                    continue

                paper = (
                    self._parse_paper(
                        raw=cited_paper,
                        query=(
                            "reference_expansion:"
                            f"{paper_id}"
                        ),
                    )
                )

                if paper:

                    paper.source = (
                        PaperSource
                        .CITATION_EXPANSION
                    )

                    papers.append(
                        paper
                    )

            return papers

        except Exception:

            logger.exception(
                "SemanticScholar "
                "reference fetch failed"
            )

            return []

    async def fetch_citations(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[Paper]:

        url = (
            f"{BASE_URL}/paper/"
            f"{paper_id}/citations"
        )

        params = {
            "fields": FIELDS,
            "limit": limit,
        }

        try:

            data = await self._get(
                url=url,
                params=params,
            )

            papers = []

            for item in data.get(
                "data",
                [],
            ):

                citing_paper = (
                    item.get(
                        "citingPaper"
                    )
                )

                if not citing_paper:
                    continue

                paper = (
                    self._parse_paper(
                        raw=citing_paper,
                        query=(
                            "citation_expansion:"
                            f"{paper_id}"
                        ),
                    )
                )

                if paper:

                    paper.source = (
                        PaperSource
                        .CITATION_EXPANSION
                    )

                    papers.append(
                        paper
                    )

            return papers

        except Exception:

            logger.exception(
                "SemanticScholar "
                "citation fetch failed"
            )

            return []

    @api_retry(
        max_attempts=(
            settings
            .semantic_scholar
            .max_retries
        )
    )
    async def _get(
        self,
        url: str,
        params: dict,
    ) -> dict:

        await self._bucket.acquire()

        async with self._semaphore:

            async with get_http_client(
                headers=self._headers
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

        for raw in data.get(
            "data",
            [],
        ):

            try:

                paper = self._parse_paper(
                    raw=raw,
                    query=query,
                )

                if paper:

                    papers.append(
                        paper
                    )

            except Exception:

                logger.exception(
                    "Failed parsing "
                    "SemanticScholar paper"
                )

        return papers

    def _parse_paper(
        self,
        raw: dict,
        query: str,
    ) -> Optional[Paper]:

        title = (
            raw.get(
                "title",
                ""
            )
            .strip()
        )

        if not title:
            return None

        return Paper(
            source=(
                PaperSource
                .SEMANTIC_SCHOLAR
            ),
            external_ids=(
                self._extract_external_ids(
                    raw
                )
            ),
            title=title,
            abstract=(
                raw.get("abstract")
                or self._extract_tldr(
                    raw
                )
            ),
            authors=(
                self._extract_authors(
                    raw
                )
            ),
            year=raw.get("year"),
            venue=raw.get("venue"),
            publication_date=(
                raw.get(
                    "publicationDate"
                )
            ),
            citation_count=(
                raw.get(
                    "citationCount",
                    0,
                )
            ),
            reference_count=(
                raw.get(
                    "referenceCount",
                    0,
                )
            ),
            fields_of_study=(
                self._extract_fields(
                    raw
                )
            ),
            is_open_access=(
                raw.get(
                    "isOpenAccess",
                    False,
                )
            ),
            pdf_url=(
                self._extract_pdf_url(
                    raw
                )
            ),
            landing_page_url=(
                raw.get("url")
            ),
            references=(
                self._extract_references(
                    raw
                )
            ),
            retrieved_from_queries=[
                query
            ],
        )

    @staticmethod
    def _extract_external_ids(
        raw: dict,
    ) -> ExternalIDs:

        external_ids = (
            raw.get(
                "externalIds"
            )
            or {}
        )

        doi = external_ids.get(
            "DOI"
        )

        if doi:

            doi = re.sub(
                r"^https?://doi\\.org/",
                "",
                doi,
            )

            doi = doi.lower()

        return ExternalIDs(
            doi=doi,
            arxiv=(
                external_ids.get(
                    "ArXiv"
                )
            ),
            semantic_scholar=(
                raw.get(
                    "paperId"
                )
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
                    author_id=(
                        author.get(
                            "authorId"
                        )
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
            "fieldsOfStudy",
            [],
        ):

            if field:

                fields.append(
                    field
                )

        for field in raw.get(
            "s2FieldsOfStudy",
            [],
        ):

            category = field.get(
                "category"
            )

            if category:

                fields.append(
                    category
                )

        unique_fields = []

        seen = set()

        for field in fields:

            normalized = (
                field.lower()
            )

            if normalized in seen:
                continue

            seen.add(normalized)

            unique_fields.append(
                field
            )

        return unique_fields[
            :MAX_FIELDS_OF_STUDY
        ]

    @staticmethod
    def _extract_pdf_url(
        raw: dict,
    ) -> Optional[str]:

        open_access_pdf = (
            raw.get(
                "openAccessPdf"
            )
            or {}
        )

        return open_access_pdf.get(
            "url"
        )

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
                reference.get(
                    "externalIds"
                )
                or {}
            )

            references.append(
                PaperReference(
                    title=reference.get(
                        "title"
                    ),
                    doi=(
                        external_ids.get(
                            "DOI"
                        )
                    ),
                )
            )

        return references

    @staticmethod
    def _extract_tldr(
        raw: dict,
    ) -> Optional[str]:

        tldr = raw.get(
            "tldr"
        )

        if not tldr:
            return None

        return tldr.get(
            "text"
        )