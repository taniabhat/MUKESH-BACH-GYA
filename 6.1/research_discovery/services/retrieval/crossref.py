"""
CrossRef retrieval and metadata enrichment adapter.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import Optional

from research_discovery.config.settings import (
    settings,
)
from research_discovery.core.runtime import (
    api_retry,
    get_http_client,
    get_logger,
)
from research_discovery.models.paper import (
    Author,
    ExternalIDs,
    Paper,
    PaperSource,
    SearchResult,
)

logger = get_logger(__name__)

BASE_URL = settings.crossref.base_url

MAILTO = settings.crossref.mailto

POLITE_HEADERS = {
    "User-Agent": (
        f"ResearchDiscoveryBot/1.0 "
        f"(mailto:{MAILTO})"
    )
}

MAX_AUTHORS = 20


class CrossRefAdapter:
    """
    CrossRef retrieval and enrichment adapter.
    """

    async def search(
        self,
        query: str,
        limit: int = 20,
    ) -> SearchResult:

        params = {
            "query": query,
            "rows": limit,
            "mailto": MAILTO,
            "select": (
                "DOI,title,author,published,"
                "published-print,published-online,"
                "container-title,abstract,"
                "is-referenced-by-count,"
                "URL,link"
            ),
        }

        papers = []

        try:

            data = await self._fetch_json(
                BASE_URL,
                params=params,
            )

            items = (
                data.get("message", {})
                .get("items", [])
            )

            for item in items:

                try:

                    paper = (
                        self._parse_work(
                            raw=item,
                            query=query,
                        )
                    )

                    if paper:
                        papers.append(
                            paper
                        )

                except Exception:

                    logger.exception(
                        "Failed parsing "
                        "CrossRef item"
                    )

            logger.info(
                "CrossRef retrieved "
                "query='%s' papers=%s",
                query,
                len(papers),
            )

            return SearchResult(
                source=PaperSource.CROSSREF,
                query=query,
                papers=papers,
                total_found=len(papers),
            )

        except Exception as exc:

            logger.exception(
                "CrossRef search failed "
                "query='%s'",
                query,
            )

            return SearchResult(
                source=PaperSource.CROSSREF,
                query=query,
                papers=[],
                total_found=0,
                error=str(exc),
            )

    async def fetch_by_doi(
        self,
        doi: str,
    ) -> Optional[Paper]:

        try:

            encoded_doi = (
                urllib.parse.quote(
                    doi,
                    safe="",
                )
            )

            url = (
                f"{BASE_URL}/"
                f"{encoded_doi}"
            )

            data = await self._fetch_json(
                url
            )

            raw = data.get(
                "message",
                {},
            )

            return self._parse_work(
                raw=raw,
                query=f"doi:{doi}",
            )

        except Exception:

            logger.exception(
                "CrossRef DOI fetch failed "
                "doi='%s'",
                doi,
            )

            return None

    async def enrich_papers(
        self,
        papers: list[Paper],
    ) -> list[Paper]:

        tasks = []

        indices = []

        for index, paper in enumerate(
            papers
        ):

            if not self._should_enrich(
                paper
            ):
                continue

            doi = (
                paper.get_best_doi()
            )

            if not doi:
                continue

            tasks.append(
                self.fetch_by_doi(
                    doi
                )
            )

            indices.append(index)

        if not tasks:
            return papers

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        enriched = papers.copy()

        for index, result in zip(
            indices,
            results,
        ):

            if not isinstance(
                result,
                Paper,
            ):
                continue

            enriched[index] = (
                self._merge_paper_metadata(
                    original=(
                        enriched[index]
                    ),
                    enrichment=result,
                )
            )

        return enriched

    @api_retry(
        max_attempts=(
            settings.http.max_retries
        )
    )
    async def _fetch_json(
        self,
        url: str,
        params: Optional[
            dict
        ] = None,
    ) -> dict:

        async with get_http_client(
            headers=POLITE_HEADERS
        ) as client:

            response = await client.get(
                url,
                params=params,
            )

            response.raise_for_status()

            return response.json()

    @staticmethod
    def _should_enrich(
        paper: Paper,
    ) -> bool:

        return any(
            [
                not paper.venue,
                not paper.year,
                not paper.abstract,
                not paper.authors,
            ]
        )

    def _parse_work(
        self,
        raw: dict,
        query: str = "",
    ) -> Optional[Paper]:

        title = self._extract_title(
            raw
        )

        if not title:
            return None

        return Paper(
            source=PaperSource.CROSSREF,
            external_ids=ExternalIDs(
                doi=self._extract_doi(
                    raw
                )
            ),
            title=title,
            abstract=self._extract_abstract(
                raw
            ),
            authors=self._extract_authors(
                raw
            ),
            year=self._extract_year(
                raw
            ),
            venue=self._extract_venue(
                raw
            ),
            citation_count=(
                self._extract_citation_count(
                    raw
                )
            ),
            pdf_url=self._extract_pdf_url(
                raw
            ),
            landing_page_url=raw.get(
                "URL"
            ),
            retrieved_from_queries=(
                [query]
                if query
                else []
            ),
        )

    @staticmethod
    def _extract_title(
        raw: dict,
    ) -> str:

        titles = raw.get(
            "title",
            [],
        )

        if not titles:
            return ""

        return titles[0].strip()

    @staticmethod
    def _extract_doi(
        raw: dict,
    ) -> Optional[str]:

        doi = raw.get(
            "DOI",
            "",
        )

        doi = doi.lower().strip()

        return doi or None

    @staticmethod
    def _extract_authors(
        raw: dict,
    ) -> list[Author]:

        authors = []

        for author in raw.get(
            "author",
            [],
        )[:MAX_AUTHORS]:

            given = author.get(
                "given",
                "",
            )

            family = author.get(
                "family",
                "",
            )

            full_name = (
                f"{given} {family}"
                .strip()
                or "Unknown"
            )

            affiliations = author.get(
                "affiliation",
                [],
            )

            affiliation = (
                affiliations[0].get(
                    "name"
                )
                if affiliations
                else None
            )

            authors.append(
                Author(
                    name=full_name,
                    affiliation=(
                        affiliation
                    ),
                    orcid=author.get(
                        "ORCID"
                    ),
                )
            )

        return authors

    @staticmethod
    def _extract_year(
        raw: dict,
    ) -> Optional[int]:

        date_fields = [
            "published",
            "published-print",
            "published-online",
        ]

        for field in date_fields:

            date_parts = (
                raw.get(field, {})
                .get(
                    "date-parts",
                    [[]],
                )[0]
            )

            if date_parts:

                try:
                    return int(
                        date_parts[0]
                    )

                except Exception:
                    pass

        return None

    @staticmethod
    def _extract_venue(
        raw: dict,
    ) -> Optional[str]:

        venues = raw.get(
            "container-title",
            [],
        )

        if not venues:
            return None

        return venues[0]

    @staticmethod
    def _extract_citation_count(
        raw: dict,
    ) -> int:

        return raw.get(
            "is-referenced-by-count",
            0,
        )

    @staticmethod
    def _extract_abstract(
        raw: dict,
    ) -> Optional[str]:

        abstract = raw.get(
            "abstract"
        )

        if not abstract:
            return None

        cleaned = re.sub(
            r"<[^>]+>",
            "",
            abstract,
        ).strip()

        return cleaned or None

    @staticmethod
    def _extract_pdf_url(
        raw: dict,
    ) -> Optional[str]:

        for link in raw.get(
            "link",
            [],
        ):

            if (
                link.get(
                    "content-type"
                )
                == "application/pdf"
            ):

                return link.get(
                    "URL"
                )

        return None

    @staticmethod
    def _merge_paper_metadata(
        original: Paper,
        enrichment: Paper,
    ) -> Paper:

        merged = (
            original.model_copy(
                deep=True
            )
        )

        if not merged.venue:
            merged.venue = (
                enrichment.venue
            )

        if not merged.year:
            merged.year = (
                enrichment.year
            )

        if not merged.abstract:
            merged.abstract = (
                enrichment.abstract
            )

        if not merged.authors:
            merged.authors = (
                enrichment.authors
            )

        if (
            merged.citation_count
            == 0
        ):
            merged.citation_count = (
                enrichment.citation_count
            )

        return merged