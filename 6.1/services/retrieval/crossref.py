"""
CrossRef Adapter — Metadata enrichment and DOI resolution.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import (
    get_http_client,
    get_logger,
)
from research_discovery.models.paper import (
    Author,
    ExternalIDs,
    Paper,
    PaperSource,
)

logger = get_logger(__name__)

BASE_URL = settings.api.crossref_base_url
MAILTO = settings.api.crossref_mailto

POLITE_HEADERS = {
    "User-Agent": (
        f"ResearchDiscoveryBot/1.0 "
        f"(mailto:{MAILTO})"
    )
}

MAX_AUTHORS = 20


class CrossRefAdapter:
    """Metadata enrichment via CrossRef."""

    async def fetch_by_doi(
        self,
        doi: str,
    ) -> Optional[Paper]:

        try:
            encoded_doi = urllib.parse.quote(
                doi,
                safe="",
            )

            url = f"{BASE_URL}/{encoded_doi}"

            data = await self._fetch_json(url)

            raw = data.get("message", {})

            return self._parse_work(
                raw,
                query=f"doi:{doi}",
            )

        except Exception:
            logger.exception(
                "CrossRef DOI fetch failed doi='%s'",
                doi,
            )
            return None

    async def search(
        self,
        query: str,
        rows: int = 10,
    ) -> list[Paper]:

        params = {
            "query": query,
            "rows": rows,
            "select": (
                "DOI,title,author,published,"
                "container-title,abstract,"
                "is-referenced-by-count,"
                "URL,link"
            ),
            "mailto": MAILTO,
        }

        papers: list[Paper] = []

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
                    paper = self._parse_work(
                        item,
                        query=query,
                    )

                    if paper:
                        papers.append(paper)

                except Exception:
                    logger.exception(
                        "Failed to parse CrossRef item"
                    )

        except Exception:
            logger.exception(
                "CrossRef search failed query='%s'",
                query,
            )

        return papers

    async def enrich_papers(
        self,
        papers: list[Paper],
    ) -> list[Paper]:

        enrichment_tasks = []
        paper_indices = []

        for index, paper in enumerate(papers):

            if not self._should_enrich(paper):
                continue

            doi = paper.get_best_doi()

            if not doi:
                continue

            enrichment_tasks.append(
                self.fetch_by_doi(doi)
            )

            paper_indices.append(index)

        if not enrichment_tasks:
            return papers

        results = await asyncio.gather(
            *enrichment_tasks,
            return_exceptions=True,
        )

        enriched_papers = papers.copy()

        for index, result in zip(
            paper_indices,
            results,
        ):

            if not isinstance(result, Paper):
                continue

            enriched_papers[index] = (
                self._merge_paper_metadata(
                    original=enriched_papers[index],
                    enrichment=result,
                )
            )

        return enriched_papers

    async def _fetch_json(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> dict:

        async with get_http_client(
            timeout=settings.api.http_timeout,
            headers=POLITE_HEADERS,
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

        return any([
            not paper.venue,
            not paper.year,
            not paper.abstract,
            not paper.authors,
        ])

    def _parse_work(
        self,
        raw: dict,
        query: str = "",
    ) -> Optional[Paper]:

        title = self._extract_title(raw)

        if not title:
            return None

        return Paper(
            source=PaperSource.CROSSREF,
            external_ids=ExternalIDs(
                doi=self._extract_doi(raw),
            ),
            title=title,
            abstract=self._extract_abstract(raw),
            authors=self._extract_authors(raw),
            year=self._extract_year(raw),
            venue=self._extract_venue(raw),
            citation_count=self._extract_citation_count(raw),
            pdf_url=self._extract_pdf_url(raw),
            landing_page_url=raw.get("URL"),
            retrieved_from_queries=(
                [query] if query else []
            ),
        )

    @staticmethod
    def _extract_title(
        raw: dict,
    ) -> str:

        titles = raw.get("title", [])

        if not titles:
            return ""

        return titles[0].strip()

    @staticmethod
    def _extract_doi(
        raw: dict,
    ) -> Optional[str]:

        doi = raw.get("DOI", "")

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

            given = author.get("given", "")
            family = author.get("family", "")

            full_name = (
                f"{given} {family}".strip()
                or "Unknown"
            )

            affiliations = author.get(
                "affiliation",
                [],
            )

            affiliation = (
                affiliations[0].get("name")
                if affiliations
                else None
            )

            authors.append(
                Author(
                    name=full_name,
                    affiliation=affiliation,
                    orcid=author.get("ORCID"),
                )
            )

        return authors

    @staticmethod
    def _extract_year(
        raw: dict,
    ) -> Optional[int]:

        date_fields = (
            "published",
            "published-print",
            "published-online",
        )

        for field in date_fields:

            date_parts = (
                raw.get(field, {})
                .get("date-parts", [[]])[0]
            )

            if date_parts:
                return int(date_parts[0])

        return None

    @staticmethod
    def _extract_venue(
        raw: dict,
    ) -> Optional[str]:

        container_titles = raw.get(
            "container-title",
            [],
        )

        if not container_titles:
            return None

        return container_titles[0]

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

        abstract = raw.get("abstract")

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

        for link in raw.get("link", []):

            if (
                link.get("content-type")
                == "application/pdf"
            ):
                return link.get("URL")

        return None

    @staticmethod
    def _merge_paper_metadata(
        original: Paper,
        enrichment: Paper,
    ) -> Paper:

        if not original.venue:
            original.venue = enrichment.venue

        if not original.year:
            original.year = enrichment.year

        if not original.abstract:
            original.abstract = enrichment.abstract

        if not original.authors:
            original.authors = enrichment.authors

        return original