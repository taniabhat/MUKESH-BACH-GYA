"""
arXiv API Adapter — Fetches research papers using the Atom feed API.
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_http_client, get_logger
from research_discovery.models.paper import (
    Author,
    ExternalIDs,
    Paper,
    PaperSource,
    SearchResult,
)

logger = get_logger(__name__)

BASE_URL = settings.api.arxiv_base_url

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

MAX_FIELDS_OF_STUDY = 10


class ArxivAdapter:
    """Adapter for fetching papers from arXiv."""

    async def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
    ) -> SearchResult:

        try:
            xml_text = await self._fetch_results(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
            )

            papers, total_found = self._parse_response(
                xml_text=xml_text,
                query=query,
            )

            logger.info(
                "arXiv query='%s' fetched=%s total=%s",
                query,
                len(papers),
                total_found,
            )

            return SearchResult(
                source=PaperSource.ARXIV,
                query=query,
                papers=papers,
                total_found=total_found,
            )

        except Exception as exc:
            logger.exception(
                "arXiv search failed for query='%s'",
                query,
            )

            return SearchResult(
                source=PaperSource.ARXIV,
                query=query,
                papers=[],
                total_found=0,
                error=str(exc),
            )

    async def _fetch_results(
        self,
        query: str,
        max_results: int,
        sort_by: str,
    ) -> str:

        params = {
            "search_query": f"all:{query}",
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        url = (
            f"{BASE_URL}?"
            f"{urllib.parse.urlencode(params)}"
        )

        async with get_http_client(
            timeout=settings.api.http_timeout,
            headers={"Accept": "application/atom+xml"},
        ) as client:

            response = await client.get(url)
            response.raise_for_status()

            return response.text

    def _parse_response(
        self,
        xml_text: str,
        query: str,
    ) -> tuple[list[Paper], int]:

        root = ET.fromstring(xml_text)

        entries = root.findall(
            f"{{{ATOM_NS}}}entry"
        )

        papers: list[Paper] = []

        for entry in entries:
            try:
                paper = self._parse_entry(
                    entry=entry,
                    query=query,
                )

                if paper:
                    papers.append(paper)

            except Exception:
                logger.exception(
                    "Failed to parse arXiv entry"
                )

        total_found = self._extract_total_results(root)

        return papers, total_found

    def _parse_entry(
        self,
        entry: ET.Element,
        query: str,
    ) -> Optional[Paper]:

        title = self._clean_text(
            self._get_text(entry, "title")
        )

        if not title:
            return None

        abstract = self._clean_text(
            self._get_text(entry, "summary")
        )

        arxiv_id = self._extract_arxiv_id(entry)

        publication_date = self._get_text(
            entry,
            "published",
        )

        year = self._extract_year(
            publication_date
        )

        doi = self._extract_doi(entry)

        authors = self._extract_authors(entry)

        fields_of_study = self._extract_fields(entry)

        pdf_url, landing_page_url = (
            self._extract_urls(
                entry,
                arxiv_id,
            )
        )

        return Paper(
            source=PaperSource.ARXIV,
            external_ids=ExternalIDs(
                doi=doi,
                arxiv=arxiv_id,
            ),
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            publication_date=(
                publication_date[:10]
                if publication_date
                else None
            ),
            fields_of_study=(
                fields_of_study[:MAX_FIELDS_OF_STUDY]
            ),
            pdf_url=pdf_url,
            landing_page_url=landing_page_url,
            is_open_access=True,
            retrieved_from_queries=[query],
        )

    def _get_text(
        self,
        element: ET.Element,
        tag: str,
        namespace: str = ATOM_NS,
    ) -> str:

        child = element.find(
            f"{{{namespace}}}{tag}"
        )

        if child is None or not child.text:
            return ""

        return child.text.strip()

    @staticmethod
    def _clean_text(value: str) -> str:
        return value.replace("\n", " ").strip()

    def _extract_arxiv_id(
        self,
        entry: ET.Element,
    ) -> Optional[str]:

        raw_id = self._get_text(entry, "id")

        match = re.search(
            r"arxiv\.org/abs/([^\s]+)",
            raw_id,
        )

        if not match:
            return None

        return re.sub(
            r"v\d+$",
            "",
            match.group(1),
        )

    @staticmethod
    def _extract_year(
        published_date: str,
    ) -> Optional[int]:

        if not published_date:
            return None

        match = re.match(
            r"(\d{4})",
            published_date,
        )

        return (
            int(match.group(1))
            if match
            else None
        )

    def _extract_doi(
        self,
        entry: ET.Element,
    ) -> Optional[str]:

        doi_element = entry.find(
            f"{{{ARXIV_NS}}}doi"
        )

        if (
            doi_element is None
            or not doi_element.text
        ):
            return None

        return doi_element.text.strip()

    def _extract_authors(
        self,
        entry: ET.Element,
    ) -> list[Author]:

        authors = []

        for author_element in entry.findall(
            f"{{{ATOM_NS}}}author"
        ):

            name_element = author_element.find(
                f"{{{ATOM_NS}}}name"
            )

            if (
                name_element is not None
                and name_element.text
            ):
                authors.append(
                    Author(
                        name=name_element.text.strip()
                    )
                )

        return authors

    def _extract_fields(
        self,
        entry: ET.Element,
    ) -> list[str]:

        fields = []

        for category in entry.findall(
            f"{{{ATOM_NS}}}category"
        ):

            term = category.get("term")

            if term:
                fields.append(term)

        return fields

    def _extract_urls(
        self,
        entry: ET.Element,
        arxiv_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:

        pdf_url = None
        landing_url = None

        for link in entry.findall(
            f"{{{ATOM_NS}}}link"
        ):

            rel = link.get("rel", "")
            href = link.get("href", "")

            if (
                link.get("type")
                == "application/pdf"
            ):
                pdf_url = href

            elif rel == "alternate":
                landing_url = href

        if arxiv_id:

            if not pdf_url:
                pdf_url = (
                    f"https://arxiv.org/pdf/"
                    f"{arxiv_id}.pdf"
                )

            if not landing_url:
                landing_url = (
                    f"https://arxiv.org/abs/"
                    f"{arxiv_id}"
                )

        return pdf_url, landing_url

    @staticmethod
    def _extract_total_results(
        root: ET.Element,
    ) -> int:

        total_element = root.find(
            f"{{{OPENSEARCH_NS}}}totalResults"
        )

        if (
            total_element is None
            or not total_element.text
        ):
            return 0

        try:
            return int(total_element.text)

        except ValueError:
            logger.warning(
                "Invalid totalResults value='%s'",
                total_element.text,
            )
            return 0