"""
arXiv API Adapter — Freshest preprints, essential for fast-moving AI domains.
Uses the Atom feed API (no auth required).
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

# arXiv Atom feed namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _text(element, tag: str, ns: Optional[str] = "atom") -> str:
    prefix = f"{{{NS[ns]}}}" if ns else ""
    child = element.find(f"{prefix}{tag}")
    return child.text.strip() if child is not None and child.text else ""


def _parse_entry(entry: ET.Element, query: str) -> Optional[Paper]:
    try:
        title = _text(entry, "title").replace("\n", " ").strip()
        if not title:
            return None

        abstract = _text(entry, "summary").replace("\n", " ").strip()

        # arXiv ID from <id> URL like http://arxiv.org/abs/2401.12345v1
        raw_id = _text(entry, "id")
        arxiv_match = re.search(r"arxiv\.org/abs/([^\s]+)", raw_id)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        if arxiv_id:
            # Strip version suffix
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        # Authors
        authors = []
        for author_el in entry.findall(f"{{{NS['atom']}}}author"):
            name_el = author_el.find(f"{{{NS['atom']}}}name")
            if name_el is not None and name_el.text:
                authors.append(Author(name=name_el.text.strip()))

        # Published date
        published = _text(entry, "published")
        year = None
        if published:
            m = re.match(r"(\d{4})", published)
            if m:
                year = int(m.group(1))

        # DOI (sometimes present in arxiv:doi)
        doi_el = entry.find(f"{{{NS['arxiv']}}}doi")
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        # Category → fields of study
        fields = []
        for cat in entry.findall(f"{{{NS['atom']}}}category"):
            term = cat.get("term", "")
            if term:
                fields.append(term)

        pdf_url = None
        landing_url = None
        for link in entry.findall(f"{{{NS['atom']}}}link"):
            rel = link.get("rel", "")
            href = link.get("href", "")
            if link.get("type") == "application/pdf":
                pdf_url = href
            elif rel == "alternate":
                landing_url = href

        if arxiv_id and not landing_url:
            landing_url = f"https://arxiv.org/abs/{arxiv_id}"
        if arxiv_id and not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return Paper(
            source=PaperSource.ARXIV,
            external_ids=ExternalIDs(doi=doi, arxiv=arxiv_id),
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            publication_date=published[:10] if published else None,
            fields_of_study=fields[:10],
            pdf_url=pdf_url,
            landing_page_url=landing_url,
            is_open_access=True,  # arXiv is always OA
            retrieved_from_queries=[query],
        )
    except Exception as exc:
        logger.debug(f"arXiv parse error: {exc}")
        return None


class ArxivAdapter:
    """Fetches papers from arXiv via the Atom feed API."""

    async def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
    ) -> SearchResult:
        """
        Args:
            query: search query
            max_results: max papers to fetch
            sort_by: relevance | lastUpdatedDate | submittedDate
        """
        params = {
            "search_query": f"all:{urllib.parse.quote(query)}",
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        url = BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        papers: list[Paper] = []

        try:
            async with get_http_client(
                timeout=settings.api.http_timeout,
                headers={"Accept": "application/atom+xml"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                xml_text = resp.text

            root = ET.fromstring(xml_text)
            entries = root.findall(f"{{{NS['atom']}}}entry")

            for entry in entries:
                p = _parse_entry(entry, query)
                if p:
                    papers.append(p)

            # Extract total from opensearch:totalResults if present
            total_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
            total = int(total_el.text) if total_el is not None and total_el.text else len(papers)

            logger.info(f"arXiv '{query}': {len(papers)}/{total} papers")
            return SearchResult(
                source=PaperSource.ARXIV,
                query=query,
                papers=papers,
                total_found=total,
            )

        except Exception as exc:
            logger.error(f"arXiv search error for '{query}': {exc}")
            return SearchResult(
                source=PaperSource.ARXIV,
                query=query,
                papers=[],
                total_found=0,
                error=str(exc),
            )