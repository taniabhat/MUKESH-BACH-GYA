"""
OpenAlex API Adapter — Primary retrieval source.

OpenAlex is the backbone of the system:
- Best metadata quality
- Free with generous quotas
- Rich references + citations
- Author graph
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_http_client, get_logger, api_retry
from research_discovery.models.paper import (
    Author,
    ExternalIDs,
    Paper,
    PaperReference,
    PaperSource,
    SearchResult,
)

logger = get_logger(__name__)

BASE_URL = settings.api.openalex_base_url
EMAIL = settings.api.openalex_email


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    # Strip URL prefix if present
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^https?://dx\.doi\.org/", "", doi)
    return doi.lower() if doi else None


def _parse_author(raw: dict) -> Author:
    name = raw.get("author", {}).get("display_name", "Unknown")
    author_id = raw.get("author", {}).get("id", "")
    affiliations = raw.get("institutions", [])
    affiliation = affiliations[0].get("display_name") if affiliations else None
    return Author(name=name, author_id=author_id, affiliation=affiliation)


def _parse_paper(raw: dict, query: str) -> Optional[Paper]:
    try:
        title = raw.get("title") or raw.get("display_name") or ""
        if not title.strip():
            return None

        # External IDs
        doi = _normalize_doi(raw.get("doi"))
        openalex_id = raw.get("id", "")

        # arXiv ID from locations
        arxiv_id = None
        for loc in raw.get("locations", []):
            source = loc.get("source") or {}
            if "arxiv" in source.get("host_organization_lineage_names", []):
                url = loc.get("landing_page_url", "")
                m = re.search(r"arxiv\.org/abs/([0-9.]+)", url)
                if m:
                    arxiv_id = m.group(1)
                    break

        external_ids = ExternalIDs(
            doi=doi,
            arxiv=arxiv_id,
            openalex=openalex_id,
        )

        # Authors
        authors = [_parse_author(a) for a in raw.get("authorships", [])]

        # Venue
        primary_loc = raw.get("primary_location") or {}
        source_info = primary_loc.get("source") or {}
        venue = source_info.get("display_name")

        # Concepts → fields of study
        fields = [c["display_name"] for c in raw.get("concepts", []) if c.get("score", 0) > 0.4]

        # References
        refs = []
        for ref_id in raw.get("referenced_works", [])[:50]:
            refs.append(PaperReference(openalex_id=ref_id))

        # Open access
        oa_status = (raw.get("open_access") or {}).get("is_oa", False)
        pdf_url = None
        for loc in raw.get("locations", []):
            if loc.get("is_oa") and loc.get("pdf_url"):
                pdf_url = loc["pdf_url"]
                break

        year = raw.get("publication_year")
        pub_date = raw.get("publication_date")

        return Paper(
            source=PaperSource.OPENALEX,
            external_ids=external_ids,
            title=title,
            abstract=raw.get("abstract"),
            authors=authors,
            year=year,
            venue=venue,
            publication_date=pub_date,
            citation_count=raw.get("cited_by_count", 0),
            reference_count=len(raw.get("referenced_works", [])),
            fields_of_study=fields[:10],
            keywords=raw.get("keywords", [])[:20],
            pdf_url=pdf_url,
            landing_page_url=raw.get("landing_page_url"),
            is_open_access=oa_status,
            references=refs,
            retrieved_from_queries=[query],
        )
    except Exception as exc:
        logger.debug(f"OpenAlex paper parse error: {exc}")
        return None


class OpenAlexAdapter:
    """Fetches papers from OpenAlex."""

    def __init__(self):
        self.base_url = BASE_URL
        # Per-docs: provide email for "polite pool" (faster, higher limit)
        self.params_base = {"mailto": EMAIL}

    async def search(
        self,
        query: str,
        per_page: int = 25,
        filter_str: Optional[str] = None,
    ) -> SearchResult:
        """Full-text search on OpenAlex works."""
        params = {
            **self.params_base,
            "search": query,
            "per-page": per_page,
            "select": (
                "id,title,abstract,authorships,publication_year,publication_date,"
                "doi,open_access,locations,primary_location,cited_by_count,"
                "referenced_works,concepts,keywords,landing_page_url"
            ),
        }
        if filter_str:
            params["filter"] = filter_str

        url = f"{self.base_url}/works"
        papers: list[Paper] = []

        try:
            async with get_http_client(timeout=settings.api.http_timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            total = data.get("meta", {}).get("count", 0)

            for raw in results:
                paper = _parse_paper(raw, query)
                if paper:
                    papers.append(paper)

            logger.info(f"OpenAlex '{query}': {len(papers)}/{total} papers retrieved")
            return SearchResult(
                source=PaperSource.OPENALEX,
                query=query,
                papers=papers,
                total_found=total,
            )

        except Exception as exc:
            logger.error(f"OpenAlex search error for '{query}': {exc}")
            return SearchResult(
                source=PaperSource.OPENALEX,
                query=query,
                papers=[],
                total_found=0,
                error=str(exc),
            )

    async def fetch_by_doi(self, doi: str) -> Optional[Paper]:
        """Fetch a single paper by DOI for enrichment."""
        encoded = urllib.parse.quote(doi, safe="")
        url = f"{self.base_url}/works/https://doi.org/{encoded}"
        try:
            async with get_http_client() as client:
                resp = await client.get(url, params=self.params_base)
                resp.raise_for_status()
                raw = resp.json()
            return _parse_paper(raw, f"doi:{doi}")
        except Exception as exc:
            logger.debug(f"OpenAlex DOI fetch failed for {doi}: {exc}")
            return None

    async def fetch_citations(self, openalex_id: str) -> list[Paper]:
        """Fetch papers that cite this work."""
        clean_id = openalex_id.replace("https://openalex.org/", "")
        params = {
            **self.params_base,
            "filter": f"cites:{clean_id}",
            "per-page": 50,
            "select": "id,title,abstract,authorships,publication_year,doi,cited_by_count",
        }
        papers = []
        try:
            async with get_http_client() as client:
                resp = await client.get(f"{self.base_url}/works", params=params)
                resp.raise_for_status()
                data = resp.json()
            for raw in data.get("results", []):
                p = _parse_paper(raw, f"citation_expansion:{clean_id}")
                if p:
                    p.source = PaperSource.CITATION_EXPANSION
                    papers.append(p)
        except Exception as exc:
            logger.debug(f"OpenAlex citation fetch failed: {exc}")
        return papers

    async def fetch_references(self, openalex_id: str) -> list[Paper]:
        """Fetch papers this work references."""
        clean_id = openalex_id.replace("https://openalex.org/", "")
        params = {
            **self.params_base,
            "filter": f"cited_by:{clean_id}",
            "per-page": 50,
            "select": "id,title,abstract,authorships,publication_year,doi,cited_by_count",
        }
        papers = []
        try:
            async with get_http_client() as client:
                resp = await client.get(f"{self.base_url}/works", params=params)
                resp.raise_for_status()
                data = resp.json()
            for raw in data.get("results", []):
                p = _parse_paper(raw, f"reference_expansion:{clean_id}")
                if p:
                    p.source = PaperSource.CITATION_EXPANSION
                    papers.append(p)
        except Exception as exc:
            logger.debug(f"OpenAlex reference fetch failed: {exc}")
        return papers