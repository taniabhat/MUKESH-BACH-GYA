"""
CrossRef Adapter — Metadata repair and DOI resolver.

Not used for primary retrieval. Used to:
- Fill missing DOIs
- Clean up venue names
- Recover publication dates
- Validate metadata

Always include User-Agent with mailto to avoid throttling.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_http_client, get_logger
from research_discovery.models.paper import Author, ExternalIDs, Paper, PaperSource

logger = get_logger(__name__)

BASE_URL = settings.api.crossref_base_url
MAILTO = settings.api.crossref_mailto
POLITE_HEADERS = {
    "User-Agent": f"ResearchDiscoveryBot/1.0 (mailto:{MAILTO})"
}


def _parse_crossref_work(raw: dict, query: str = "") -> Optional[Paper]:
    try:
        title_list = raw.get("title", [])
        title = title_list[0] if title_list else ""
        if not title:
            return None

        doi = raw.get("DOI", "").lower().strip()

        # Authors
        authors = []
        for a in raw.get("author", [])[:20]:
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip() or "Unknown"
            affils = a.get("affiliation", [])
            affil = affils[0].get("name") if affils else None
            authors.append(Author(name=name, affiliation=affil, orcid=a.get("ORCID")))

        # Year
        year = None
        for date_field in ("published", "published-print", "published-online"):
            dp = raw.get(date_field, {}).get("date-parts", [[]])[0]
            if dp:
                year = int(dp[0])
                break

        # Venue
        container = raw.get("container-title", [])
        venue = container[0] if container else None

        # Citation count (not always present in CrossRef)
        cites = raw.get("is-referenced-by-count", 0)

        # Abstract
        abstract = raw.get("abstract", "")
        if abstract:
            # CrossRef sometimes includes JATS XML tags
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        # PDF
        links = raw.get("link", [])
        pdf_url = None
        for link in links:
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        landing_url = raw.get("URL")

        return Paper(
            source=PaperSource.CROSSREF,
            external_ids=ExternalIDs(doi=doi or None),
            title=title,
            abstract=abstract or None,
            authors=authors,
            year=year,
            venue=venue,
            citation_count=cites,
            pdf_url=pdf_url,
            landing_page_url=landing_url,
            retrieved_from_queries=[query] if query else [],
        )
    except Exception as exc:
        logger.debug(f"CrossRef parse error: {exc}")
        return None


class CrossRefAdapter:
    """Metadata enrichment via CrossRef."""

    async def fetch_by_doi(self, doi: str) -> Optional[Paper]:
        """Fetch full metadata for a DOI."""
        encoded = urllib.parse.quote(doi, safe="")
        url = f"{BASE_URL}/{encoded}"
        try:
            async with get_http_client(
                timeout=settings.api.http_timeout,
                headers=POLITE_HEADERS,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            raw = data.get("message", {})
            return _parse_crossref_work(raw, f"doi:{doi}")
        except Exception as exc:
            logger.debug(f"CrossRef DOI fetch failed for {doi}: {exc}")
            return None

    async def search(self, query: str, rows: int = 10) -> list[Paper]:
        """Search CrossRef for metadata repair (rarely used directly)."""
        params = {
            "query": query,
            "rows": rows,
            "select": "DOI,title,author,published,container-title,abstract,is-referenced-by-count,URL,link",
            "mailto": MAILTO,
        }
        papers = []
        try:
            async with get_http_client(
                timeout=settings.api.http_timeout,
                headers=POLITE_HEADERS,
            ) as client:
                resp = await client.get(BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("message", {}).get("items", []):
                p = _parse_crossref_work(item, query)
                if p:
                    papers.append(p)
        except Exception as exc:
            logger.error(f"CrossRef search error: {exc}")
        return papers

    async def enrich_papers(self, papers: list[Paper]) -> list[Paper]:
        """
        For papers missing metadata (venue, year, abstract),
        attempt to enrich from CrossRef using their DOI.
        """
        tasks = []
        indices = []

        for i, paper in enumerate(papers):
            doi = paper.get_best_doi()
            if doi and (not paper.venue or not paper.year):
                tasks.append(self.fetch_by_doi(doi))
                indices.append(i)

        if not tasks:
            return papers

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in zip(indices, results):
            if isinstance(result, Paper):
                paper = papers[idx]
                # Only fill missing fields — don't overwrite better data
                if not paper.venue and result.venue:
                    paper.venue = result.venue
                if not paper.year and result.year:
                    paper.year = result.year
                if not paper.abstract and result.abstract:
                    paper.abstract = result.abstract
                if not paper.authors and result.authors:
                    paper.authors = result.authors

        return papers