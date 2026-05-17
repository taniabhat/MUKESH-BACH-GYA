"""
OpenAlex API Adapter — Primary retrieval backend.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.runtime import (
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

BASE_URL = settings.openalex.base_url
EMAIL = settings.openalex.email

MAX_REFERENCES = 50
MAX_FIELDS_OF_STUDY = 10
MAX_KEYWORDS = 20
CONCEPT_SCORE_THRESHOLD = 0.4


class OpenAlexAdapter:
    """Primary OpenAlex retrieval adapter."""

    def __init__(self):

        self.base_url = BASE_URL

        self.default_params = {
            "mailto": EMAIL,
        }

    async def search(
        self,
        query: str,
        per_page: int = 25,
        filter_str: Optional[str] = None,
    ) -> SearchResult:

        params = {
            **self.default_params,
            "search": query,
            "per-page": per_page,
            "select": (
                "id,title,abstract,authorships,"
                "publication_year,publication_date,"
                "doi,open_access,locations,"
                "primary_location,cited_by_count,"
                "referenced_works,concepts,"
                "keywords,landing_page_url"
            ),
        }

        if filter_str:
            params["filter"] = filter_str

        try:
            data = await self._fetch_json(
                f"{self.base_url}/works",
                params=params,
            )

            papers = self._parse_results(
                data,
                query=query,
            )

            total_found = (
                data.get("meta", {})
                .get("count", 0)
            )

            logger.info(
                "OpenAlex query='%s' fetched=%s total=%s",
                query,
                len(papers),
                total_found,
            )

            return SearchResult(
                source=PaperSource.OPENALEX,
                query=query,
                papers=papers,
                total_found=total_found,
            )

        except Exception:
            logger.exception(
                "OpenAlex search failed query='%s'",
                query,
            )

            return SearchResult(
                source=PaperSource.OPENALEX,
                query=query,
                papers=[],
                total_found=0,
            )

    async def fetch_by_doi(
        self,
        doi: str,
    ) -> Optional[Paper]:

        try:
            encoded_doi = urllib.parse.quote(
                doi,
                safe="",
            )

            url = (
                f"{self.base_url}/works/"
                f"https://doi.org/{encoded_doi}"
            )

            raw = await self._fetch_json(
                url,
                params=self.default_params,
            )

            return self._parse_paper(
                raw,
                query=f"doi:{doi}",
            )

        except Exception:
            logger.exception(
                "OpenAlex DOI fetch failed doi='%s'",
                doi,
            )

            return None

    async def fetch_citations(
        self,
        openalex_id: str,
    ) -> list[Paper]:

        return await self._fetch_related_papers(
            openalex_id=openalex_id,
            filter_key="cites",
            query_prefix="citation_expansion",
        )

    async def fetch_references(
        self,
        openalex_id: str,
    ) -> list[Paper]:

        return await self._fetch_related_papers(
            openalex_id=openalex_id,
            filter_key="cited_by",
            query_prefix="reference_expansion",
        )

    async def _fetch_related_papers(
        self,
        openalex_id: str,
        filter_key: str,
        query_prefix: str,
    ) -> list[Paper]:

        clean_id = self._clean_openalex_id(
            openalex_id,
        )

        params = {
            **self.default_params,
            "filter": (
                f"{filter_key}:{clean_id}"
            ),
            "per-page": 50,
            "select": (
                "id,title,abstract,"
                "authorships,"
                "publication_year,"
                "doi,cited_by_count"
            ),
        }

        try:
            data = await self._fetch_json(
                f"{self.base_url}/works",
                params=params,
            )

            papers = self._parse_results(
                data,
                query=(
                    f"{query_prefix}:{clean_id}"
                ),
            )

            for paper in papers:
                paper.source = (
                    PaperSource.CITATION_EXPANSION
                )

            return papers

        except Exception:
            logger.exception(
                "OpenAlex related paper fetch failed"
            )

            return []

    @api_retry(max_attempts=settings.openalex.max_retries)
    async def _fetch_json(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> dict:

        async with get_http_client(
            timeout=settings.http.timeout,
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

        for raw in data.get("results", []):

            try:
                paper = self._parse_paper(
                    raw,
                    query=query,
                )

                if paper:
                    papers.append(paper)

            except Exception:
                logger.exception(
                    "Failed to parse OpenAlex paper"
                )

        return papers

    def _parse_paper(
        self,
        raw: dict,
        query: str,
    ) -> Optional[Paper]:

        title = self._extract_title(raw)

        if not title:
            return None

        return Paper(
            source=PaperSource.OPENALEX,
            external_ids=self._extract_external_ids(raw),
            title=title,
            abstract=raw.get("abstract"),
            authors=self._extract_authors(raw),
            year=raw.get("publication_year"),
            venue=self._extract_venue(raw),
            publication_date=raw.get(
                "publication_date"
            ),
            citation_count=raw.get(
                "cited_by_count",
                0,
            ),
            reference_count=len(
                raw.get(
                    "referenced_works",
                    [],
                )
            ),
            fields_of_study=(
                self._extract_fields(raw)
            ),
            keywords=(
                raw.get(
                    "keywords",
                    [],
                )[:MAX_KEYWORDS]
            ),
            pdf_url=self._extract_pdf_url(raw),
            landing_page_url=raw.get(
                "landing_page_url"
            ),
            is_open_access=(
                self._extract_oa_status(raw)
            ),
            references=self._extract_references(raw),
            retrieved_from_queries=[query],
        )

    @staticmethod
    def _extract_title(
        raw: dict,
    ) -> str:

        return (
            raw.get("title")
            or raw.get("display_name")
            or ""
        ).strip()

    def _extract_external_ids(
        self,
        raw: dict,
    ) -> ExternalIDs:

        return ExternalIDs(
            doi=self._normalize_doi(
                raw.get("doi")
            ),
            arxiv=self._extract_arxiv_id(raw),
            openalex=raw.get("id"),
        )

    @staticmethod
    def _normalize_doi(
        doi: Optional[str],
    ) -> Optional[str]:

        if not doi:
            return None

        doi = doi.strip()

        doi = re.sub(
            r"^https?://doi\.org/",
            "",
            doi,
        )

        doi = re.sub(
            r"^https?://dx\.doi\.org/",
            "",
            doi,
        )

        return doi.lower() or None

    def _extract_arxiv_id(
        self,
        raw: dict,
    ) -> Optional[str]:

        for location in raw.get(
            "locations",
            [],
        ):

            source = (
                location.get("source")
                or {}
            )

            lineage = source.get(
                "host_organization_lineage_names",
                [],
            )

            if "arxiv" not in lineage:
                continue

            url = location.get(
                "landing_page_url",
                "",
            )

            match = re.search(
                r"arxiv\.org/abs/([0-9.]+)",
                url,
            )

            if match:
                return match.group(1)

        return None

    @staticmethod
    def _extract_authors(
        raw: dict,
    ) -> list[Author]:

        authors = []

        for authorship in raw.get(
            "authorships",
            [],
        ):

            author_data = (
                authorship.get("author")
                or {}
            )

            institutions = authorship.get(
                "institutions",
                [],
            )

            affiliation = (
                institutions[0].get(
                    "display_name"
                )
                if institutions
                else None
            )

            authors.append(
                Author(
                    name=author_data.get(
                        "display_name",
                        "Unknown",
                    ),
                    author_id=author_data.get(
                        "id"
                    ),
                    affiliation=affiliation,
                )
            )

        return authors

    @staticmethod
    def _extract_venue(
        raw: dict,
    ) -> Optional[str]:

        primary_location = (
            raw.get("primary_location")
            or {}
        )

        source = (
            primary_location.get("source")
            or {}
        )

        return source.get("display_name")

    def _extract_fields(
        self,
        raw: dict,
    ) -> list[str]:

        fields = []

        for concept in raw.get(
            "concepts",
            [],
        ):

            if (
                concept.get("score", 0)
                <= CONCEPT_SCORE_THRESHOLD
            ):
                continue

            display_name = concept.get(
                "display_name"
            )

            if display_name:
                fields.append(display_name)

        return fields[:MAX_FIELDS_OF_STUDY]

    @staticmethod
    def _extract_pdf_url(
        raw: dict,
    ) -> Optional[str]:

        for location in raw.get(
            "locations",
            [],
        ):

            if (
                location.get("is_oa")
                and location.get("pdf_url")
            ):
                return location["pdf_url"]

        return None

    @staticmethod
    def _extract_oa_status(
        raw: dict,
    ) -> bool:

        open_access = (
            raw.get("open_access")
            or {}
        )

        return open_access.get(
            "is_oa",
            False,
        )

    def _extract_references(
        self,
        raw: dict,
    ) -> list[PaperReference]:

        references = []

        for reference_id in raw.get(
            "referenced_works",
            [],
        )[:MAX_REFERENCES]:

            references.append(
                PaperReference(
                    openalex_id=reference_id
                )
            )

        return references

    @staticmethod
    def _clean_openalex_id(
        openalex_id: str,
    ) -> str:

        return openalex_id.replace(
            "https://openalex.org/",
            "",
        )