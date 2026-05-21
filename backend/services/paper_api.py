import asyncio
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import feedparser
import httpx

from config import get_settings


settings = get_settings()


SEMANTIC_SCHOLAR_API = (
    "https://api.semanticscholar.org/graph/v1"
)

ARXIV_API = (
    "https://export.arxiv.org/api/query"
)

CROSSREF_API = (
    "https://api.crossref.org"
)

OPENALEX_API = (
    "https://api.openalex.org"
)


SEMANTIC_SCHOLAR_SEMAPHORE = (
    asyncio.Semaphore(5)
)

OPENALEX_SEMAPHORE = (
    asyncio.Semaphore(10)
)

CROSSREF_SEMAPHORE = (
    asyncio.Semaphore(5)
)

PDF_DOWNLOAD_SEMAPHORE = (
    asyncio.Semaphore(3)
)


# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------


def normalize_paper(
    *,
    external_id: str | None,
    title: str | None,
    authors: list[str] | None,
    year: int | None,
    doi: str | None,
    abstract: str | None,
    citation_count: int | None,
    pdf_url: str | None,
    source: str
) -> dict:

    return {
        "external_id": external_id,
        "title": title,
        "authors": authors or [],
        "year": year,
        "doi": doi,
        "abstract": abstract,
        "citation_count": citation_count or 0,
        "pdf_url": pdf_url,
        "source": source
    }


# -------------------------------------------------------------------
# Backoff
# -------------------------------------------------------------------




RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504
}


async def request_with_backoff(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 5,
    **kwargs
):

    last_error = None

    for attempt in range(max_retries):

        try:

            response = await client.request(
                method,
                url,
                **kwargs
            )

            if response.status_code in RETRYABLE_STATUS_CODES:

                wait_time = (
                    (2 ** attempt)
                    + random.uniform(0, 1)
                )

                await asyncio.sleep(wait_time)

                continue

            response.raise_for_status()

            return response

        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError
        ) as error:

            last_error = error

            wait_time = (
                (2 ** attempt)
                + random.uniform(0, 1)
            )

            await asyncio.sleep(wait_time)

    raise RuntimeError(
        f"Request failed after retries: {url}"
    ) from last_error


# -------------------------------------------------------------------
# Semantic Scholar
# -------------------------------------------------------------------


async def _fetch_semantic_scholar(
    query: str,
    limit: int
) -> dict:

    async with SEMANTIC_SCHOLAR_SEMAPHORE:

        url = (
            f"{SEMANTIC_SCHOLAR_API}"
            "/paper/search/bulk"
        )

        params = {
            "query": query,
            "fields": ",".join([
                "title",
                "authors",
                "year",
                "abstract",
                "citationCount",
                "openAccessPdf",
                "externalIds",
                "url"
            ]),
            "sort": "citationCount:desc",
            "minCitationCount": 5,
            "year": "2020-",
            "limit": limit
        }

        headers = {
            "User-Agent":
                "ResearchOS/1.0"
        }

        if settings.S2_API_KEY:

            headers["x-api-key"] = (
                settings.S2_API_KEY
            )

        async with httpx.AsyncClient(
            timeout=60
        ) as client:

            response = (
                await request_with_backoff(
                    client,
                    "GET",
                    url,
                    params=params,
                    headers=headers
                )
            )

            return response.json()


def _normalize_semantic_scholar(
    data: dict
) -> list[dict]:

    papers = []

    for paper in data.get("data", []):

        pdf = (
            paper.get(
                "openAccessPdf"
            )
            or {}
        )

        external_ids = (
            paper.get(
                "externalIds"
            )
            or {}
        )

        papers.append(
            normalize_paper(
                external_id=paper.get(
                    "paperId"
                ),

                title=paper.get(
                    "title"
                ),

                authors=[
                    author.get("name")
                    for author in paper.get(
                        "authors",
                        []
                    )
                ],

                year=paper.get(
                    "year"
                ),

                doi=external_ids.get(
                    "DOI"
                ),

                abstract=paper.get(
                    "abstract"
                ),

                citation_count=paper.get(
                    "citationCount"
                ),

                pdf_url=pdf.get(
                    "url"
                ),

                source="semantic_scholar"
            )
        )

    return papers


async def search_semantic_scholar(
    query: str,
    limit: int = 50
) -> list[dict]:

    data = await _fetch_semantic_scholar(
        query,
        limit
    )

    return _normalize_semantic_scholar(
        data
    )


async def get_paper_references(
    s2_paper_id: str
) -> list[dict]:

    async with SEMANTIC_SCHOLAR_SEMAPHORE:

        url = (
            f"{SEMANTIC_SCHOLAR_API}"
            f"/paper/{s2_paper_id}/references"
        )

        params = {
            "fields": ",".join([
                "title",
                "authors",
                "year",
                "abstract",
                "citationCount",
                "openAccessPdf",
                "externalIds"
            ])
        }

        headers = {
            "User-Agent":
                "ResearchOS/1.0"
        }

        if settings.S2_API_KEY:

            headers["x-api-key"] = (
                settings.S2_API_KEY
            )

        async with httpx.AsyncClient(
            timeout=60
        ) as client:

            response = (
                await request_with_backoff(
                    client,
                    "GET",
                    url,
                    params=params,
                    headers=headers
                )
            )

            data = response.json()

    references = []

    for item in data.get("data", []):

        cited = item.get(
            "citedPaper"
        )

        if not cited:
            continue

        pdf = (
            cited.get(
                "openAccessPdf"
            )
            or {}
        )

        external_ids = (
            cited.get(
                "externalIds"
            )
            or {}
        )

        references.append(
            normalize_paper(
                external_id=cited.get(
                    "paperId"
                ),

                title=cited.get(
                    "title"
                ),

                authors=[
                    author.get("name")
                    for author in cited.get(
                        "authors",
                        []
                    )
                ],

                year=cited.get(
                    "year"
                ),

                doi=external_ids.get(
                    "DOI"
                ),

                abstract=cited.get(
                    "abstract"
                ),

                citation_count=cited.get(
                    "citationCount"
                ),

                pdf_url=pdf.get(
                    "url"
                ),

                source="semantic_scholar"
            )
        )

    return references


# -------------------------------------------------------------------
# arXiv
# -------------------------------------------------------------------


async def search_arxiv(
    query: str,
    limit: int = 50
) -> list[dict]:

    params = {
        "search_query": (
            f'all:"{query}"'
        ),
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    headers = {
        "User-Agent": (
            f"ResearchPipeline/1.0 "
            f"({settings.CROSSREF_EMAIL})"
        )
    }

    max_retries = 5

    async with httpx.AsyncClient(
        timeout=60,
        follow_redirects=True
    ) as client:

        for attempt in range(max_retries):

            response = await client.get(
                ARXIV_API,
                params=params,
                headers=headers
            )

            if response.status_code == 429:

                wait_time = (
                    (2 ** attempt)
                    + random.uniform(1, 3)
                )

                print(
                    f"arXiv rate limited. "
                    f"Retrying in "
                    f"{wait_time:.2f}s"
                )

                await asyncio.sleep(
                    wait_time
                )

                continue

            response.raise_for_status()

            feed = feedparser.parse(
                response.text
            )

            papers = []

            for entry in feed.entries:

                pdf_url = None

                for link in entry.links:

                    if (
                        getattr(
                            link,
                            "type",
                            ""
                        )
                        == "application/pdf"
                    ):

                        pdf_url = link.href
                        break

                papers.append({
                    "external_id":
                        entry.get("id"),

                    "title":
                        entry.get(
                            "title"
                        ),

                    "authors": [
                        author.name
                        for author in entry.get(
                            "authors",
                            []
                        )
                    ],

                    "year": (
                        int(
                            entry.published[:4]
                        )
                        if entry.get(
                            "published"
                        )
                        else None
                    ),

                    "doi": None,

                    "abstract":
                        entry.get(
                            "summary",
                            ""
                        ),

                    "citation_count":
                        0,

                    "pdf_url":
                        pdf_url,

                    "source":
                        "arxiv"
                })

            return papers

    raise RuntimeError(
        "arXiv API failed after retries"
    )


# -------------------------------------------------------------------
# CrossRef
# -------------------------------------------------------------------


async def search_crossref(
    query: str,
    limit: int = 50
) -> list[dict]:

    async with CROSSREF_SEMAPHORE:

        url = (
            f"{CROSSREF_API}/works"
        )

        headers = {
            "User-Agent": (
                "ResearchOS/1.0 "
                f"(mailto:{settings.CROSSREF_EMAIL})"
            )
        }

        params = {
            "query": query,
            "rows": limit,
            "filter":
                "type:journal-article"
        }

        async with httpx.AsyncClient(
            timeout=60
        ) as client:

            response = (
                await request_with_backoff(
                    client,
                    "GET",
                    url,
                    params=params,
                    headers=headers
                )
            )

            data = response.json()

    papers = []

    items = (
        data.get("message", {})
        .get("items", [])
    )

    for work in items:

        authors = []

        for author in work.get(
            "author",
            []
        ):

            given = author.get(
                "given",
                ""
            )

            family = author.get(
                "family",
                ""
            )

            authors.append(
                f"{given} {family}".strip()
            )

        title = (
            work.get(
                "title",
                ["Untitled"]
            )[0]
        )

        published = (
            work.get(
                "created",
                {}
            )
            .get(
                "date-time",
                ""
            )
        )

        year = None

        if published:
            year = int(
                published[:4]
            )

        papers.append(
            normalize_paper(
                external_id=work.get(
                    "DOI"
                ),

                title=title,

                authors=authors,

                year=year,

                doi=work.get(
                    "DOI"
                ),

                abstract=work.get(
                    "abstract"
                ),

                citation_count=work.get(
                    "is-referenced-by-count",
                    0
                ),

                pdf_url=extract_crossref_pdf(
                    work
                ),

                source="crossref"
            )
        )

    return papers


def extract_crossref_pdf(
    work: dict
) -> str | None:

    links = work.get(
        "link",
        []
    )

    for link in links:

        content_type = (
            link.get(
                "content-type",
                ""
            )
        )

        if (
            "pdf"
            in content_type.lower()
        ):
            return link.get("URL")

    return None


# -------------------------------------------------------------------
# OpenAlex
# -------------------------------------------------------------------


async def search_openalex(
    query: str,
    limit: int = 50
) -> list[dict]:

    async with OPENALEX_SEMAPHORE:

        url = (
            f"{OPENALEX_API}/works"
        )

        params = {
            "search": query,

            "filter": (
                "is_oa:true,"
                "has_fulltext:true,"
                "publication_year:>2020,"
                "cited_by_count:>5"
            ),

            "sort":
                "cited_by_count:desc",

            "per_page":
                limit,

            "select":
                ",".join([
                    "id",
                    "doi",
                    "title",
                    "publication_year",
                    "cited_by_count",
                    "primary_location",
                    "authorships",
                    "open_access",
                    "abstract_inverted_index"
                ])
        }

        headers = {
            "User-Agent":
                "ResearchOS/1.0"
        }

        async with httpx.AsyncClient(
            timeout=60
        ) as client:

            response = (
                await request_with_backoff(
                    client,
                    "GET",
                    url,
                    params=params,
                    headers=headers
                )
            )

            data = response.json()

    papers = []

    for work in data.get(
        "results",
        []
    ):

        authors = []

        for authorship in work.get(
            "authorships",
            []
        ):

            author = (
                authorship.get(
                    "author",
                    {}
                )
            )

            name = author.get(
                "display_name"
            )

            if name:
                authors.append(name)

        papers.append(
            normalize_paper(
                external_id=work.get(
                    "id"
                ),

                title=work.get(
                    "title"
                ),

                authors=authors,

                year=work.get(
                    "publication_year"
                ),

                doi=work.get(
                    "doi"
                ),

                abstract=(
                    reconstruct_abstract(
                        work.get(
                            "abstract_inverted_index"
                        )
                    )
                ),

                citation_count=work.get(
                    "cited_by_count",
                    0
                ),

                pdf_url=extract_openalex_pdf(
                    work
                ),

                source="openalex"
            )
        )

    return papers


def reconstruct_abstract(
    inverted_index: dict | None
) -> str | None:

    if not inverted_index:
        return None

    position_map = {}

    for word, positions in (
        inverted_index.items()
    ):

        for pos in positions:
            position_map[pos] = word

    ordered = [
        position_map[i]
        for i in sorted(
            position_map.keys()
        )
    ]

    return " ".join(ordered)


def extract_openalex_pdf(
    work: dict
) -> str | None:

    open_access = (
        work.get(
            "open_access",
            {}
        )
    )

    oa_url = open_access.get(
        "oa_url"
    )

    if oa_url:
        return oa_url

    primary_location = (
        work.get(
            "primary_location",
            {}
        )
    )

    pdf_url = primary_location.get(
        "pdf_url"
    )

    return pdf_url


# -------------------------------------------------------------------
# PDF Download
# -------------------------------------------------------------------


async def download_pdf(
    url: str,
    save_path: str
) -> bool:

    async with PDF_DOWNLOAD_SEMAPHORE:

        headers = {
            "User-Agent":
                "ResearchOS/1.0"
        }

        for attempt in range(5):

            try:

                async with httpx.AsyncClient(
                    timeout=120,
                    follow_redirects=True
                ) as client:

                    response = await client.get(
                        url,
                        headers=headers
                    )

                    if (
                        response.status_code
                        != 200
                    ):
                        raise Exception(
                            "Bad response"
                        )

                    content_type = (
                        response.headers.get(
                            "content-type",
                            ""
                        )
                    )

                    if (
                        "application/pdf"
                        not in content_type
                        and "octet-stream"
                        not in content_type
                    ):
                        raise Exception(
                            "Not a PDF"
                        )

                    output_path = Path(
                        save_path
                    )

                    output_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    output_path.write_bytes(
                        response.content
                    )

                    return True

            except Exception:

                if attempt == 4:
                    return False

                delay = (
                    (2 ** attempt)
                    + random.uniform(0, 1)
                )

                await asyncio.sleep(delay)

    return False