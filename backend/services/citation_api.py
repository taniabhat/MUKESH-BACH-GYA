import asyncio
import random

import httpx

from config import get_settings


settings = get_settings()


CROSSREF_API = (
    "https://api.crossref.org"
)

SEMANTIC_SCHOLAR_API = (
    "https://api.semanticscholar.org/graph/v1"
)


CROSSREF_SEMAPHORE = (
    asyncio.Semaphore(5)
)

SEMANTIC_SCHOLAR_SEMAPHORE = (
    asyncio.Semaphore(5)
)


# -------------------------------------------------------------------
# Helpers
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


def normalize_crossref_result(
    work: dict
) -> dict:

    title = (
        work.get(
            "title",
            ["Untitled"]
        )[0]
    )

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

    venue = None

    container_titles = (
        work.get(
            "container-title",
            []
        )
    )

    if container_titles:
        venue = container_titles[0]

    return {
        "doi": work.get(
            "DOI"
        ),

        "title": title,

        "authors": authors,

        "year": year,

        "venue": venue,

        "publisher": work.get(
            "publisher"
        )
    }


def normalize_semantic_scholar_result(
    paper: dict
) -> dict:

    external_ids = (
        paper.get(
            "externalIds"
        )
        or {}
    )

    return {
        "doi": external_ids.get(
            "DOI"
        ),

        "title": paper.get(
            "title"
        ),

        "authors": [
            author.get("name")
            for author in paper.get(
                "authors",
                []
            )
        ],

        "year": paper.get(
            "year"
        ),

        "venue": paper.get(
            "venue"
        ),

        "publisher": None
    }


# -------------------------------------------------------------------
# DOI Lookup
# -------------------------------------------------------------------


async def lookup_doi(
    title: str,
    authors: list[str],
    year: int
) -> dict | None:

    async with CROSSREF_SEMAPHORE:

        url = (
            f"{CROSSREF_API}/works"
        )

        author_query = ", ".join(
            authors[:3]
        )

        filters = []

        if year:

            filters.append(
                f"from-pub-date:{year - 1}"
            )

        params = {
            "query.title": title,
            "query.author": author_query,
            "rows": 5
        }

        if filters:

            params["filter"] = ",".join(
                filters
            )

        headers = {
            "User-Agent": (
                "ResearchOS/1.0 "
                f"(mailto:{settings.CROSSREF_EMAIL})"
            )
        }

        async with httpx.AsyncClient(
            timeout=60
        ) as client:

            try:

                response = (
                    await request_with_backoff(
                        client,
                        "GET",
                        url,
                        params=params,
                        headers=headers
                    )
                )

            except Exception:
                return None

    data = response.json()

    items = (
        data.get(
            "message",
            {}
        )
        .get(
            "items",
            []
        )
    )

    if not items:
        return None

    return normalize_crossref_result(
        items[0]
    )


# -------------------------------------------------------------------
# Semantic Scholar Fallback
# -------------------------------------------------------------------


async def search_by_title(
    title: str
) -> dict | None:

    async with SEMANTIC_SCHOLAR_SEMAPHORE:

        url = (
            f"{SEMANTIC_SCHOLAR_API}"
            "/paper/search"
        )

        params = {
            "query": title,
            "limit": 1,
            "fields": ",".join([
                "title",
                "authors",
                "year",
                "venue",
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

            try:

                response = (
                    await request_with_backoff(
                        client,
                        "GET",
                        url,
                        params=params,
                        headers=headers
                    )
                )

            except Exception:
                return None

    data = response.json()

    papers = data.get(
        "data",
        []
    )

    if not papers:
        return None

    return normalize_semantic_scholar_result(
        papers[0]
    )


# -------------------------------------------------------------------
# Canonical DOI Metadata
# -------------------------------------------------------------------


async def get_canonical_metadata(
    doi: str
) -> dict | None:

    async with CROSSREF_SEMAPHORE:

        url = (
            f"{CROSSREF_API}"
            f"/works/{doi}"
        )

        headers = {
            "User-Agent": (
                "ResearchOS/1.0 "
                f"(mailto:{settings.CROSSREF_EMAIL})"
            )
        }

        async with httpx.AsyncClient(
            timeout=60
        ) as client:

            try:

                response = (
                    await request_with_backoff(
                        client,
                        "GET",
                        url,
                        headers=headers
                    )
                )

            except httpx.HTTPStatusError as error:

                if (
                    error.response.status_code
                    == 404
                ):
                    return None

                raise

            except Exception:
                return None

    data = response.json()

    work = data.get(
        "message",
        {}
    )

    if not work:
        return None

    return normalize_crossref_result(
        work
    )