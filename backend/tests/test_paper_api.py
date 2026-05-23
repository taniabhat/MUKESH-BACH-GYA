import asyncio

from services.paper_api import (
    search_semantic_scholar,
    search_arxiv,
    search_crossref,
    search_openalex,
    download_pdf
)


QUERY = (
    "multimodal retrieval augmented generation"
)


async def test_semantic_scholar():

    print("\n=== Semantic Scholar ===\n")

    papers = await search_semantic_scholar(
        QUERY,
        limit=3
    )

    for index, paper in enumerate(
        papers,
        start=1
    ):

        print(f"[{index}] {paper['title']}")
        print(
            f"Year: {paper['year']}"
        )
        print(
            f"Citations: "
            f"{paper['citation_count']}"
        )
        print(
            f"PDF: {paper['pdf_url']}"
        )
        print()


async def test_arxiv():

    print("\n=== arXiv ===\n")

    papers = await search_arxiv(
        QUERY,
        limit=3
    )

    for index, paper in enumerate(
        papers,
        start=1
    ):

        print(f"[{index}] {paper['title']}")
        print(
            f"Year: {paper['year']}"
        )
        print(
            f"PDF: {paper['pdf_url']}"
        )
        print()


async def test_crossref():

    print("\n=== CrossRef ===\n")

    papers = await search_crossref(
        QUERY,
        limit=3
    )

    for index, paper in enumerate(
        papers,
        start=1
    ):

        print(f"[{index}] {paper['title']}")
        print(
            f"DOI: {paper['doi']}"
        )
        print(
            f"Citations: "
            f"{paper['citation_count']}"
        )
        print()


async def test_openalex():

    print("\n=== OpenAlex ===\n")

    papers = await search_openalex(
        QUERY,
        limit=3
    )

    for index, paper in enumerate(
        papers,
        start=1
    ):

        print(f"[{index}] {paper['title']}")
        print(
            f"Year: {paper['year']}"
        )
        print(
            f"Citations: "
            f"{paper['citation_count']}"
        )
        print(
            f"PDF: {paper['pdf_url']}"
        )
        print()

    return papers


async def test_download():

    papers = await search_openalex(
        QUERY,
        limit=1
    )

    if not papers:

        print("No papers found")
        return

    paper = papers[0]

    pdf_url = paper.get(
        "pdf_url"
    )

    if not pdf_url:

        print("No PDF URL found")
        return

    success = await download_pdf(
        pdf_url,
        "./downloads/test_paper.pdf"
    )

    if success:

        print(
            "\nPDF downloaded successfully:"
        )

        print(
            "./downloads/test_paper.pdf"
        )

    else:

        print(
            "\nPDF download failed"
        )


async def main():

    await test_semantic_scholar()

    # await test_arxiv()

    await test_crossref()

    await test_openalex()

    await test_download()


if __name__ == "__main__":

    asyncio.run(main())