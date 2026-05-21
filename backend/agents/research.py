import asyncio
import hashlib
import re
from pathlib import Path

from sqlalchemy import select

from core import document
from core import graph
from core import rag
from core.llm import build_system_message
from core.llm import build_user_message
from core.llm import chat
from core.llm import get_model
from core.logging import get_logger
from models.db import AgentRun
from models.db import AsyncSessionLocal
from models.db import Paper
from models.db import Project
from services import paper_api
from config import settings


logger = get_logger("agents.research")


# -------------------------------------------------------------------
# Query Expansion
# -------------------------------------------------------------------


async def expand_queries(
    idea: str
) -> list[str]:

    logger.debug("research.expand_queries.started")

    prompt = (
        f"Idea: {idea}\n\n"
        f"Generate 5 academic search queries."
    )

    response = await chat(
        messages=[
            build_system_message("You are an expert researcher. Generate 5 highly specific academic search queries. YOU MUST ONLY RETURN THE QUERIES, ONE PER LINE. DO NOT OUTPUT ANY OTHER TEXT OR REASONING."),
            build_user_message(prompt)
        ],
        model=get_model("research"),
        temperature=0.7,
        max_tokens=256
    )

    queries = [
        line.strip("- *").strip()
        for line in response.split("\n")
        if line.strip()
    ]

    logger.info("research.expand_queries.success", count=len(queries), queries=queries)
    return queries


# -------------------------------------------------------------------
# Multi-Source Retrieval
# -------------------------------------------------------------------


async def fetch_all_sources(
    query: str
) -> list[dict]:

    logger.debug("research.fetch_all_sources.started", query=query)
    tasks = [
        paper_api.search_semantic_scholar(
            query,
            limit=20
        ),

        paper_api.search_arxiv(
            query,
            limit=10
        ),

        paper_api.search_crossref(
            query,
            limit=10
        ),

        paper_api.search_openalex(
            query,
            limit=10
        )
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )

    all_papers = []

    for idx, source_result in enumerate(results):

        if isinstance(
            source_result,
            Exception
        ):
            logger.warning("research.fetch_source.failed", query=query, source_idx=idx, error=str(source_result))
            continue

        all_papers.extend(
            source_result
        )

    logger.info("research.fetch_all_sources.success", query=query, total_papers=len(all_papers))
    return all_papers


# -------------------------------------------------------------------
# Deduplication
# -------------------------------------------------------------------


def normalize_title(
    title: str
) -> str:

    title = title.lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        "",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip()


def build_title_hash(
    title: str
) -> str:

    normalized = normalize_title(title)

    return hashlib.md5(
        normalized.encode()
    ).hexdigest()


def paper_metadata_score(
    paper: dict
) -> int:

    score = 0

    important_fields = [
        "abstract",
        "authors",
        "doi",
        "year",
        "pdf_url"
    ]

    for field in important_fields:

        if paper.get(field):
            score += 1

    return score


def deduplicate(
    papers: list[dict]
) -> list[dict]:

    logger.debug("research.deduplicate.started", input_count=len(papers))
    grouped = {}

    for paper in papers:

        doi = paper.get("doi")

        if doi:
            key = f"doi:{doi.lower()}"

        else:
            key = (
                "title:"
                + build_title_hash(
                    paper.get("title", "")
                )
            )

        existing = grouped.get(key)

        if not existing:

            grouped[key] = paper
            continue

        current_score = paper_metadata_score(
            paper
        )

        existing_score = paper_metadata_score(
            existing
        )

        if current_score > existing_score:
            grouped[key] = paper

    result = list(grouped.values())
    logger.info("research.deduplicate.success", output_count=len(result))
    return result


# -------------------------------------------------------------------
# Relevance Ranking
# -------------------------------------------------------------------


def cosine_similarity(
    a: list[float],
    b: list[float]
) -> float:

    numerator = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = sum(x * x for x in a) ** 0.5

    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


async def rank_by_relevance(
    papers: list[dict],
    idea: str
) -> list[dict]:

    if not papers:
        return []

    idea_embedding = (await rag.embed_text([
        idea
    ]))[0]

    abstracts = [
        paper.get("abstract", "")
        for paper in papers
    ]

    abstract_embeddings = await rag.embed_text(
        abstracts
    )

    ranked = []

    for paper, embedding in zip(
        papers,
        abstract_embeddings
    ):

        similarity = cosine_similarity(
            idea_embedding,
            embedding
        )

        paper["relevance_score"] = similarity

        ranked.append(paper)

    ranked.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    return ranked


# -------------------------------------------------------------------
# Citation Expansion
# -------------------------------------------------------------------


async def expand_citations(
    top_papers: list[dict],
    limit: int = 20
) -> list[dict]:

    expanded = list(top_papers)

    tasks = []

    for paper in top_papers[:limit]:

        paper_id = (
            paper.get("external_id")
            or paper.get("doi")
        )

        if not paper_id:
            continue

        tasks.append(
            paper_api.get_paper_references(
                paper_id
            )
        )

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )

    for result in results:

        if isinstance(result, Exception):
            continue

        expanded.extend(result)

    return deduplicate(expanded)


# -------------------------------------------------------------------
# PDF Download
# -------------------------------------------------------------------


async def log_agent_event(
    project_id: str,
    agent_name: str,
    status: str,
    output: dict | None = None,
    error: str | None = None
) -> None:

    async with AsyncSessionLocal() as db:

        run = AgentRun(
            project_id=project_id,
            agent_name=agent_name,
            status=status,
            output=output,
            error=error
        )

        db.add(run)

        await db.commit()


async def download_pdf(
    paper: dict
) -> str | None:

    pdf_url = paper.get("pdf_url")

    if not pdf_url:
        return None

    paper_id = (
        paper.get("id")
        or paper.get("external_id")
    )

    save_path = (
        settings.papers_dir
        / f"{paper_id}.pdf"
    )

    try:

        downloaded = (
            await paper_api.download_pdf(
                pdf_url,
                str(save_path)
            )
        )

        if not downloaded:
            return None

        return str(save_path)

    except Exception:
        return None


# -------------------------------------------------------------------
# Database Persistence
# -------------------------------------------------------------------


async def save_papers_to_db(
    papers: list[dict],
    project_id: str
) -> list[str]:

    inserted_ids = []

    async with AsyncSessionLocal() as db:

        for paper in papers:

            db_paper = Paper(
                project_id=project_id,
                external_id=paper.get(
                    "external_id"
                ),
                title=paper.get("title"),
                authors=paper.get(
                    "authors",
                    []
                ),
                year=paper.get("year"),
                doi=paper.get("doi"),
                abstract=paper.get("abstract"),
                pdf_path=paper.get("pdf_path"),
                relevance_score=paper.get(
                    "relevance_score",
                    0.0
                )
            )

            db.add(db_paper)

            await db.flush()

            inserted_ids.append(db_paper.id)

        await db.commit()

    return inserted_ids


# -------------------------------------------------------------------
# Discovery Pipeline
# -------------------------------------------------------------------


async def load_project_idea(
    project_id: str
) -> str:

    async with AsyncSessionLocal() as db:

        query = select(Project).where(
            Project.id == project_id
        )

        result = await db.execute(query)

        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(
                "Project not found"
            )

        return project.research_idea


async def run_discovery(
    project_id: str,
    idea: str | None = None
) -> dict:

    if not idea:
        idea = await load_project_idea(
            project_id
        )

    try:

        queries = await expand_queries(
            idea
        )

        all_papers = []

        for query in queries:

            results = await fetch_all_sources(
                query
            )

            all_papers.extend(results)

        deduped = deduplicate(
            all_papers
        )

        ranked = await rank_by_relevance(
            deduped,
            idea
        )

        expanded = await expand_citations(
            ranked[:20]
        )

        reranked = await rank_by_relevance(
            expanded,
            idea
        )

        top_papers = reranked[:50]

        for paper in top_papers:

            pdf_path = await download_pdf(
                paper
            )

            paper["pdf_path"] = pdf_path

        paper_ids = await save_papers_to_db(
            top_papers,
            project_id
        )

        logger.info("research.discovery.success", project_id=project_id, paper_count=len(paper_ids))
        await log_agent_event(
            project_id=project_id,
            agent_name="research.discovery",
            status="complete",
            output={
                "paper_count":
                    len(paper_ids)
            }
        )

        return {
            "paper_ids": paper_ids
        }

    except Exception as error:

        logger.error("research.discovery.failed", project_id=project_id, error=str(error), exc_info=True)
        await log_agent_event(
            project_id=project_id,
            agent_name="research.discovery",
            status="failed",
            error=str(error)
        )

        raise


# -------------------------------------------------------------------
# Paper Analysis
# -------------------------------------------------------------------


async def analyze_paper(
    paper_id: str,
    pdf_path: str
) -> None:

    doc_result = (
        await document.process_paper(
            pdf_path,
            paper_id
        )
    )

    await rag.index_paper(
        doc_result,
        paper_id
    )

    async with AsyncSessionLocal() as db:

        query = select(Paper).where(
            Paper.id == paper_id
        )

        result = await db.execute(query)

        paper = result.scalar_one_or_none()

        if not paper:
            return

        paper.parsed_json = {
            "title":
                doc_result.title,

            "abstract":
                doc_result.abstract,

            "authors":
                doc_result.authors,

            "sections":
                doc_result.sections,

            "references":
                doc_result.references,

            "figures":
                doc_result.figures,

            "tables":
                doc_result.tables,

            "equations":
                doc_result.equations
        }

        await db.commit()

    await graph.build_paper_node({
        "id": paper_id,
        "title": doc_result.title,
        "year": paper.year if paper else None,
        "doi": paper.doi if paper else None,
        "venue": None
    })

    await graph.build_relationships(
        {
            "id": paper_id
        },
        doc_result
    )


# -------------------------------------------------------------------
# Document Analysis Pipeline
# -------------------------------------------------------------------


async def run_document_analysis(
    project_id: str
) -> dict:

    async with AsyncSessionLocal() as db:

        query = select(Paper).where(
            Paper.project_id == project_id,
            Paper.pdf_path.is_not(None)
        )

        result = await db.execute(query)

        papers = result.scalars().all()

    tasks = []

    for paper in papers:

        logger.debug("research.analysis.processing", paper_id=paper.id)
        tasks.append(
            analyze_paper(
                paper.id,
                paper.pdf_path
            )
        )

    if tasks:
        logger.info("research.analysis.started", task_count=len(tasks))
        await asyncio.gather(*tasks)

    logger.info("research.analysis.success", project_id=project_id, processed_papers=len(tasks))
    await log_agent_event(
        project_id=project_id,
        agent_name="research.analysis",
        status="complete",
        output={
            "processed_papers":
                len(tasks)
        }
    )

    return {
        "documents": [
            {
                "paper_id": paper.id,
                "title": paper.title
            }
            for paper in papers
        ]
    }