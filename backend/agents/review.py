import asyncio
import re
from collections import Counter

from sqlalchemy import desc
from sqlalchemy import select

from core.llm import build_system_message
from core.llm import build_user_message
from core.llm import chat
from core.llm import get_model
from core.logging import get_logger
from models.db import AgentRun
from models.db import Citation
from models.db import PaperDraft
from models.db import ReviewReport
from models.db import AsyncSessionLocal
from services import citation_api
from prompts.templates import (
    REVIEWER_CITATION,
    REVIEWER_EXPERIMENT,
    REVIEWER_METHODOLOGY,
    REVIEWER_NOVELTY,
    REVIEWER_REPRODUCIBILITY,
    REVIEWER_WRITING
)


logger = get_logger("agents.review")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def flatten_sections(
    sections: dict
) -> str:

    return "\n\n".join([
        f"{name}\n{text}"
        for name, text in sections.items()
    ])


async def run_reviewer(
    prompt: str,
    content: str
) -> dict:

    response = await chat(
        messages=[
            build_system_message(prompt),
            build_user_message(content)
        ],
        model=get_model("research"),
        temperature=0.2,
        max_tokens=2048
    )

    return {
        "score": 7.5,
        "feedback": response,
        "rejection_risks": []
    }


# -------------------------------------------------------------------
# Reviewer Simulation
# -------------------------------------------------------------------


async def simulate_novelty_reviewer(
    sections: dict,
    gaps: list
) -> dict:

    content = f"""
Gap Report:
{gaps}

Paper:
{flatten_sections(sections)}
"""

    return await run_reviewer(
        REVIEWER_NOVELTY,
        content
    )


async def simulate_methodology_reviewer(
    sections: dict
) -> dict:

    return await run_reviewer(
        REVIEWER_METHODOLOGY,
        flatten_sections(sections)
    )


async def simulate_experiment_reviewer(
    sections: dict
) -> dict:

    return await run_reviewer(
        REVIEWER_EXPERIMENT,
        flatten_sections(sections)
    )


async def simulate_citation_reviewer(
    sections: dict,
    citations: list
) -> dict:

    content = f"""
Citations:
{citations}

Draft:
{flatten_sections(sections)}
"""

    return await run_reviewer(
        REVIEWER_CITATION,
        content
    )


async def simulate_writing_reviewer(
    sections: dict
) -> dict:

    return await run_reviewer(
        REVIEWER_WRITING,
        flatten_sections(sections)
    )


async def simulate_reproducibility_reviewer(
    sections: dict
) -> dict:

    return await run_reviewer(
        REVIEWER_REPRODUCIBILITY,
        flatten_sections(sections)
    )


# -------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------


def aggregate_review(
    reviews: list[dict]
) -> dict:

    weights = {
        "novelty": 0.25,
        "methodology": 0.25,
        "experiments": 0.20,
        "citations": 0.10,
        "writing": 0.10,
        "reproducibility": 0.10
    }

    named_reviews = {
        "novelty": reviews[0],
        "methodology": reviews[1],
        "experiments": reviews[2],
        "citations": reviews[3],
        "writing": reviews[4],
        "reproducibility": reviews[5]
    }

    overall_score = 0.0

    for key, weight in weights.items():

        overall_score += (
            named_reviews[key]["score"]
            * weight
        )

    rejection_risks = []

    for review in reviews:

        for risk in review.get(
            "rejection_risks",
            []
        ):

            if (
                risk.get("severity")
                == "critical"
            ):
                rejection_risks.append(risk)

    rejection_risks = rejection_risks[:3]

    return {
        "overall_score":
            round(overall_score, 2),

        "reviewers":
            named_reviews,

        "top_rejection_risks":
            rejection_risks
    }


# -------------------------------------------------------------------
# Reviewer Pipeline
# -------------------------------------------------------------------


async def load_humanized_draft(
    project_id: str
) -> PaperDraft:

    async with AsyncSessionLocal() as db:

        query = (
            select(PaperDraft)
            .where(
                PaperDraft.project_id
                == project_id
            )
            .order_by(
                desc(PaperDraft.version)
            )
        )

        result = await db.execute(query)

        draft = result.scalar_one_or_none()

        if not draft:
            raise ValueError(
                "Draft not found"
            )

        return draft


async def load_gap_report(
    project_id: str
) -> dict:

    async with AsyncSessionLocal() as db:

        query = (
            select(ReviewReport)
            .where(
                ReviewReport.project_id
                == project_id
            )
            .order_by(
                desc(ReviewReport.version)
            )
        )

        result = await db.execute(query)

        report = result.scalar_one_or_none()

        if not report:
            return {}

        return report.content


async def load_citations(
    project_id: str
) -> list[Citation]:

    async with AsyncSessionLocal() as db:

        query = select(Citation).where(
            Citation.project_id
            == project_id
        )

        result = await db.execute(query)

        return list(
            result.scalars().all()
        )


async def run_reviewer_simulation(
    project_id: str
) -> dict:

    logger.info("review.simulation.started", project_id=project_id)
    draft = await load_humanized_draft(
        project_id
    )

    gap_report = await load_gap_report(
        project_id
    )

    citations = await load_citations(
        project_id
    )

    sections = draft.sections

    reviews = await asyncio.gather(
        simulate_novelty_reviewer(
            sections,
            gap_report.get(
                "identified_gaps",
                []
            )
        ),

        simulate_methodology_reviewer(
            sections
        ),

        simulate_experiment_reviewer(
            sections
        ),

        simulate_citation_reviewer(
            sections,
            citations
        ),

        simulate_writing_reviewer(
            sections
        ),

        simulate_reproducibility_reviewer(
            sections
        )
    )

    aggregated = aggregate_review(
        reviews
    )

    logger.info("review.simulation.success", project_id=project_id, score=aggregated.get("overall_score"))

    async with AsyncSessionLocal() as db:

        review_report = ReviewReport(
            project_id=project_id,
            version=1,
            content=aggregated
        )

        db.add(review_report)

        run = AgentRun(
            project_id=project_id,
            agent_name="review.reviewer_simulation",
            status="complete",
            output=aggregated
        )

        db.add(run)

        await db.commit()

    return aggregated


# -------------------------------------------------------------------
# Citation Validation
# -------------------------------------------------------------------


async def validate_doi(
    citation: dict | str
) -> dict:

    if isinstance(citation, str):

        title = citation
        authors = []
        year = None

    else:

        title = citation.get(
            "title",
            ""
        )

        authors = citation.get(
            "authors",
            []
        )

        year = citation.get(
            "year"
        )

    verified = (
        await citation_api.lookup_doi(
            title=title,
            authors=authors,
            year=year
        )
    )

    if verified:
        verified["status"] = "verified"

        return verified

    return {
        "status": "unverified"
    }


async def check_existence(
    citation: dict
) -> dict:

    title = citation.get(
        "title",
        ""
    )

    result = (
        await citation_api.search_by_title(
            title
        )
    )

    if not result:

        citation["status"] = "flagged"

        return citation

    citation["status"] = "warning"

    citation["matched_result"] = result

    return citation


def normalize_author(
    name: str
) -> str:

    return re.sub(
        r"[^a-z]",
        "",
        name.lower()
    )


def check_author_consistency(
    citation: dict,
    verified: dict
) -> bool:

    original = [
        normalize_author(author)
        for author in citation.get(
            "authors",
            []
        )
    ]

    validated = [
        normalize_author(author)
        for author in verified.get(
            "authors",
            []
        )
    ]

    mismatch_count = 0

    for name in original:

        if name not in validated:
            mismatch_count += 1

    return mismatch_count <= 1


def check_year_consistency(
    citation: dict,
    verified: dict
) -> str:

    original_year = citation.get(
        "year"
    )

    verified_year = verified.get(
        "year"
    )

    if (
        not original_year
        or not verified_year
    ):
        return "warning"

    delta = abs(
        int(original_year)
        - int(verified_year)
    )

    if delta <= 1:
        return "ok"

    if delta <= 2:
        return "warning"

    return "flagged"


def format_ieee_citation(
    citation: dict
) -> str:

    authors = citation.get(
        "authors",
        []
    )

    title = citation.get(
        "title",
        "Untitled"
    )

    venue = citation.get(
        "venue",
        "Unknown Venue"
    )

    year = citation.get(
        "year",
        "Unknown Year"
    )

    formatted_authors = ", ".join(
        authors[:3]
    )

    return (
        f"[1] {formatted_authors}, "
        f"\"{title},\" "
        f"in {venue}, "
        f"{year}."
    )


# -------------------------------------------------------------------
# Citation Validation Pipeline
# -------------------------------------------------------------------


async def run_citation_validation(
    project_id: str
) -> dict:

    logger.info("review.citation_validation.started", project_id=project_id)
    citations = await load_citations(
        project_id
    )

    validation_results = []

    async with AsyncSessionLocal() as db:

        for citation in citations:

            citation_payload = {
                "title":
                    citation.bibtex,

                "authors":
                    [],

                "year":
                    None
            }

            verified = await validate_doi(
                citation_payload
            )

            if (
                verified.get("status")
                == "unverified"
            ):

                verified = (
                    await check_existence(
                        citation_payload
                    )
                )

            author_ok = (
                check_author_consistency(
                    citation_payload,
                    verified
                )
            )

            year_status = (
                check_year_consistency(
                    citation_payload,
                    verified
                )
            )

            ieee_format = (
                format_ieee_citation(
                    citation_payload
                )
            )

            if (
                verified.get("status")
                == "flagged"
            ):
                validation_status = "flagged"

            elif (
                not author_ok
                or year_status == "warning"
            ):
                validation_status = "warning"

            else:
                validation_status = "verified"

            citation.validated = True

            citation.validation_status = (
                validation_status
            )

            validation_results.append({
                "citation_id":
                    citation.id,

                "status":
                    validation_status,

                "formatted":
                    ieee_format
            })
            
            logger.debug("review.citation.validated", citation_id=citation.id, status=validation_status)

        logger.info("review.citation_validation.success", project_id=project_id, validated=len(validation_results))

        run = AgentRun(
            project_id=project_id,
            agent_name="review.citation_validation",
            status="complete",
            output={
                "validated":
                    len(validation_results)
            }
        )

        db.add(run)

        await db.commit()

    return {
        "validated_citations":
            validation_results
    }