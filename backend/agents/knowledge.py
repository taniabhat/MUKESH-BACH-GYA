import statistics
from collections import Counter
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import select

from core import graph
from core import rag
from core.llm import build_system_message
from core.llm import build_user_message
from core.llm import chat
from core.llm import get_model
from core.llm import structured_chat
from core.logging import get_logger
from models.db import AgentRun
from models.db import AsyncSessionLocal
from models.db import Paper
from models.db import ReviewReport
from pydantic import BaseModel
from prompts.templates import LIMITATION_EXTRACTION, GAP_SYNTHESIS, NOVELTY_CHECK


logger = get_logger("agents.knowledge")


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------


class LimitationItem(BaseModel):
    category: str
    description: str
    severity: str
    evidence: str


class LimitationExtraction(BaseModel):
    limitations: list[LimitationItem]


class GapSynthesis(BaseModel):
    gaps: list[dict]


class NoveltyAssessment(BaseModel):
    novelty_score: float
    assessment: str
    closest_prior_art: list[str]
    differentiators: list[str]


# -------------------------------------------------------------------
# Limitation Extraction
# -------------------------------------------------------------------


async def extract_limitations(
    paper: dict,
    chunks: list[dict]
) -> list[dict]:

    limitation_chunks = []

    for chunk in chunks:

        section = (
            chunk.get(
                "section",
                ""
            ).lower()
        )

        if (
            "limitation" in section
            or "future work" in section
        ):
            limitation_chunks.append(
                chunk["content"]
            )

    if not limitation_chunks:
        return []

    content = "\n\n".join(
        limitation_chunks
    )

    extraction = await structured_chat(
        messages=[
            build_system_message(LIMITATION_EXTRACTION),
            build_user_message(content)
        ],
        model=get_model("research"),
        output_schema=LimitationExtraction
    )

    tagged_limitations = []

    for lim_item in extraction.limitations:
        limitation_text = f"[{lim_item.category}] {lim_item.description} (Evidence: {lim_item.evidence})"

        category_response = await chat(
            messages=[
                build_system_message(
                    """
Classify this limitation into ONE category:
- dataset
- method
- evaluation
- scope

Return category only.
"""
                ),
                build_user_message(limitation_text)
            ],
            model=get_model("fast"),
            temperature=0.1,
            max_tokens=16
        )

        tagged_limitations.append({
            "paper_id": paper["id"],
            "text": limitation_text,
            "category": (
                category_response
                .strip()
                .lower()
            )
        })

    return tagged_limitations


# -------------------------------------------------------------------
# Limitation Clustering
# -------------------------------------------------------------------


def determine_cluster_count(
    n_samples: int
) -> int:

    if n_samples <= 5:
        return 2

    if n_samples <= 20:
        return 5

    if n_samples <= 50:
        return 8

    return 12


def cluster_limitations(
    limitations: list[dict]
) -> list[dict]:

    if not limitations:
        return []

    # embeddings computed by caller via embed_text before clustering
    # use text directly for KMeans via TF-IDF-style hashing
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [item["text"] for item in limitations]

    vectorizer = TfidfVectorizer(max_features=128)
    embeddings = vectorizer.fit_transform(texts).toarray()

    cluster_count = determine_cluster_count(
        len(limitations)
    )

    cluster_count = min(
        cluster_count,
        len(limitations)
    )

    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init="auto"
    )

    labels = kmeans.fit_predict(
        embeddings
    )

    grouped = defaultdict(list)

    for limitation, label in zip(
        limitations,
        labels
    ):
        grouped[int(label)].append(
            limitation
        )

    clusters = []

    for cluster_id, members in grouped.items():

        representative = members[0]["text"]

        member_papers = list({
            item["paper_id"]
            for item in members
        })

        clusters.append({
            "cluster_id": cluster_id,
            "representative_text":
                representative,

            "member_papers":
                member_papers,

            "size":
                len(members),

            "limitations":
                members
        })

    return clusters


# -------------------------------------------------------------------
# Benchmark Gap Detection
# -------------------------------------------------------------------


def detect_benchmark_gaps(
    papers: list[dict]
) -> list[dict]:

    metric_counter = Counter()

    dataset_counter = Counter()

    recent_years = []

    old_years = []

    for paper in papers:

        parsed = (
            paper.get("parsed_json")
            or {}
        )

        sections = parsed.get(
            "sections",
            []
        )

        year = paper.get("year")

        for section in sections:

            text = section.get(
                "body",
                ""
            ).lower()

            if "accuracy" in text:
                metric_counter["accuracy"] += 1

            if "f1" in text:
                metric_counter["f1"] += 1

            if "bleu" in text:
                metric_counter["bleu"] += 1

            if "imagenet" in text:
                dataset_counter["imagenet"] += 1

            if "cifar" in text:
                dataset_counter["cifar"] += 1

        if year:

            if year >= 2022:
                recent_years.append(paper)

            else:
                old_years.append(paper)

    total_papers = max(len(papers), 1)

    gaps = []

    for metric, count in metric_counter.items():

        ratio = count / total_papers

        if ratio < 0.2:

            gaps.append({
                "gap_type":
                    "under_evaluated_metric",

                "description":
                    f"Metric '{metric}' appears in only {count} papers",

                "evidence_papers":
                    count
            })

    for dataset, count in dataset_counter.items():

        recent_mentions = 0

        for paper in recent_years:

            parsed = (
                paper.get("parsed_json")
                or {}
            )

            text = str(parsed).lower()

            if dataset in text:
                recent_mentions += 1

        if count > 3 and recent_mentions == 0:

            gaps.append({
                "gap_type":
                    "superseded_dataset",

                "description":
                    f"Dataset '{dataset}' common in older work but absent recently",

                "evidence_papers":
                    count
            })

    return gaps


# -------------------------------------------------------------------
# Temporal Trends
# -------------------------------------------------------------------


async def analyze_temporal_trends(
    papers: list[dict]
) -> dict:

    methods = [
        "transformer",
        "retrieval",
        "diffusion",
        "contrastive learning",
        "graph neural network"
    ]

    growing = []
    declining = []

    for method in methods:

        trend = await graph.analyze_trends(
            method
        )

        if trend["trend"] == "growing":
            growing.append(trend)

        elif trend["trend"] == "declining":
            declining.append(trend)

    return {
        "growing_methods":
            growing,

        "declining_methods":
            declining,

        "solved_problems":
            [],

        "emerging_problems":
            []
    }


# -------------------------------------------------------------------
# Contradiction Analysis
# -------------------------------------------------------------------


async def find_contradictions(
) -> list[dict]:

    contradictions = (
        await graph.find_contradictions()
    )

    annotated = []

    for contradiction in contradictions:

        prompt = f"""
Analyze whether this contradiction is:
1. methodological difference
2. genuine conflicting finding

Contradiction:
{contradiction}
"""

        response = await chat(
            messages=[
                build_user_message(prompt)
            ],
            model=get_model("research"),
            temperature=0.2,
            max_tokens=128
        )

        contradiction["assessment"] = (
            response.strip()
        )

        annotated.append(
            contradiction
        )

    return annotated


# -------------------------------------------------------------------
# Gap Synthesis
# -------------------------------------------------------------------


async def synthesize_gaps(
    clusters: list[dict],
    benchmark_gaps: list[dict],
    contradictions: list[dict],
    graph_gaps: list[dict]
) -> list[dict]:

    synthesis_prompt = f"""
Given these signals:

Limitation Clusters:
{clusters}

Benchmark Gaps:
{benchmark_gaps}

Contradictions:
{contradictions}

Graph Gaps:
{graph_gaps}

Identify the top 10 research opportunities.

For each gap include:
- title
- problem_statement
- severity
- novelty_opportunity
- suggested_contributions
"""

    result = await structured_chat(
        messages=[
            build_system_message(GAP_SYNTHESIS),
            build_user_message(synthesis_prompt)
        ],
        model=get_model("research"),
        output_schema=GapSynthesis
    )

    return result.gaps


# -------------------------------------------------------------------
# Novelty Scoring
# -------------------------------------------------------------------


async def score_novelty(
    proposed_idea: str,
    project_id: str
) -> dict:

    retrieved = await rag.retrieve(
        query=proposed_idea,
        project_id=project_id,
        top_k=20
    )

    graph_novelty = (
        await graph.check_novelty(
            proposed_idea
        )
    )

    prior_art = [
        chunk["content"][:300]
        for chunk in retrieved[:5]
    ]

    prompt = f"""
Research Idea:
{proposed_idea}

Closest Prior Art:
{prior_art}

Graph Similarity:
{graph_novelty}

Assess whether this idea is genuinely novel.
"""

    result = await structured_chat(
        messages=[
            build_system_message(NOVELTY_CHECK),
            build_user_message(prompt)
        ],
        model=get_model("research"),
        output_schema=NoveltyAssessment
    )

    return result.model_dump()


# -------------------------------------------------------------------
# Gap Report Builder
# -------------------------------------------------------------------


async def build_gap_report(
    project_id: str,
    gaps: list[dict],
    novelty: dict,
    trends: dict,
    contradictions: list[dict]
) -> dict:

    report = {
        "executive_summary": {
            "top_gap":
                gaps[0] if gaps else None,

            "novelty":
                novelty
        },

        "identified_gaps":
            gaps,

        "novelty_analysis":
            novelty,

        "temporal_trends":
            trends,

        "contradictions":
            contradictions,

        "recommended_directions":
            [
                gap.get("title")
                for gap in gaps[:5]
            ],

        "methodology_opportunities":
            [],

        "benchmarking_weaknesses":
            [],

        "evaluation_risks":
            [],

        "future_datasets":
            [],

        "commercial_opportunities":
            [],

        "conclusion":
            "Gap analysis completed."
    }

    async with AsyncSessionLocal() as db:

        existing_query = (
            select(ReviewReport)
            .where(
                ReviewReport.project_id
                == project_id
            )
        )

        existing_result = await db.execute(
            existing_query
        )

        reports = (
            existing_result.scalars()
            .all()
        )

        version = len(reports) + 1

        db_report = ReviewReport(
            project_id=project_id,
            version=version,
            content=report
        )

        db.add(db_report)

        await db.commit()

    return report


# -------------------------------------------------------------------
# Agent Logging
# -------------------------------------------------------------------


async def log_agent_run(
    project_id: str,
    status: str,
    output: dict | None = None,
    error: str | None = None
) -> None:

    async with AsyncSessionLocal() as db:

        run = AgentRun(
            project_id=project_id,
            agent_name="knowledge.gap_analysis",
            status=status,
            output=output,
            error=error
        )

        db.add(run)

        await db.commit()


# -------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------


async def run_gap_analysis(
    project_id: str
) -> dict:

    try:
        logger.info("knowledge.gap_analysis.started", project_id=project_id)

        async with AsyncSessionLocal() as db:

            query = select(Paper).where(
                Paper.project_id
                == project_id
            )

            result = await db.execute(query)

            papers = result.scalars().all()

        import asyncio

        async def process_single_paper(p):
            logger.debug("knowledge.extract_limitations.start", paper_id=p.id)
            try:
                chunks = await rag.retrieve(
                    query="limitations future work weaknesses",
                    project_id=project_id,
                    top_k=20
                )

                extracted = await extract_limitations(
                    {
                        "id": p.id
                    },
                    chunks
                )
                logger.debug("knowledge.extract_limitations.success", paper_id=p.id, count=len(extracted))
                return extracted
            except Exception as e:
                logger.error("knowledge.extract_limitations.failed", paper_id=p.id, error=str(e), exc_info=True)
                return []

        tasks = [process_single_paper(paper) for paper in papers]
        results = await asyncio.gather(*tasks)

        all_limitations = []
        for r in results:
            all_limitations.extend(r)

        clusters = cluster_limitations(
            all_limitations
        )

        logger.debug("knowledge.clusters", count=len(clusters))

        benchmark_gaps = (
            detect_benchmark_gaps(
                [
                    {
                        "parsed_json":
                            paper.parsed_json,

                        "year":
                            paper.year
                    }
                    for paper in papers
                ]
            )
        )

        trends = await analyze_temporal_trends(
            [
                {
                    "year": paper.year
                }
                for paper in papers
            ]
        )

        contradictions = (
            await find_contradictions()
        )

        graph_gaps = await graph.detect_gaps()

        synthesized_gaps = (
            await synthesize_gaps(
                clusters,
                benchmark_gaps,
                contradictions,
                graph_gaps
            )
        )

        logger.info("knowledge.synthesis_done", gap_count=len(synthesized_gaps))

        ranked_gaps = await graph.rank_gaps(
            synthesized_gaps
        )

        proposed_idea = (
            ranked_gaps[0]["title"]
            if ranked_gaps
            else "Novel research direction"
        )

        novelty = await score_novelty(
            proposed_idea,
            project_id
        )

        report = await build_gap_report(
            project_id=project_id,
            gaps=ranked_gaps,
            novelty=novelty,
            trends=trends,
            contradictions=contradictions
        )

        logger.info("knowledge.gap_analysis.success", project_id=project_id, gap_count=len(ranked_gaps))
        await log_agent_run(
            project_id=project_id,
            status="complete",
            output={
                "gap_count":
                    len(ranked_gaps)
            }
        )

        return report

    except Exception as error:

        logger.error("knowledge.gap_analysis.failed", project_id=project_id, error=str(error), exc_info=True)
        await log_agent_run(
            project_id=project_id,
            status="failed",
            error=str(error)
        )

        raise