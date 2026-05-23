import re
from collections import defaultdict

from neo4j import AsyncGraphDatabase

from config import get_settings
from core.document import DocumentResult
from core.embeddings import embed_single
from core.logging import get_logger


settings = get_settings()

logger = get_logger("core.graph")


# -------------------------------------------------------------------
# Driver Singleton
# -------------------------------------------------------------------


_driver = None


def get_driver():

    global _driver

    if _driver is None:

        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(
                settings.NEO4J_USER,
                settings.NEO4J_PASSWORD
            )
        )

    return _driver


# -------------------------------------------------------------------
# Paper Node
# -------------------------------------------------------------------


async def build_paper_node(
    paper: dict
) -> None:

    logger.info("graph.build_paper_node.started", paper_id=paper.get("id"))
    query = """
    MERGE (p:Paper {id: $id})

    SET p.title = $title,
        p.year = $year,
        p.doi = $doi,
        p.venue = $venue
    """

    driver = get_driver()
    async with driver.session() as session:

        await session.run(
            query,
            {
                "id": paper.get("id"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "venue": paper.get("venue")
            }
        )

    logger.info("graph.build_paper_node.success", paper_id=paper.get("id"))


# -------------------------------------------------------------------
# Relationship Builder
# -------------------------------------------------------------------


def extract_entities(
    parsed_doc: DocumentResult
) -> dict:

    methods = []
    datasets = []
    metrics = []
    limitations = []
    contributions = []

    for section in parsed_doc.sections:

        body = section.get(
            "body",
            ""
        )

        heading = (
            section.get(
                "heading",
                ""
            ).lower()
        )

        if "method" in heading:
            methods.append(body[:500])

        if "dataset" in heading:
            datasets.append(body[:500])

        if "metric" in heading:
            # Extract basic metric value if present
            # Look for numbers near keywords like 'accuracy', 'f1', 'score'
            match = re.search(r'(accuracy|f1|precision|recall|score|bleu|rouge)[^\d]{0,20}(\d+\.?\d*)', body.lower())
            if match:
                name = match.group(1).title()
                try:
                    value = float(match.group(2))
                    metrics.append({"name": name, "value": value})
                except ValueError:
                    pass

        if "limitation" in heading:
            limitations.append(body[:500])

        if "contribution" in heading:
            contributions.append(body[:500])

    return {
        "methods": methods,
        "datasets": datasets,
        "metrics": metrics,
        "limitations": limitations,
        "contributions": contributions
    }


async def build_relationships(
    paper: dict,
    parsed_doc: DocumentResult
) -> None:

    logger.info("graph.build_relationships.started", paper_id=paper.get("id"))
    entities = extract_entities(parsed_doc)

    driver = get_driver()

    async with driver.session() as session:

        async with await session.begin_transaction() as tx:

            # --------------------------------------------------------
            # Authors
            # --------------------------------------------------------

            for author in parsed_doc.authors:

                author_name = (
                    author.get("name")
                    if isinstance(author, dict)
                    else str(author)
                )

                await tx.run(
                    """
                    MERGE (a:Author {name: $author})

                    WITH a

                    MATCH (p:Paper {id: $paper_id})

                    MERGE (p)-[:AUTHORED_BY]->(a)
                    """,
                    {
                        "author": author_name,
                        "paper_id": paper["id"]
                    }
                )

            # --------------------------------------------------------
            # Methods
            # --------------------------------------------------------

            for method in entities["methods"]:

                embedding = await embed_single(method)

                await tx.run(
                    """
                    MERGE (m:Method {
                        description: $description
                    })

                    SET m.embedding = $embedding

                    WITH m

                    MATCH (p:Paper {id: $paper_id})

                    MERGE (p)-[:USES_METHOD]->(m)
                    """,
                    {
                        "description": method,
                        "embedding": embedding,
                        "paper_id": paper["id"]
                    }
                )

            # --------------------------------------------------------
            # Datasets
            # --------------------------------------------------------

            for dataset in entities["datasets"]:

                await tx.run(
                    """
                    MERGE (d:Dataset {
                        name: $dataset
                    })

                    WITH d

                    MATCH (p:Paper {id: $paper_id})

                    MERGE (p)-[:TESTED_ON]->(d)
                    """,
                    {
                        "dataset": dataset,
                        "paper_id": paper["id"]
                    }
                )

            # --------------------------------------------------------
            # Metrics
            # --------------------------------------------------------

            for metric_data in entities["metrics"]:

                await tx.run(
                    """
                    MERGE (m:Metric {
                        name: $metric
                    })

                    WITH m

                    MATCH (p:Paper {id: $paper_id})

                    MERGE (p)-[:REPORTS_METRIC {
                        value: $value
                    }]->(m)
                    """,
                    {
                        "metric": metric_data["name"],
                        "value": metric_data["value"],
                        "paper_id": paper["id"]
                    }
                )

            # --------------------------------------------------------
            # Limitations
            # --------------------------------------------------------

            for limitation in entities["limitations"]:

                await tx.run(
                    """
                    MERGE (l:Limitation {
                        description: $limitation
                    })

                    WITH l

                    MATCH (p:Paper {id: $paper_id})

                    MERGE (p)-[:LIMITED_BY]->(l)
                    """,
                    {
                        "limitation": limitation,
                        "paper_id": paper["id"]
                    }
                )

            # --------------------------------------------------------
            # Contributions
            # --------------------------------------------------------

            for contribution in entities["contributions"]:

                await tx.run(
                    """
                    MERGE (c:Contribution {
                        description: $contribution
                    })

                    WITH c

                    MATCH (p:Paper {id: $paper_id})

                    MERGE (p)-[:CONTRIBUTES]->(c)
                    """,
                    {
                        "contribution": contribution,
                        "paper_id": paper["id"]
                    }
                )

            # --------------------------------------------------------
            # Citations
            # --------------------------------------------------------

            for reference in parsed_doc.references:

                doi = reference.get("doi")

                if not doi:
                    continue

                await tx.run(
                    """
                    MATCH (source:Paper {
                        id: $source_id
                    })

                    MATCH (target:Paper {
                        doi: $doi
                    })

                    MERGE (source)-[:CITES]->(target)
                    """,
                    {
                        "source_id": paper["id"],
                        "doi": doi
                    }
                )

            await tx.commit()
            
    logger.info("graph.build_relationships.success", paper_id=paper.get("id"))


# -------------------------------------------------------------------
# Gap Detection
# -------------------------------------------------------------------


async def detect_gaps() -> list[dict]:

    logger.debug("graph.detect_gaps.started")
    query = """
    MATCH (l:Limitation)

    WHERE NOT EXISTS {
        MATCH (l)<-[:LIMITED_BY]-(:Paper)
              -[:CONTRIBUTES*1..3]-(:Contribution)
    }

    RETURN
        l.description AS limitation_description,
        count(l) AS frequency
    """

    driver = get_driver()
    async with driver.session() as session:

        results = await session.run(query)

        gaps = []

        async for row in results:

            gaps.append({
                "limitation_description":
                    row["limitation_description"],

                "supporting_paper_ids": [],

                "frequency":
                    row["frequency"]
            })

        logger.debug("graph.detect_gaps.success", gap_count=len(gaps))
        return gaps


# -------------------------------------------------------------------
# Contradiction Finder
# -------------------------------------------------------------------


async def find_contradictions() -> list[dict]:

    logger.debug("graph.find_contradictions.started")
    query = """
    MATCH
        (p1:Paper)-[r1:REPORTS_METRIC]->(m:Metric),
        (p2:Paper)-[r2:REPORTS_METRIC]->(m)

    WHERE p1.id <> p2.id

    AND abs(r1.value - r2.value)
        > (0.05 * r1.value)

    RETURN
        p1.id AS paper_a,
        p2.id AS paper_b,
        m.name AS metric,
        r1.value AS value_a,
        r2.value AS value_b
    """

    driver = get_driver()
    async with driver.session() as session:

        results = await session.run(query)

        contradictions = []

        async for row in results:

            contradictions.append({
                "paper_a": row["paper_a"],
                "paper_b": row["paper_b"],
                "metric": row["metric"],
                "dataset": None,
                "value_a": row["value_a"],
                "value_b": row["value_b"]
            })

        logger.debug("graph.find_contradictions.success", contradiction_count=len(contradictions))
        return contradictions


# -------------------------------------------------------------------
# Novelty Reasoning
# -------------------------------------------------------------------


import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return 0.0 if denom == 0 else float(np.dot(va, vb) / denom)


async def check_novelty(
    proposed_method: str
) -> dict:

    logger.debug("graph.check_novelty.started")
    embedding = await embed_single(proposed_method)

    query = """
    MATCH (m:Method)

    RETURN
        m.description AS description,
        m.embedding AS embedding
    """

    similarities = []

    driver = get_driver()
    async with driver.session() as session:

        results = await session.run(query)

        async for row in results:

            stored_embedding = row["embedding"]

            similarity = cosine_similarity(
                embedding,
                stored_embedding
            )

            similarities.append({
                "description":
                    row["description"],

                "score":
                    similarity
            })

    similarities.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    top_match = similarities[0] if similarities else None

    novelty_score = (
        1 - top_match["score"]
        if top_match
        else 1.0
    )

    logger.debug("graph.check_novelty.success", novelty_score=novelty_score)
    return {
        "novelty_score": novelty_score,
        "similar_methods": similarities[:5],
        "closest_paper": None
    }


# -------------------------------------------------------------------
# Citation Lineage
# -------------------------------------------------------------------


async def get_citation_lineage(
    paper_id: str,
    depth: int = 2
) -> list[dict]:

    logger.debug("graph.get_citation_lineage.started", paper_id=paper_id, depth=depth)
    query = f"""
    MATCH path =
        (p:Paper {{id: $paper_id}})
        -[:CITES*1..{depth}]->
        (cited:Paper)

    RETURN
        cited.id AS paper_id,
        cited.title AS title,
        length(path) AS depth
    """

    driver = get_driver()
    async with driver.session() as session:

        results = await session.run(
            query,
            {
                "paper_id": paper_id
            }
        )

        lineage = []

        async for row in results:

            lineage.append({
                "paper_id": row["paper_id"],
                "title": row["title"],
                "depth": row["depth"]
            })

        logger.debug("graph.get_citation_lineage.success", lineage_count=len(lineage))
        return lineage


# -------------------------------------------------------------------
# Trend Analysis
# -------------------------------------------------------------------


async def analyze_trends(
    method_name: str | None = None
) -> dict:

    logger.debug("graph.analyze_trends.started", target_method=method_name)
    if method_name:

        query = """
        MATCH
            (p:Paper)-[:USES_METHOD]->
            (m:Method)

        WHERE m.description CONTAINS $method

        RETURN
            p.year AS year,
            count(p) AS count
        """

        params = {
            "method": method_name
        }

    else:

        query = """
        MATCH (p:Paper)

        RETURN
            p.year AS year,
            count(p) AS count
        """

        params = {}

    year_counts = {}

    driver = get_driver()
    async with driver.session() as session:

        results = await session.run(
            query,
            params
        )

        async for row in results:

            year = row["year"]

            if year:
                year_counts[year] = row["count"]

    sorted_years = sorted(year_counts.keys())

    trend = "stable"

    if len(sorted_years) >= 2:

        first = year_counts[sorted_years[0]]

        last = year_counts[sorted_years[-1]]

        if last > first:
            trend = "growing"

        elif last < first:
            trend = "declining"

    return {
        "method": method_name,
        "year_counts": year_counts,
        "trend": trend
    }


# -------------------------------------------------------------------
# Gap Ranking
# -------------------------------------------------------------------


async def rank_gaps(
    gaps: list[dict]
) -> list[dict]:

    logger.debug("graph.rank_gaps.started")
    contradictions = await find_contradictions()

    contradiction_count = len(contradictions)

    ranked = []

    for gap in gaps:

        frequency = gap.get(
            "frequency",
            1
        )

        age_score = 1.0

        contradiction_score = (
            contradiction_count * 0.1
        )

        composite_score = (
            (frequency * 0.6)
            + (age_score * 0.2)
            + (contradiction_score * 0.2)
        )

        gap["score"] = composite_score

        ranked.append(gap)

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    logger.debug("graph.rank_gaps.success")
    return ranked