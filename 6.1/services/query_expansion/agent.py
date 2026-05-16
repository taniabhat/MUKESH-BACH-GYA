"""
Query expansion agent for research retrieval.
"""

from __future__ import annotations

import json
import re

from typing import Optional

from research_discovery.config.settings import settings
from research_discovery.core.runtime import get_logger
from research_discovery.providers.factory import ProviderFactory

logger = get_logger(__name__)

DEFAULT_QUERY_COUNT = 10

settings.llm.request_timeout = 60

MIN_QUERY_COUNT = 3

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 120


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert academic research query expansion assistant.

Given a research idea, generate diverse search queries to maximize recall
across academic databases (OpenAlex, arXiv, Semantic Scholar).

Return ONLY a valid JSON array of strings. No preamble. No markdown fences.

Cover ALL of these angles:
1. Direct paraphrases
2. Terminology expansion
3. Acronym expansion
4. Adjacent subfields
5. Methodology variants
6. Benchmark/dataset variants
7. Foundational + emerging terminology

Rules:
- Each query must be 3–10 words
- Queries must differ meaningfully
- Use natural academic phrasing
"""


def build_user_prompt(
    idea: str,
    num_queries: int,
) -> str:

    return (
        f"Research idea: {idea}\n\n"
        f"Generate {num_queries} "
        f"diverse academic search queries."
    )


# ---------------------------------------------------------------------------
# Rule Expansion
# ---------------------------------------------------------------------------

EXPANSION_RULES = {
    "llm": [
        "large language model",
        "foundation model",
        "language model",
    ],
    "nlp": [
        "natural language processing",
        "computational linguistics",
        "text processing",
    ],
    "ml": [
        "machine learning",
        "statistical learning",
        "predictive modeling",
    ],
    "rag": [
        "retrieval augmented generation",
        "grounded generation",
        "retrieval enhanced generation",
    ],
    "agent": [
        "autonomous agent",
        "ai agent",
        "tool use agent",
    ],
}


class RuleBasedExpander:
    """Fallback rule-based query expansion."""

    @staticmethod
    def expand(
        idea: str,
        limit: int,
    ) -> list[str]:

        queries = [idea]

        lowered = idea.lower()

        for term, expansions in (
            EXPANSION_RULES.items()
        ):

            if term not in lowered:
                continue

            for expansion in expansions:

                variant = re.sub(
                    re.escape(term),
                    expansion,
                    lowered,
                    flags=re.IGNORECASE,
                )

                queries.append(
                    variant.strip()
                )

        queries.extend(
            RuleBasedExpander._structural_variants(
                idea
            )
        )

        return QueryCleaner.deduplicate(
            queries
        )[:limit]

    @staticmethod
    def _structural_variants(
        idea: str,
    ) -> list[str]:

        variants = []

        words = idea.split()

        if len(words) >= 3:

            variants.append(
                " ".join(words[:3])
            )

            variants.append(
                " ".join(words[-3:])
            )

        if len(words) >= 4:

            variants.append(
                " ".join(words[:4])
            )

        return variants


# ---------------------------------------------------------------------------
# LLM Expander
# ---------------------------------------------------------------------------

class LLMQueryExpander:
    """LLM-based query expansion."""
    
    def __init__(self):
        self.provider = ProviderFactory.create_llm_provider()

    async def expand(
        self,
        idea: str,
        num_queries: int,
    ) -> Optional[list[str]]:

        try:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(
                        idea,
                        num_queries,
                    ),
                },
            ]

            content = await self.provider.generate(messages)

            return QueryResponseParser.parse(
                content=content,
                original_idea=idea,
            )

        except Exception:
            logger.exception(
                "LLM query expansion failed",
            )
            return None

# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

class QueryResponseParser:
    """Parses and validates LLM responses."""

    @staticmethod
    def parse(
        content: str,
        original_idea: str,
    ) -> Optional[list[str]]:

        if not content:
            return None

        cleaned = (
            QueryResponseParser
            ._strip_markdown(
                content
            )
        )

        extracted = (
            QueryResponseParser
            ._extract_json_array(
                cleaned
            )
        )

        if not extracted:
            return None

        try:

            parsed = json.loads(
                extracted
            )

        except json.JSONDecodeError:

            logger.warning(
                "LLM JSON parse failed"
            )

            return None

        if not isinstance(parsed, list):
            return None

        queries = [
            QueryCleaner.clean(query)
            for query in parsed
            if isinstance(query, str)
        ]

        queries = [
            query
            for query in queries
            if QueryCleaner.is_valid(
                query
            )
        ]

        queries = QueryCleaner.deduplicate(
            queries
        )

        if len(queries) < MIN_QUERY_COUNT:

            logger.warning(
                "Insufficient query expansion "
                "query_count=%s",
                len(queries),
            )

            return None

        logger.info(
            "Expanded query '%s' into %s queries",
            original_idea,
            len(queries),
        )

        return queries

    @staticmethod
    def _strip_markdown(
        content: str,
    ) -> str:

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.MULTILINE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
            flags=re.MULTILINE,
        )

        return content.strip()

    @staticmethod
    def _extract_json_array(
        content: str,
    ) -> Optional[str]:

        start = content.find("[")

        end = content.rfind("]")

        if start == -1 or end == -1:
            return None

        if end <= start:
            return None

        return content[start:end + 1]


# ---------------------------------------------------------------------------
# Query Cleaning
# ---------------------------------------------------------------------------

class QueryCleaner:
    """Query normalization and filtering."""

    @staticmethod
    def clean(
        query: str,
    ) -> str:

        query = query.strip()

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query

    @staticmethod
    def is_valid(
        query: str,
    ) -> bool:

        if not query:
            return False

        return (
            MIN_QUERY_LENGTH
            <= len(query)
            <= MAX_QUERY_LENGTH
        )

    @staticmethod
    def deduplicate(
        queries: list[str],
    ) -> list[str]:

        seen = set()

        deduplicated = []

        for query in queries:

            normalized = (
                query.lower()
            )

            if normalized in seen:
                continue

            seen.add(normalized)

            deduplicated.append(query)

        return deduplicated


# ---------------------------------------------------------------------------
# Public Agent
# ---------------------------------------------------------------------------

class QueryExpansionAgent:
    """
    Expands research ideas into retrieval queries.
    """

    def __init__(
        self,
        num_queries: int = DEFAULT_QUERY_COUNT,
    ):

        self.num_queries = num_queries

        self._llm_expander = (
            LLMQueryExpander()
        )

    async def expand(
        self,
        research_idea: str,
    ) -> list[str]:

        logger.info(
            "Expanding research idea='%s'",
            research_idea,
        )

        queries = (
            await self._llm_expander.expand(
                idea=research_idea,
                num_queries=(
                    self.num_queries
                ),
            )
        )

        if not queries:

            logger.info(
                "Falling back to "
                "rule-based expansion"
            )

            queries = (
                RuleBasedExpander.expand(
                    research_idea,
                    limit=self.num_queries,
                )
            )

        if research_idea not in queries:

            queries.insert(
                0,
                research_idea,
            )

        queries = QueryCleaner.deduplicate(
            queries
        )

        logger.info(
            "Final expanded query count=%s",
            len(queries),
        )

        return queries