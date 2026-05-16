"""
Query Expansion Agent.

Takes a single research idea and generates a rich set of retrieval queries
covering: paraphrases, terminology expansion, acronym expansion,
adjacent subfields, methodology variants, and benchmark variants.

Falls back to rule-based expansion when no LLM key is configured.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from research_discovery.config.settings import settings
from research_discovery.core.utils import get_logger, get_http_client

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt for LLM-based expansion
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert research query expansion assistant.
Given a research idea, generate diverse search queries to maximize recall
across academic databases. Return ONLY a JSON array of strings — no preamble,
no markdown fences.

Cover all of these angles:
1. Direct paraphrases (2-3 variants)
2. Terminology expansion (synonyms, alternative phrasings)
3. Acronym expansions and contractions
4. Adjacent subfields / related disciplines
5. Methodology / technique expansions
6. Benchmark / dataset expansions
7. Temporal variants (historical origins + emerging work)

Generate 8–12 unique queries. Each query should be 3–10 words.
Return format: ["query 1", "query 2", ...]"""

# ---------------------------------------------------------------------------
# Rule-based fallback expansion
# ---------------------------------------------------------------------------

EXPANSION_RULES: dict[str, list[str]] = {
    "llm": ["large language model", "foundation model", "language model"],
    "nlp": ["natural language processing", "computational linguistics"],
    "ml": ["machine learning", "statistical learning"],
    "dl": ["deep learning", "neural network"],
    "rl": ["reinforcement learning", "reward learning"],
    "transformer": ["attention mechanism", "encoder decoder", "self attention"],
    "code review": ["software review", "pull request analysis", "code quality"],
    "rag": ["retrieval augmented generation", "retrieval enhanced generation"],
    "agent": ["autonomous agent", "ai agent", "language agent"],
}


def _rule_based_expand(idea: str, n: int = 8) -> list[str]:
    """Simple rule-based fallback when no LLM is available."""
    queries = [idea]
    idea_lower = idea.lower()

    for term, expansions in EXPANSION_RULES.items():
        if term in idea_lower:
            for exp in expansions:
                variant = re.sub(re.escape(term), exp, idea_lower, flags=re.IGNORECASE)
                queries.append(variant)

    # Add some structural variants
    words = idea.split()
    if len(words) >= 3:
        queries.append(" ".join(words[:3]))  # leading keywords
        queries.append(" ".join(words[-3:]))  # trailing keywords

    # Deduplicate, preserve order
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            result.append(q)

    return result[:n]


# ---------------------------------------------------------------------------
# LLM-based expansion
# ---------------------------------------------------------------------------

async def _llm_expand(idea: str, n: int = 10) -> Optional[list[str]]:
    """Call the Anthropic API to generate expanded queries."""
    api_key = settings.api.anthropic_api_key
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY set — using rule-based expansion.")
        return None

    payload = {
        "model": settings.api.llm_model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Research idea: {idea}\n\nGenerate {n} search queries.",
            }
        ],
    }

    try:
        async with get_http_client(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text content
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        # Parse JSON array
        text = text.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        queries: list[str] = json.loads(text)
        if not isinstance(queries, list):
            raise ValueError("LLM did not return a JSON array")

        logger.info(f"LLM expanded '{idea}' → {len(queries)} queries")
        return [str(q).strip() for q in queries if q]

    except Exception as exc:
        logger.error(f"LLM expansion failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class QueryExpansionAgent:
    """
    Expands a research idea into multiple retrieval queries.

    Strategy priority:
    1. LLM (Anthropic API) — highest quality
    2. Rule-based — fast fallback
    """

    def __init__(self, num_queries: int = 10):
        self.num_queries = num_queries

    async def expand(self, research_idea: str) -> list[str]:
        """
        Returns a list of diverse search queries for the given idea.
        Always includes the original idea as the first query.
        """
        logger.info(f"Expanding query: '{research_idea}'")

        # Try LLM first
        queries = await _llm_expand(research_idea, n=self.num_queries)

        # Fall back to rules
        if not queries:
            queries = _rule_based_expand(research_idea, n=self.num_queries)

        # Always ensure original is present
        if research_idea not in queries:
            queries.insert(0, research_idea)

        # Deduplicate
        seen: set[str] = set()
        deduped: list[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                deduped.append(q)

        logger.info(f"Final expanded query set ({len(deduped)} queries): {deduped}")
        return deduped