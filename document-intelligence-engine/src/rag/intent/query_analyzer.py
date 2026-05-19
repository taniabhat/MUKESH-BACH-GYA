"""
src/rag/intent/query_analyzer.py
==================================
Intent Analyzer + Query Rewriter — Step 1 & 2 of the retrieval pipeline.

Intent classification:
  Classifies an incoming query into one of 7 research intents:
    methodology  — "How does the model work?" / "What architecture?"
    results      — "What accuracy did they achieve?" / "benchmark scores"
    dataset      — "What data was used?" / "training set"
    limitation   — "What are the weaknesses?" / "future work"
    comparison   — "How does X compare to Y?"
    definition   — "What is attention?" / "define transformer"
    general      — catch-all for broad queries

  Classification is rule-based (keyword patterns) + heuristic scoring.
  No LLM call needed for intent — keeps latency near-zero.

Query rewriting:
  Generates 3 reformulated variants of the original query to improve
  retrieval recall. Each variant targets a different aspect:
    Variant 1 — Technical expansion (adds domain synonyms)
    Variant 2 — Simplified version (removes jargon)
    Variant 3 — Aspect-flipped (different angle on the same question)

  Rule-based templates per intent — no LLM dependency.

Usage:
    from src.rag.intent.query_analyzer import QueryAnalyzer
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("What optimizer was used and why?")
    print(result.intent)       # "methodology"
    print(result.variants)     # ["...", "...", "..."]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Intent keyword map ─────────────────────────────────────────────────────────
_INTENT_PATTERNS: dict[str, list[re.Pattern]] = {
    "methodology": [
        re.compile(r"\b(how|architecture|model|method|approach|design|implement|train|network|layer|mechanism|algorithm|pipeline|framework|technique)\b", re.I),
    ],
    "results": [
        re.compile(r"\b(result|performance|accuracy|score|metric|benchmark|BLEU|ROUGE|F1|precision|recall|AUC|improve|beat|state.of.the.art|SOTA|outperform)\b", re.I),
    ],
    "dataset": [
        re.compile(r"\b(dataset|data|corpus|train|test|validation|split|sample|annotate|label|benchmark dataset|ImageNet|COCO|SQuAD|GLUE)\b", re.I),
    ],
    "limitation": [
        re.compile(r"\b(limit|weakness|drawback|future|issue|problem|challenge|fail|not handle|cannot|difficult|gap|shortcom)\b", re.I),
    ],
    "comparison": [
        re.compile(r"\b(compar|versus|vs\.?|differ|contrast|against|baseline|ablat|relative|better than|worse than|outperform)\b", re.I),
    ],
    "definition": [
        re.compile(r"\b(what is|define|definition|mean|explain|describe|introduc|background|concept|term|notation)\b", re.I),
    ],
}

# Template expansions per intent
_EXPANSIONS: dict[str, list[str]] = {
    "methodology": [
        "{q} architecture design neural network",
        "{q} implementation training procedure",
        "{q} technical approach method algorithm",
    ],
    "results": [
        "{q} performance evaluation benchmark scores",
        "{q} accuracy metrics quantitative results",
        "{q} experimental results comparison table",
    ],
    "dataset": [
        "{q} dataset training data corpus statistics",
        "{q} data collection preprocessing splits",
        "{q} benchmark evaluation dataset annotation",
    ],
    "limitation": [
        "{q} limitations future work open problems",
        "{q} weaknesses failure cases challenges",
        "{q} discussion conclusion caveats",
    ],
    "comparison": [
        "{q} comparison baseline ablation study",
        "{q} versus alternative methods performance",
        "{q} relative improvement competitive analysis",
    ],
    "definition": [
        "{q} definition concept background introduction",
        "{q} explanation overview notation",
        "{q} related work prior art survey",
    ],
    "general": [
        "{q} overview summary",
        "{q} key contributions findings",
        "{q} main results methodology",
    ],
}


@dataclass
class QueryAnalysis:
    original:  str
    intent:    str
    confidence: float            # 0.0 – 1.0
    variants:  list[str]         # 3 rewritten queries
    modalities: list[str]        # which Qdrant collections to search


class QueryAnalyzer:
    """
    Rule-based intent classifier and query rewriter.
    Adds zero latency to the retrieval pipeline (no model inference).
    """

    # Intent → which modalities to prioritise
    _MODALITY_MAP: dict[str, list[str]] = {
        "methodology":  ["text_chunks", "figure_chunks", "code_chunks", "equation_chunks"],
        "results":      ["text_chunks", "table_chunks", "figure_chunks"],
        "dataset":      ["text_chunks", "table_chunks"],
        "limitation":   ["text_chunks"],
        "comparison":   ["text_chunks", "table_chunks", "figure_chunks"],
        "definition":   ["text_chunks", "equation_chunks"],
        "general":      ["text_chunks", "figure_chunks", "table_chunks"],
    }

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Classify the query intent and generate 3 retrieval variants.
        """
        intent, confidence = self._classify(query)
        variants           = self._rewrite(query, intent)
        modalities         = self._MODALITY_MAP.get(intent, ["text_chunks"])

        return QueryAnalysis(
            original=query,
            intent=intent,
            confidence=confidence,
            variants=variants,
            modalities=modalities,
        )

    # ── Classification ─────────────────────────────────────────────────────────

    def _classify(self, query: str) -> tuple[str, float]:
        scores: dict[str, int] = {k: 0 for k in _INTENT_PATTERNS}

        for intent, patterns in _INTENT_PATTERNS.items():
            for pat in patterns:
                matches = pat.findall(query)
                scores[intent] += len(matches)

        total = sum(scores.values())
        if total == 0:
            return "general", 0.5

        best_intent = max(scores, key=lambda k: scores[k])
        confidence  = round(scores[best_intent] / total, 3)
        return best_intent, confidence

    # ── Query rewriting ────────────────────────────────────────────────────────

    @staticmethod
    def _rewrite(query: str, intent: str) -> list[str]:
        """Generate 3 query variants using intent-specific templates."""
        templates = _EXPANSIONS.get(intent, _EXPANSIONS["general"])
        variants: list[str] = []

        for tmpl in templates[:3]:
            variant = tmpl.format(q=query).strip()
            if variant != query:
                variants.append(variant)

        # Always include the original as a variant if < 3 generated
        while len(variants) < 3:
            variants.append(query)

        return variants[:3]
