"""
Deduplication engine for canonical paper records.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from research_discovery.core.runtime import get_logger
from research_discovery.models.paper import (
    ExternalIDs,
    Paper,
)

logger = get_logger(__name__)

FUZZY_THRESHOLD = 0.94


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but",
    "in", "on", "at", "to", "for",
    "of", "with", "by", "from",
    "is", "are", "was", "were",
    "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "could",
    "should", "may", "might",
    "shall", "can",
    "it", "its",
    "this", "that",
    "these", "those",
    "via", "using", "based",
})


# ---------------------------------------------------------------------------
# Fuzzy Matching Backend
# ---------------------------------------------------------------------------

try:

    from rapidfuzz import fuzz as _fuzz

    def fuzzy_ratio(
        left: str,
        right: str,
    ) -> float:

        return (
            _fuzz.token_sort_ratio(
                left,
                right,
            )
            / 100.0
        )

    USING_RAPIDFUZZ = True

except ImportError:

    import difflib

    def fuzzy_ratio(
        left: str,
        right: str,
    ) -> float:

        return difflib.SequenceMatcher(
            None,
            left,
            right,
        ).ratio()

    USING_RAPIDFUZZ = False

    logger.warning(
        "rapidfuzz unavailable — "
        "falling back to difflib"
    )


# ---------------------------------------------------------------------------
# Dedup Stats
# ---------------------------------------------------------------------------

class MatchType(str, Enum):
    DOI = "doi"
    FINGERPRINT = "fingerprint"
    FUZZY = "fuzzy"
    NEW = "new"


@dataclass
class DedupStats:
    doi: int = 0
    fingerprint: int = 0
    fuzzy: int = 0
    new: int = 0


# ---------------------------------------------------------------------------
# Normalization Utilities
# ---------------------------------------------------------------------------

def normalize_title(
    title: str,
) -> str:

    normalized = unicodedata.normalize(
        "NFKD",
        title,
    )

    normalized = normalized.lower()

    normalized = re.sub(
        r"[^\w\s-]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\b\d+\b",
        "",
        normalized,
    )

    tokens = normalized.split()

    filtered_tokens = [
        token
        for token in tokens
        if (
            token not in STOPWORDS
            and len(token) > 1
        )
    ]

    return " ".join(filtered_tokens)


def normalize_doi(
    doi: Optional[str],
) -> Optional[str]:

    if not doi:
        return None

    normalized = doi.strip().lower()

    normalized = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        normalized,
    )

    return normalized or None


def generate_title_fingerprint(
    normalized_title: str,
) -> str:

    return hashlib.sha256(
        normalized_title.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Deduplication Engine
# ---------------------------------------------------------------------------

class DeduplicationEngine:
    """
    Multi-stage paper deduplication.
    """

    def __init__(
        self,
        fuzzy_threshold: float = FUZZY_THRESHOLD,
    ):

        self.fuzzy_threshold = fuzzy_threshold

    def deduplicate(
        self,
        papers: list[Paper],
    ) -> list[Paper]:

        if not papers:
            return []

        stats = DedupStats()

        canonical_papers = []

        doi_index = {}
        fingerprint_index = {}

        normalized_titles = []

        for paper in papers:

            normalized_title = normalize_title(
                paper.title
            )

            fingerprint = (
                generate_title_fingerprint(
                    normalized_title
                )
            )

            paper.title_fingerprint = fingerprint

            matched_paper = (
                self._find_duplicate(
                    paper=paper,
                    normalized_title=(
                        normalized_title
                    ),
                    fingerprint=fingerprint,
                    canonical_papers=(
                        canonical_papers
                    ),
                    normalized_titles=(
                        normalized_titles
                    ),
                    doi_index=doi_index,
                    fingerprint_index=(
                        fingerprint_index
                    ),
                    stats=stats,
                )
            )

            if matched_paper:

                merged = self._merge_papers(
                    matched_paper,
                    paper,
                )

                existing_index = (
                    canonical_papers.index(
                        matched_paper
                    )
                )

                canonical_papers[
                    existing_index
                ] = merged

                continue

            canonical_index = len(
                canonical_papers
            )

            canonical_papers.append(paper)

            normalized_titles.append(
                normalized_title
            )

            doi = normalize_doi(
                paper.external_ids.doi
            )

            if doi:
                doi_index[doi] = canonical_index

            fingerprint_index[
                fingerprint
            ] = canonical_index

            stats.new += 1

        self._log_stats(
            stats=stats,
            input_count=len(papers),
            output_count=len(
                canonical_papers
            ),
        )

        return canonical_papers

    def _find_duplicate(
        self,
        paper: Paper,
        normalized_title: str,
        fingerprint: str,
        canonical_papers: list[Paper],
        normalized_titles: list[str],
        doi_index: dict[str, int],
        fingerprint_index: dict[str, int],
        stats: DedupStats,
    ) -> Optional[Paper]:

        doi = normalize_doi(
            paper.external_ids.doi
        )

        # DOI match
        if doi and doi in doi_index:

            stats.doi += 1

            return canonical_papers[
                doi_index[doi]
            ]

        # Fingerprint match
        if fingerprint in fingerprint_index:

            stats.fingerprint += 1

            return canonical_papers[
                fingerprint_index[
                    fingerprint
                ]
            ]

        # Fuzzy match
        for index, existing_title in enumerate(
            normalized_titles
        ):

            similarity = fuzzy_ratio(
                normalized_title,
                existing_title,
            )

            if (
                similarity
                >= self.fuzzy_threshold
            ):

                stats.fuzzy += 1

                return canonical_papers[index]

        return None

    def _merge_papers(
        self,
        left: Paper,
        right: Paper,
    ) -> Paper:

        better = self._select_better_paper(
            left,
            right,
        )

        worse = (
            right
            if better is left
            else left
        )

        merged_queries = sorted(
            set(
                better.retrieved_from_queries
                + worse.retrieved_from_queries
            )
        )

        better.retrieved_from_queries = (
            merged_queries
        )

        self._fill_missing_fields(
            target=better,
            source=worse,
        )

        return better

    @staticmethod
    def _select_better_paper(
        left: Paper,
        right: Paper,
    ) -> Paper:

        left_score = (
            DeduplicationEngine._metadata_score(
                left
            )
        )

        right_score = (
            DeduplicationEngine._metadata_score(
                right
            )
        )

        return (
            left
            if left_score >= right_score
            else right
        )

    @staticmethod
    def _metadata_score(
        paper: Paper,
    ) -> int:

        score = 0

        fields = [
            paper.abstract,
            paper.venue,
            paper.year,
            paper.pdf_url,
            paper.external_ids.doi,
            paper.external_ids.arxiv,
            paper.authors,
        ]

        for field in fields:
            if field:
                score += 1

        score += min(
            paper.citation_count,
            1000,
        ) // 100

        return score

    @staticmethod
    def _fill_missing_fields(
        target: Paper,
        source: Paper,
    ) -> None:

        if (
            not target.abstract
            and source.abstract
        ):
            target.abstract = source.abstract

        if (
            not target.venue
            and source.venue
        ):
            target.venue = source.venue

        if (
            not target.year
            and source.year
        ):
            target.year = source.year

        if (
            not target.pdf_url
            and source.pdf_url
        ):
            target.pdf_url = source.pdf_url

        if (
            not target.authors
            and source.authors
        ):
            target.authors = source.authors

        target.citation_count = max(
            target.citation_count,
            source.citation_count,
        )

        DeduplicationEngine._merge_external_ids(
            target.external_ids,
            source.external_ids,
        )

    @staticmethod
    def _merge_external_ids(
        target: ExternalIDs,
        source: ExternalIDs,
    ) -> None:

        if (
            not target.doi
            and source.doi
        ):
            target.doi = source.doi

        if (
            not target.arxiv
            and source.arxiv
        ):
            target.arxiv = source.arxiv

        if (
            not target.semantic_scholar
            and source.semantic_scholar
        ):
            target.semantic_scholar = (
                source.semantic_scholar
            )

        if (
            not target.openalex
            and source.openalex
        ):
            target.openalex = source.openalex

    @staticmethod
    def _log_stats(
        stats: DedupStats,
        input_count: int,
        output_count: int,
    ) -> None:

        removed = (
            input_count - output_count
        )

        logger.info(
            "Dedup completed "
            "input=%s output=%s removed=%s "
            "doi=%s fingerprint=%s fuzzy=%s",
            input_count,
            output_count,
            removed,
            stats.doi,
            stats.fingerprint,
            stats.fuzzy,
        )