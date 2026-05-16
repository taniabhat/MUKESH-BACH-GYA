"""
Deduplication Engine.

Three-layer dedup strategy:
1. DOI exact match (normalized)
2. Title fingerprint hash (normalized + stopword-removed)
3. Fuzzy title similarity (RapidFuzz, threshold 0.94)

Papers from multiple sources or queries often appear multiple times.
Dedup runs after retrieval, before embedding generation.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Optional

from research_discovery.core.utils import get_logger
from research_discovery.models.paper import Paper

logger = get_logger(__name__)

# English stopwords for title normalization
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "it", "its", "this", "that", "these", "those", "via", "using", "based",
})

# Try to import rapidfuzz; fall back to difflib if not available
try:
    from rapidfuzz import fuzz as _fuzz
    def _fuzzy_ratio(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b) / 100.0
    _HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    def _fuzzy_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()
    _HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed — using difflib for fuzzy dedup (slower)")


def _normalize_title(title: str) -> str:
    """
    Lowercase, remove punctuation, remove stopwords, collapse spaces.
    Returns a canonical form for fingerprinting and fuzzy comparison.
    """
    # Unicode normalization
    title = unicodedata.normalize("NFKD", title)
    # Lowercase
    title = title.lower()
    # Remove punctuation except hyphens (important in compound terms)
    title = re.sub(r"[^\w\s-]", " ", title)
    # Remove digits-only tokens (version numbers etc.)
    title = re.sub(r"\b\d+\b", "", title)
    # Tokenize
    tokens = title.split()
    # Remove stopwords
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def _title_fingerprint(title: str) -> str:
    """Returns a stable hash for a normalized title."""
    normalized = _normalize_title(title)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi if doi else None


class DeduplicationEngine:
    """
    Deduplicates a list of Paper objects.

    Merge strategy: when two papers are identified as duplicates,
    keep the one with more complete metadata and merge retrieved_from_queries.
    """

    def __init__(self, fuzzy_threshold: float = 0.94):
        self.fuzzy_threshold = fuzzy_threshold

    def deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """
        Returns deduplicated list of papers.
        Preserves insertion order of the first occurrence.
        """
        if not papers:
            return []

        # Add fingerprints
        for paper in papers:
            if not paper.title_fingerprint:
                paper.title_fingerprint = _title_fingerprint(paper.title)

        canonical: list[Paper] = []  # Result set (one paper per cluster)
        doi_index: dict[str, int] = {}        # doi -> index in canonical
        fingerprint_index: dict[str, int] = {}  # fingerprint -> index in canonical
        normalized_titles: list[str] = []      # for fuzzy comparison

        def _merge(existing: Paper, duplicate: Paper) -> None:
            """Merge duplicate's provenance into existing."""
            for q in duplicate.retrieved_from_queries:
                if q not in existing.retrieved_from_queries:
                    existing.retrieved_from_queries.append(q)
            # Prefer higher citation count data
            if duplicate.citation_count > existing.citation_count:
                existing.citation_count = duplicate.citation_count
            # Fill missing fields
            if not existing.abstract and duplicate.abstract:
                existing.abstract = duplicate.abstract
            if not existing.venue and duplicate.venue:
                existing.venue = duplicate.venue
            if not existing.year and duplicate.year:
                existing.year = duplicate.year
            if not existing.pdf_url and duplicate.pdf_url:
                existing.pdf_url = duplicate.pdf_url
            if not existing.external_ids.doi and duplicate.external_ids.doi:
                existing.external_ids.doi = duplicate.external_ids.doi
            if not existing.external_ids.arxiv and duplicate.external_ids.arxiv:
                existing.external_ids.arxiv = duplicate.external_ids.arxiv
            if not existing.external_ids.semantic_scholar and duplicate.external_ids.semantic_scholar:
                existing.external_ids.semantic_scholar = duplicate.external_ids.semantic_scholar

        stats = {"doi": 0, "fingerprint": 0, "fuzzy": 0, "new": 0}

        for paper in papers:
            doi = _normalize_doi(paper.external_ids.doi)
            fp = paper.title_fingerprint
            norm = _normalize_title(paper.title)

            # Step 1: DOI exact match
            if doi and doi in doi_index:
                _merge(canonical[doi_index[doi]], paper)
                stats["doi"] += 1
                continue

            # Step 2: Title fingerprint
            if fp in fingerprint_index:
                _merge(canonical[fingerprint_index[fp]], paper)
                stats["fingerprint"] += 1
                continue

            # Step 3: Fuzzy similarity against existing
            matched = False
            for idx, existing_norm in enumerate(normalized_titles):
                if _fuzzy_ratio(norm, existing_norm) >= self.fuzzy_threshold:
                    _merge(canonical[idx], paper)
                    stats["fuzzy"] += 1
                    matched = True
                    break

            if not matched:
                # New unique paper
                idx = len(canonical)
                canonical.append(paper)
                normalized_titles.append(norm)
                if doi:
                    doi_index[doi] = idx
                fingerprint_index[fp] = idx
                stats["new"] += 1

        total_in = len(papers)
        total_out = len(canonical)
        removed = total_in - total_out
        logger.info(
            f"Dedup: {total_in} → {total_out} papers "
            f"(removed {removed}: doi={stats['doi']}, "
            f"fingerprint={stats['fingerprint']}, fuzzy={stats['fuzzy']})"
        )

        return canonical