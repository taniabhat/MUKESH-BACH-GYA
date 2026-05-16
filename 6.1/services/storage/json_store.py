"""
Storage Service — JSON-based persistence for discovery results.

Designed to plug in PostgreSQL + pgvector or Qdrant later.
Currently writes clean JSON to disk for portability.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from research_discovery.core.utils import get_logger
from research_discovery.models.paper import DiscoveryResult, Paper

logger = get_logger(__name__)

DEFAULT_STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/research_discovery"))


class JSONStorageService:
    """Persist and retrieve discovery results as JSON files."""

    def __init__(self, storage_dir: Path = DEFAULT_STORAGE_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: DiscoveryResult) -> str:
        """
        Save a DiscoveryResult to disk.
        Returns the file path.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_idea = result.query.original_idea[:40].replace(" ", "_").replace("/", "-")
        filename = f"{timestamp}_{safe_idea}.json"
        filepath = self.storage_dir / filename

        # Strip embeddings from saved output (large, not needed for review)
        data = result.model_dump()
        for tier in ("highly_relevant", "relevant_background", "adjacent_work", "historical_foundations"):
            for paper in data.get(tier, []):
                paper["embedding"] = []  # strip before save

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Result saved to {filepath}")
        return str(filepath)

    def load_result(self, filepath: str) -> Optional[DiscoveryResult]:
        """Load a previously saved result."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DiscoveryResult(**data)
        except Exception as exc:
            logger.error(f"Failed to load result from {filepath}: {exc}")
            return None

    def list_results(self) -> list[dict]:
        """List all saved results with basic metadata."""
        results = []
        for filepath in sorted(self.storage_dir.glob("*.json"), reverse=True):
            try:
                stat = filepath.stat()
                results.append({
                    "filename": filepath.name,
                    "path": str(filepath),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                })
            except Exception:
                pass
        return results

    def export_csv(self, result: DiscoveryResult, output_path: str) -> str:
        """Export papers to CSV for spreadsheet analysis."""
        import csv
        all_papers = result.all_papers()

        fieldnames = [
            "tier", "title", "year", "venue", "authors",
            "citation_count", "final_score", "semantic_similarity",
            "doi", "arxiv_id", "pdf_url", "abstract_preview",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for paper in all_papers:
                author_names = ", ".join(a.name for a in paper.authors[:3])
                if len(paper.authors) > 3:
                    author_names += " et al."
                writer.writerow({
                    "tier": paper.tier or "",
                    "title": paper.title,
                    "year": paper.year or "",
                    "venue": paper.venue or "",
                    "authors": author_names,
                    "citation_count": paper.citation_count,
                    "final_score": round(paper.final_score, 4),
                    "semantic_similarity": round(paper.ranking_features.semantic_similarity, 4),
                    "doi": paper.external_ids.doi or "",
                    "arxiv_id": paper.external_ids.arxiv or "",
                    "pdf_url": paper.pdf_url or "",
                    "abstract_preview": (paper.abstract or "")[:200],
                })

        logger.info(f"Exported {len(all_papers)} papers to CSV: {output_path}")
        return output_path