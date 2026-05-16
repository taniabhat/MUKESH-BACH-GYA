"""
Persistent storage service for discovery results.
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from research_discovery.core.utils import (
    get_logger,
)
from research_discovery.models.paper import (
    DiscoveryResult,
    Paper,
)

logger = get_logger(__name__)

DEFAULT_STORAGE_DIR = Path(
    os.getenv(
        "STORAGE_DIR",
        "/tmp/research_discovery",
    )
)

SCHEMA_VERSION = 1

MAX_FILENAME_LENGTH = 80

EMBEDDING_STRIP_FIELDS = {
    "embedding",
}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class ResultSerializer:
    """Handles JSON-safe serialization."""

    @staticmethod
    def serialize(
        result: DiscoveryResult,
        strip_embeddings: bool = True,
    ) -> dict[str, Any]:

        data = result.model_dump(
            mode="json",
        )

        data["_schema_version"] = (
            SCHEMA_VERSION
        )

        if strip_embeddings:
            ResultSerializer._strip_embeddings(
                data
            )

        return data

    @staticmethod
    def deserialize(
        data: dict[str, Any],
    ) -> DiscoveryResult:

        data.pop(
            "_schema_version",
            None,
        )

        return DiscoveryResult(**data)

    @staticmethod
    def _strip_embeddings(
        data: dict[str, Any],
    ) -> None:

        tiers = [
            "highly_relevant",
            "relevant_background",
            "adjacent_work",
            "historical_foundations",
        ]

        for tier in tiers:

            for paper in data.get(
                tier,
                [],
            ):

                for field in (
                    EMBEDDING_STRIP_FIELDS
                ):

                    if field in paper:
                        paper[field] = []


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

class CSVExporter:
    """Exports papers into spreadsheet-friendly CSV."""

    FIELDNAMES = [
        "tier",
        "title",
        "year",
        "venue",
        "authors",
        "citation_count",
        "final_score",
        "semantic_similarity",
        "doi",
        "arxiv_id",
        "pdf_url",
        "abstract_preview",
    ]

    @classmethod
    def export(
        cls,
        result: DiscoveryResult,
        output_path: str,
    ) -> str:

        papers = result.all_papers()

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=cls.FIELDNAMES,
            )

            writer.writeheader()

            for paper in papers:

                writer.writerow(
                    cls._paper_to_row(
                        paper
                    )
                )

        logger.info(
            "CSV export complete papers=%s path=%s",
            len(papers),
            output_path,
        )

        return output_path

    @staticmethod
    def _paper_to_row(
        paper: Paper,
    ) -> dict[str, Any]:

        authors = [
            author.name
            for author in paper.authors[:3]
        ]

        author_text = ", ".join(authors)

        if len(paper.authors) > 3:
            author_text += " et al."

        return {
            "tier": paper.tier or "",
            "title": paper.title,
            "year": paper.year or "",
            "venue": paper.venue or "",
            "authors": author_text,
            "citation_count": (
                paper.citation_count
            ),
            "final_score": round(
                paper.final_score,
                4,
            ),
            "semantic_similarity": round(
                paper.ranking_features.semantic_similarity,
                4,
            ),
            "doi": (
                paper.external_ids.doi
                or ""
            ),
            "arxiv_id": (
                paper.external_ids.arxiv
                or ""
            ),
            "pdf_url": (
                paper.pdf_url
                or ""
            ),
            "abstract_preview": (
                (
                    paper.abstract
                    or ""
                )[:200]
            ),
        }


# ---------------------------------------------------------------------------
# Storage Service
# ---------------------------------------------------------------------------

class JSONStorageService:
    """
    JSON persistence backend.
    """

    def __init__(
        self,
        storage_dir: Path = (
            DEFAULT_STORAGE_DIR
        ),
    ):

        self.storage_dir = storage_dir

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_result(
        self,
        result: DiscoveryResult,
        strip_embeddings: bool = True,
    ) -> str:

        filename = self._build_filename(
            result
        )

        filepath = (
            self.storage_dir
            / filename
        )

        payload = (
            ResultSerializer.serialize(
                result,
                strip_embeddings=(
                    strip_embeddings
                ),
            )
        )

        self._atomic_write_json(
            filepath,
            payload,
        )

        logger.info(
            "Discovery result saved path=%s",
            filepath,
        )

        return str(filepath)

    def load_result(
        self,
        filepath: str,
    ) -> Optional[DiscoveryResult]:

        try:

            with open(
                filepath,
                "r",
                encoding="utf-8",
            ) as handle:

                data = json.load(
                    handle
                )

            return (
                ResultSerializer.deserialize(
                    data
                )
            )

        except Exception:

            logger.exception(
                "Failed to load discovery result path=%s",
                filepath,
            )

            return None

    def list_results(
        self,
    ) -> list[dict[str, Any]]:

        results = []

        for filepath in sorted(
            self.storage_dir.glob(
                "*.json"
            ),
            reverse=True,
        ):

            try:

                stat = filepath.stat()

                results.append({
                    "filename": (
                        filepath.name
                    ),
                    "path": (
                        str(filepath)
                    ),
                    "size_kb": round(
                        stat.st_size
                        / 1024,
                        1,
                    ),
                    "created": (
                        datetime.fromtimestamp(
                            stat.st_ctime,
                            tz=timezone.utc,
                        ).isoformat()
                    ),
                })

            except Exception:

                logger.exception(
                    "Failed to inspect result file path=%s",
                    filepath,
                )

        return results

    def export_csv(
        self,
        result: DiscoveryResult,
        output_path: str,
    ) -> str:

        return CSVExporter.export(
            result=result,
            output_path=output_path,
        )

    @staticmethod
    def _build_filename(
        result: DiscoveryResult,
    ) -> str:

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d_%H%M%S"
        )

        safe_query = (
            JSONStorageService
            ._sanitize_filename(
                result.query.original_idea
            )
        )

        return (
            f"{timestamp}_"
            f"{safe_query}.json"
        )

    @staticmethod
    def _sanitize_filename(
        value: str,
    ) -> str:

        value = value.lower()

        value = re.sub(
            r"[^\w\s-]",
            "",
            value,
        )

        value = re.sub(
            r"\s+",
            "_",
            value,
        )

        value = value.strip("_")

        return value[
            :MAX_FILENAME_LENGTH
        ]

    @staticmethod
    def _atomic_write_json(
        filepath: Path,
        payload: dict[str, Any],
    ) -> None:

        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=filepath.parent,
            encoding="utf-8",
        ) as temp_file:

            json.dump(
                payload,
                temp_file,
                indent=2,
                ensure_ascii=False,
            )

            temp_path = Path(
                temp_file.name
            )

        temp_path.replace(filepath)