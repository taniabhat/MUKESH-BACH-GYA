"""
src/extractors/table_extractor.py
===================================
Extracts tables from a PDF using Camelot.

Strategy (two-pass with automatic fallback):
  Pass 1 — Camelot "lattice" mode:
    Works on tables with visible ruling lines (borders).
    Most reliable; produces clean row/column data.

  Pass 2 — Camelot "stream" mode (fallback):
    Works on borderless / whitespace-delimited tables.
    Used when lattice finds nothing on a given page.

  Accuracy filter:
    Camelot assigns each extracted table an accuracy score (0–100).
    Tables below MIN_ACCURACY are discarded.

Caption detection:
    PyMuPDF scans the text just above each table's bounding box
    for lines matching "Table N" / "TABLE N".

Output per table:
  {
    "caption":  "Table 2: Benchmark Results",
    "data":     [["Model", "Acc"], ["Ours", "94.2"]],
    "chunk_id": null        ← assigned in Stage 4
    "page":     3
  }

Usage:
    from src.extractors.table_extractor import TableExtractor
    extractor = TableExtractor()
    tables = extractor.extract(pdf_path, paper_id="abc123")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz
from loguru import logger

# Camelot is imported lazily — it pulls in ghostscript at import time
# which can fail loudly if ghostscript isn't on PATH.

# ── Tunables ───────────────────────────────────────────────────────────────────
MIN_ACCURACY           = 70.0   # discard tables below this Camelot score
MAX_CAPTION_DIST_PT    = 40     # points above table bbox to look for caption
CAPTION_PATTERN        = re.compile(r"^table\s*\d+", re.IGNORECASE)
MAX_PAGES_PER_BATCH    = 10     # process in batches to limit memory use


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class TableRecord:
    caption:  str
    data:     list[list[str]]
    chunk_id: None = None   # assigned in Stage 4
    page:     int  = 0
    accuracy: float = 0.0


# ── Extractor ──────────────────────────────────────────────────────────────────
class TableExtractor:
    """
    Two-pass Camelot table extractor with PyMuPDF caption matching.

    Parameters
    ----------
    min_accuracy : float
        Minimum Camelot accuracy score to keep a table (0–100).
    """

    def __init__(self, min_accuracy: float = MIN_ACCURACY) -> None:
        self.min_accuracy = min_accuracy

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, pdf_path: Path, paper_id: str = "") -> list[TableRecord]:
        """
        Extract all tables from *pdf_path*.
        Returns a list of TableRecord; never raises.
        """
        try:
            import camelot
        except ImportError:
            logger.error("camelot-py not installed. Run: pip install camelot-py[cv]")
            return []

        if not pdf_path.exists():
            logger.error(f"TableExtractor: file not found: {pdf_path}")
            return []

        # Get total page count via PyMuPDF (cheaper than Camelot for this)
        try:
            doc        = fitz.open(str(pdf_path))
            total_pages = len(doc)
            doc.close()
        except Exception as exc:
            logger.error(f"TableExtractor: cannot open PDF: {exc}")
            return []

        records: list[TableRecord] = []

        # Process in page batches to avoid OOM on large PDFs
        for batch_start in range(1, total_pages + 1, MAX_PAGES_PER_BATCH):
            batch_end  = min(batch_start + MAX_PAGES_PER_BATCH - 1, total_pages)
            page_range = f"{batch_start}-{batch_end}"

            batch_records = self._process_page_range(
                camelot, pdf_path, page_range
            )
            records.extend(batch_records)

        # Attach captions using PyMuPDF
        records = self._attach_captions(pdf_path, records)

        logger.info(
            f"TableExtractor: {len(records)} table(s) extracted "
            f"from {pdf_path.name}"
        )
        return records

    # ── Two-pass extraction ────────────────────────────────────────────────────

    def _process_page_range(
        self, camelot, pdf_path: Path, page_range: str
    ) -> list[TableRecord]:
        """Run lattice → stream fallback on a page range."""
        records: list[TableRecord] = []

        # Pass 1: lattice (ruled tables)
        lattice_tables = self._run_camelot(
            camelot, pdf_path, page_range, flavor="lattice"
        )
        pages_with_lattice: set[int] = set()

        for t in lattice_tables:
            if t.accuracy >= self.min_accuracy:
                records.append(self._camelot_to_record(t))
                pages_with_lattice.add(t.page)

        # Pass 2: stream on pages where lattice found nothing
        stream_tables = self._run_camelot(
            camelot, pdf_path, page_range, flavor="stream"
        )
        for t in stream_tables:
            if t.page not in pages_with_lattice and t.accuracy >= self.min_accuracy:
                records.append(self._camelot_to_record(t))

        return records

    @staticmethod
    def _run_camelot(
        camelot, pdf_path: Path, page_range: str, flavor: str
    ) -> list:
        """Run camelot.read_pdf(); return empty list on any error."""
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=page_range,
                flavor=flavor,
                suppress_stdout=True,
            )
            return list(tables)
        except Exception as exc:
            logger.debug(f"Camelot {flavor} on pages {page_range}: {exc}")
            return []

    # ── Conversion ─────────────────────────────────────────────────────────────

    @staticmethod
    def _camelot_to_record(table) -> TableRecord:
        """Convert a Camelot Table object to a TableRecord."""
        df   = table.df
        data = df.values.tolist()
        # Normalise: all cells to stripped strings
        data = [
            [str(cell).strip() for cell in row]
            for row in data
        ]
        return TableRecord(
            caption="",         # filled by _attach_captions
            data=data,
            page=table.page,
            accuracy=round(table.accuracy, 2),
        )

    # ── Caption matching via PyMuPDF ───────────────────────────────────────────

    def _attach_captions(
        self, pdf_path: Path, records: list[TableRecord]
    ) -> list[TableRecord]:
        """
        For each TableRecord, search the PDF page text for a
        "Table N" line above the table's position.
        """
        if not records:
            return records

        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            return records   # captions optional — don't break on open failure

        for rec in records:
            page_idx = rec.page - 1
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            rec.caption = self._find_caption_on_page(page)

        doc.close()
        return records

    @staticmethod
    def _find_caption_on_page(page: fitz.Page) -> str:
        """
        Scan all text blocks on a page for the first Table caption line.
        Returns the caption string or "".
        """
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            text = block[4].strip().replace("\n", " ")
            if CAPTION_PATTERN.match(text):
                return text
        return ""
