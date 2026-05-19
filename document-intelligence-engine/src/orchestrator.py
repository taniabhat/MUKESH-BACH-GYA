"""
src/orchestrator.py
====================
Document Intelligence Engine — top-level orchestrator.

This module ties all four stages together into a single callable:

    result = Orchestrator().process(pdf_path)

Pipeline (per PDF):
  1. Detect whether the PDF needs OCR (needs_ocr heuristic).
  2. Parse with GrobidClient (digital) or OcrParser (scanned).
  3. Extract figures   → FigureExtractor
  4. Extract tables    → TableExtractor
  5. Extract equations → EquationExtractor
  6. Chunk sections    → SemanticChunker
  7. Compile final JSON matching the target schema.
  8. Write JSON to data/output/<paper_id>.json
  9. Return the PaperDocument Pydantic model.

Target output schema (exact):
  {
    "paper_id":  "uuid",
    "title":     "...",
    "abstract":  "...",
    "sections":  [{"heading": "...", "body": "...", "chunk_id": "uuid"}],
    "figures":   [{"caption": "...", "image_path": "...", "embedding": null}],
    "tables":    [{"caption": "...", "data": [[...]], "chunk_id": "uuid"}],
    "equations": ["E = mc^2", ...],
    "citations": [{"ref_id": "1", "title": "...", "authors": [],
                   "year": 2023, "doi": "..."}]
  }

Usage:
    from pathlib import Path
    from src.orchestrator import Orchestrator

    orch   = Orchestrator()
    result = orch.process(Path("data/input/paper.pdf"))
    print(result.title)
    print(result.model_dump_json(indent=2))
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import orjson
from loguru import logger
from pydantic import BaseModel, Field

from src.chunkers.semantic_chunker  import SemanticChunker
from src.extractors.equation_extractor import EquationExtractor
from src.extractors.figure_extractor   import FigureExtractor
from src.extractors.table_extractor    import TableExtractor
from src.parsers.grobid_client  import GrobidClient
from src.parsers.ocr_fallback   import OcrParser, needs_ocr


# ── Pydantic output models (enforce schema) ────────────────────────────────────

class SectionOut(BaseModel):
    heading:  str
    body:     str
    chunk_id: str


class FigureOut(BaseModel):
    caption:    str
    image_path: str
    embedding:  Optional[Any] = None


class TableOut(BaseModel):
    caption:  str
    data:     list[list[str]]
    chunk_id: str


class CitationOut(BaseModel):
    ref_id:  str
    title:   str
    authors: list[str]
    year:    Optional[int]
    doi:     Optional[str]


class PaperDocument(BaseModel):
    paper_id:  str
    title:     str
    abstract:  str
    sections:  list[SectionOut]  = Field(default_factory=list)
    figures:   list[FigureOut]   = Field(default_factory=list)
    tables:    list[TableOut]    = Field(default_factory=list)
    equations: list[str]         = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)


# ── Orchestrator ───────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Full pipeline orchestrator.

    Parameters
    ----------
    figures_dir : Path
        Where extracted figure PNGs are saved.
    output_dir : Path
        Where final JSON files are written.
    grobid_url : str
        GROBID server base URL.
    """

    def __init__(
        self,
        figures_dir: Path = Path("data/figures"),
        output_dir:  Path = Path("data/output"),
        grobid_url:  str  = "http://localhost:8070",
    ) -> None:
        self.figures_dir = Path(figures_dir)
        self.output_dir  = Path(output_dir)
        self.grobid_url  = grobid_url

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate sub-components
        self._grobid   = GrobidClient(base_url=grobid_url)
        self._ocr      = OcrParser()
        self._figures  = FigureExtractor(output_dir=self.figures_dir)
        self._tables   = TableExtractor()
        self._equations = EquationExtractor()
        self._chunker  = SemanticChunker()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, pdf_path: Path) -> PaperDocument:
        """
        Run the full pipeline on a single PDF.

        Parameters
        ----------
        pdf_path : Path
            Path to the input PDF file.

        Returns
        -------
        PaperDocument
            Validated Pydantic model matching the target schema.
            Also written to data/output/<paper_id>.json.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_id = str(uuid.uuid4())
        logger.info(f"{'='*60}")
        logger.info(f"Processing: {pdf_path.name}  [paper_id={paper_id}]")

        # ── Stage 2: Parse ────────────────────────────────────────────────────
        parse_result = self._parse(pdf_path)
        logger.info(
            f"Parse complete: title={parse_result.title[:60]!r}, "
            f"sections={len(parse_result.sections)}, "
            f"citations={len(parse_result.citations)}"
        )

        # ── Stage 3: Extract assets ───────────────────────────────────────────
        logger.info("Extracting figures …")
        figure_records = self._figures.extract(pdf_path, paper_id=paper_id)

        logger.info("Extracting tables …")
        table_records = self._tables.extract(pdf_path, paper_id=paper_id)

        logger.info("Extracting equations …")
        # Build full text from parsed sections for Layer 1/2 equation detection
        full_text = parse_result.abstract + "\n\n" + "\n\n".join(
            s.body for s in parse_result.sections
        )
        equation_list = self._equations.extract(
            pdf_path=pdf_path,
            grobid_text=full_text,
        )

        # ── Stage 4: Chunk sections ───────────────────────────────────────────
        logger.info("Chunking sections …")
        chunks = self._chunker.chunk(parse_result.sections, paper_id=paper_id)

        # ── Compile final document ────────────────────────────────────────────
        doc = self._compile(
            paper_id      = paper_id,
            parse_result  = parse_result,
            chunks        = chunks,
            figure_records = figure_records,
            table_records  = table_records,
            equation_list  = equation_list,
        )

        # ── Write JSON output ─────────────────────────────────────────────────
        out_path = self.output_dir / f"{paper_id}.json"
        self._write_json(doc, out_path)
        logger.info(f"Output written → {out_path}")
        logger.info(f"{'='*60}")

        return doc

    def process_batch(self, pdf_dir: Path) -> list[PaperDocument]:
        """
        Process all PDFs in a directory.
        Errors on individual files are logged and skipped.
        """
        pdf_dir = Path(pdf_dir)
        pdfs    = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Batch: {len(pdfs)} PDF(s) in {pdf_dir}")

        results: list[PaperDocument] = []
        for pdf in pdfs:
            try:
                results.append(self.process(pdf))
            except Exception as exc:
                logger.error(f"Failed to process {pdf.name}: {exc}")

        logger.info(f"Batch complete: {len(results)}/{len(pdfs)} succeeded")
        return results

    # ── Private: parsing router ────────────────────────────────────────────────

    def _parse(self, pdf_path: Path):
        """Route to GROBID or OCR fallback based on needs_ocr()."""
        if needs_ocr(pdf_path):
            logger.info("Scanned PDF detected — using OCR fallback")
            result = self._ocr.process_pdf(pdf_path)
        else:
            logger.info("Digital PDF detected — using GROBID")
            result = self._grobid.process_pdf(pdf_path)

        if not result.success:
            logger.warning(f"Parser failed ({result.error}) — returning empty result")

        return result

    # ── Private: document compiler ─────────────────────────────────────────────

    def _compile(
        self,
        paper_id:       str,
        parse_result,
        chunks:         list,
        figure_records: list,
        table_records:  list,
        equation_list:  list[str],
    ) -> PaperDocument:
        """Assemble all extracted data into the PaperDocument schema."""

        # Sections — one SectionOut per chunk (chunk_id already assigned)
        sections_out: list[SectionOut] = [
            SectionOut(
                heading  = chunk.heading,
                body     = chunk.body,
                chunk_id = chunk.chunk_id,
            )
            for chunk in chunks
        ]

        # Figures
        figures_out: list[FigureOut] = [
            FigureOut(
                caption    = fig.caption,
                image_path = fig.image_path,
                embedding  = None,
            )
            for fig in figure_records
        ]

        # Tables — assign a chunk_id to each table
        tables_out: list[TableOut] = []
        for idx, tbl in enumerate(table_records):
            tbl_chunk_id = SemanticChunker._make_uuid(
                f"{paper_id}::table", idx
            )
            tables_out.append(TableOut(
                caption  = tbl.caption,
                data     = tbl.data,
                chunk_id = tbl_chunk_id,
            ))

        # Citations
        citations_out: list[CitationOut] = [
            CitationOut(
                ref_id  = cit.ref_id,
                title   = cit.title,
                authors = cit.authors,
                year    = cit.year,
                doi     = cit.doi,
            )
            for cit in parse_result.citations
        ]

        return PaperDocument(
            paper_id  = paper_id,
            title     = parse_result.title,
            abstract  = parse_result.abstract,
            sections  = sections_out,
            figures   = figures_out,
            tables    = tables_out,
            equations = equation_list,
            citations = citations_out,
        )

    # ── Private: JSON writer ───────────────────────────────────────────────────

    @staticmethod
    def _write_json(doc: PaperDocument, path: Path) -> None:
        """Serialise PaperDocument to pretty-printed JSON via orjson."""
        json_bytes = orjson.dumps(
            doc.model_dump(),
            option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
        )
        path.write_bytes(json_bytes)
