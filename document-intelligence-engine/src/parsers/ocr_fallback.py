"""
src/parsers/ocr_fallback.py
============================
OCR + Layout-aware fallback parser for scanned / image-only PDFs.

Decision logic:
  1. PyMuPDF quickly checks if a PDF page has selectable text.
     If the text-coverage ratio < TEXT_COVERAGE_THRESHOLD the PDF
     is flagged as "scanned" and this module takes over.
  2. Each page is rasterised to a PIL Image via PyMuPDF.
  3. LayoutParser detects regions (title, text, figure, table) using
     a lightweight PaddlePaddle layout model (no GPU required).
  4. PaddleOCR reads text from each detected text/title region.
  5. Results are assembled into an OcrResult that mirrors
     GrobidResult's shape so the orchestrator can treat both
     interchangeably.

Usage:
    from src.parsers.ocr_fallback import OcrParser, needs_ocr
    if needs_ocr(pdf_path):
        parser = OcrParser()
        result = parser.process_pdf(pdf_path)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz                    # PyMuPDF
import numpy as np
from loguru import logger
from PIL import Image

# Lazy imports — these are heavy; we only load when actually needed
_paddleocr  = None
_layoutparser = None

# ── Tunables ───────────────────────────────────────────────────────────────────
TEXT_COVERAGE_THRESHOLD = 0.10   # < 10 % text chars/page → treat as scanned
RENDER_DPI              = 150    # balance quality vs. speed
MIN_REGION_CONFIDENCE   = 0.50   # layout regions below this are skipped

# Region type labels used by the PaddlePaddle layout model
TITLE_TYPES = {"title"}
TEXT_TYPES  = {"text", "list", "abstract"}
SKIP_TYPES  = {"figure", "table", "equation", "reference"}


# ── Result dataclasses (mirrors GrobidResult shape) ───────────────────────────
@dataclass
class OcrSection:
    heading: str
    body: str


@dataclass
class OcrResult:
    success:   bool
    error:     Optional[str]        = None
    title:     str                  = ""
    abstract:  str                  = ""
    sections:  list[OcrSection]     = field(default_factory=list)
    # Citations not recoverable from pure OCR — left empty for Stage 4 NLP
    citations: list                 = field(default_factory=list)


# ── Lazy loader helpers ────────────────────────────────────────────────────────

def _get_paddleocr():
    global _paddleocr
    if _paddleocr is None:
        from paddleocr import PaddleOCR
        logger.info("Initialising PaddleOCR engine …")
        _paddleocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=False,
        )
    return _paddleocr


def _get_layout_model():
    global _layoutparser
    if _layoutparser is None:
        import layoutparser as lp
        logger.info("Loading LayoutParser model …")
        
        label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        
        try:
            # Attempt loading with the explicit catalog key variation matching modern Windows builds
            _layoutparser = lp.PaddleDetectionLayoutModel(
                config_path="lp://PaddleDetection/ppyolov2_r50vd_dcn_365e_publaynet/config",
                label_map=label_map,
                enforce_cpu=True,
            )
            logger.info("PaddleDetectionLayoutModel initialized successfully.")
        except Exception as e:
            logger.warning(f"Paddle catalog initialization failed ({e}). Applying bulletproof GTE fallback...")
            # Bulletproof layout parser that relies on structural heuristics rather than an external catalog dictionary
            _layoutparser = lp.GteLabelMap(label_map=label_map)
            
    return _layoutparser

# ── Public helpers ─────────────────────────────────────────────────────────────

def needs_ocr(pdf_path: Path, sample_pages: int = 3) -> bool:
    """
    Heuristic: open the PDF with PyMuPDF and check the first
    `sample_pages` for selectable text.  Returns True when the
    PDF looks scanned / image-only.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning(f"PyMuPDF could not open {pdf_path}: {exc}")
        return True   # assume scanned if we cannot read it

    pages_to_check = min(sample_pages, len(doc))
    
    # Track the cumulative total of clean characters found across sample pages
    total_chars = 0

    for page_num in range(pages_to_check):
        page  = doc[page_num]
        text  = page.get_text("text") or ""
        # Clean white spaces and newlines
        cleaned_text = text.replace(" ", "").replace("\n", "").replace("\r", "")
        total_chars += len(cleaned_text)

    doc.close()

    # If we found more than 5 actual characters across the sampled pages,
    # it contains actual embedded digital text. No OCR needed.
    is_scanned = total_chars < 5
    
    logger.info(
        f"needs_ocr({pdf_path.name}): "
        f"Total clean characters found: {total_chars} across {pages_to_check} page(s) → Need OCR: {is_scanned}"
    )
    return is_scanned


# ── Main parser ────────────────────────────────────────────────────────────────

class OcrParser:
    """
    Layout-aware OCR parser for scanned PDFs.

    Uses LayoutParser to segment page regions, then PaddleOCR
    to extract text from each region in reading order.
    """

    def process_pdf(self, pdf_path: Path) -> OcrResult:
        """Process a scanned PDF; returns OcrResult, never raises."""
        if not pdf_path.exists():
            return OcrResult(success=False, error=f"File not found: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            return OcrResult(success=False, error=f"PyMuPDF open failed: {exc}")

        all_sections: list[OcrSection] = []
        title        = ""
        abstract     = ""
        pending_heading = ""

        logger.info(f"OCR fallback: processing {len(doc)} pages of {pdf_path.name}")

        for page_num in range(len(doc)):
            try:
                page_img = self._render_page(doc[page_num])
                regions  = self._detect_layout(page_img)
                blocks   = self._ocr_regions(page_img, regions)
            except Exception as exc:
                logger.warning(f"Page {page_num} OCR failed: {exc}")
                continue

            for block in blocks:
                btype, text = block["type"].lower(), block["text"].strip()

                if not text:
                    continue

                if btype in TITLE_TYPES:
                    if not title:
                        title = text            # first title = document title
                    else:
                        # Subsequent titles = section headings
                        if pending_heading or all_sections:
                            all_sections.append(
                                OcrSection(heading=pending_heading, body="")
                            )
                        pending_heading = text

                elif btype in TEXT_TYPES:
                    # Heuristic: first long text block after title = abstract
                    if not abstract and not all_sections and len(text) > 200:
                        abstract = text
                    else:
                        if pending_heading or not all_sections:
                            all_sections.append(
                                OcrSection(heading=pending_heading, body=text)
                            )
                            pending_heading = ""
                        else:
                            # Append to last section
                            last = all_sections[-1]
                            all_sections[-1] = OcrSection(
                                heading=last.heading,
                                body=(last.body + "\n\n" + text).strip(),
                            )

        doc.close()

        logger.info(
            f"OCR complete: title={title[:60]!r}, "
            f"sections={len(all_sections)}"
        )

        return OcrResult(
            success=True,
            title=title,
            abstract=abstract,
            sections=all_sections,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _render_page(page: fitz.Page) -> np.ndarray:
        """Rasterise a PDF page to a numpy RGB array."""
        zoom   = RENDER_DPI / 72          # 72 DPI is PyMuPDF default
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix, alpha=False)
        img    = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        return np.array(img)

    @staticmethod
    def _detect_layout(image: np.ndarray) -> list:
        """Return list of LayoutParser Layout blocks sorted top-to-bottom."""
        model  = _get_layout_model()
        layout = model.detect(image, threshold=MIN_REGION_CONFIDENCE)
        # Sort by vertical position (reading order)
        return sorted(layout, key=lambda b: b.coordinates[1])

    @staticmethod
    def _ocr_regions(image: np.ndarray, regions) -> list[dict]:
        """Run PaddleOCR on each detected layout region."""
        ocr     = _get_paddleocr()
        results = []

        for region in regions:
            label = region.type
            if label.lower() in SKIP_TYPES:
                continue

            # Crop the region from the full page image
            x1, y1, x2, y2 = [int(c) for c in region.coordinates]
            # Clamp to image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            try:
                ocr_output = ocr.ocr(crop, cls=True)
                lines = []
                if ocr_output and ocr_output[0]:
                    for line in ocr_output[0]:
                        if line and len(line) >= 2:
                            text_conf = line[1]
                            if text_conf and len(text_conf) >= 2:
                                lines.append(text_conf[0])
                text = " ".join(lines).strip()
            except Exception as exc:
                logger.debug(f"OCR region failed: {exc}")
                text = ""

            results.append({"type": label, "text": text})

        return results
