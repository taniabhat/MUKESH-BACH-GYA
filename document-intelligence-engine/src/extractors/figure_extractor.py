"""
src/extractors/figure_extractor.py
====================================
Extracts figures (images) from a PDF using PyMuPDF.

Strategy (in priority order):
  1. Raster images embedded in the PDF (XOBJECT images) — extracted as PNG
     via fitz.Pixmap, saved to disk, paired with nearby caption text.
  2. Vector figures (no embedded raster) — whole pages that contain
     predominantly vector graphics are rasterised and saved as fallback.

Caption detection heuristic:
  - Text blocks immediately below an image bounding box
  - Starting with "Fig", "Figure", "FIGURE", or "fig."
  - Within MAX_CAPTION_DISTANCE_PT points vertically

Output per figure:
  {
    "caption":    "Figure 1: Architecture overview",
    "image_path": "data/figures/paper_id_fig_p1_0.png",
    "embedding":  null          ← filled in Stage 4 by CLIP/embedding model
    "page":       1
  }

Usage:
    from src.extractors.figure_extractor import FigureExtractor
    extractor = FigureExtractor(output_dir=Path("data/figures"))
    figures = extractor.extract(pdf_path, paper_id="abc123")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger

# ── Tunables ───────────────────────────────────────────────────────────────────
MIN_IMAGE_WIDTH_PT      = 50    # ignore tiny icons / bullets
MIN_IMAGE_HEIGHT_PT     = 50
MAX_CAPTION_DISTANCE_PT = 60    # vertical gap between image bottom and caption
CAPTION_PATTERN         = re.compile(
    r"^(fig(ure|\.)?|table)\s*\d+", re.IGNORECASE
)
VECTOR_PAGE_MIN_DRAWINGS = 20   # pages with ≥ this many paths → rasterise


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class FigureRecord:
    caption:    str
    image_path: str          # relative path stored; caller resolves to absolute
    embedding:  None = None  # populated in Stage 4
    page:       int  = 0


# ── Extractor ──────────────────────────────────────────────────────────────────
class FigureExtractor:
    """
    Extract embedded raster images and vector-figure pages from a PDF.

    Parameters
    ----------
    output_dir : Path
        Directory where extracted PNG files are saved.
        Created automatically if it does not exist.
    """

    def __init__(self, output_dir: Path = Path("data/figures")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, pdf_path: Path, paper_id: str) -> list[FigureRecord]:
        """
        Extract all figures from *pdf_path*.
        Returns a list of FigureRecord; never raises.
        """
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error(f"FigureExtractor: cannot open {pdf_path}: {exc}")
            return []

        records: list[FigureRecord] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # --- embedded raster images ---
            records.extend(
                self._extract_raster_images(page, page_num, paper_id)
            )
            # --- vector-heavy pages (diagrams drawn with PDF paths) ---
            if not records or self._is_vector_figure_page(page):
                vfig = self._extract_vector_page(page, page_num, paper_id)
                if vfig:
                    records.append(vfig)

        doc.close()
        logger.info(
            f"FigureExtractor: {len(records)} figure(s) extracted "
            f"from {pdf_path.name}"
        )
        return records

    # ── Raster image extraction ────────────────────────────────────────────────

    def _extract_raster_images(
        self, page: fitz.Page, page_num: int, paper_id: str
    ) -> list[FigureRecord]:
        records: list[FigureRecord] = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
            except Exception as exc:
                logger.debug(f"Image xref {xref} extract failed: {exc}")
                continue

            width  = base_image.get("width",  0)
            height = base_image.get("height", 0)
            if width < MIN_IMAGE_WIDTH_PT or height < MIN_IMAGE_HEIGHT_PT:
                continue  # skip tiny decorative images

            # Build save path
            fname = f"{paper_id}_fig_p{page_num + 1}_{img_idx}.png"
            save_path = self.output_dir / fname

            # Save as PNG (convert from any source format via Pixmap)
            try:
                pix = fitz.Pixmap(page.parent, xref)
                if pix.n > 4:           # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(str(save_path))
            except Exception as exc:
                logger.warning(f"Could not save image xref {xref}: {exc}")
                continue

            # Find bounding rect on page for caption search
            img_rect = self._get_image_rect(page, xref)
            caption  = self._find_caption(page, img_rect) if img_rect else ""

            records.append(FigureRecord(
                caption=caption,
                image_path=str(save_path),
                page=page_num + 1,
            ))
            logger.debug(f"  Saved figure: {fname}  caption={caption[:60]!r}")

        return records

    def _get_image_rect(
        self, page: fitz.Page, xref: int
    ) -> Optional[fitz.Rect]:
        """Return the bounding rect of the first occurrence of xref on page."""
        for block in page.get_text("rawdict")["blocks"]:
            if block.get("type") != 1:   # type 1 = image block
                continue
            if block.get("number") == xref or True:
                try:
                    return fitz.Rect(block["bbox"])
                except Exception:
                    pass
        return None

    # ── Caption detection ──────────────────────────────────────────────────────

    def _find_caption(
        self, page: fitz.Page, img_rect: fitz.Rect
    ) -> str:
        """
        Search text blocks below img_rect for a caption line.
        Returns the caption string, or "" if none found.
        """
        best_caption = ""
        best_dist    = MAX_CAPTION_DISTANCE_PT + 1

        for block in page.get_text("blocks"):
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) < 5:
                continue
            bx0, by0, bx1, by1, btext = block[:5]
            text = btext.strip().replace("\n", " ")

            if not text:
                continue

            # Must be below the image
            vertical_gap = by0 - img_rect.y1
            if vertical_gap < -5 or vertical_gap > MAX_CAPTION_DISTANCE_PT:
                continue

            # Must overlap horizontally with the image
            h_overlap = min(bx1, img_rect.x1) - max(bx0, img_rect.x0)
            if h_overlap < 0:
                continue

            if CAPTION_PATTERN.match(text) and vertical_gap < best_dist:
                best_caption = text
                best_dist    = vertical_gap

        return best_caption

    # ── Vector figure pages ────────────────────────────────────────────────────

    @staticmethod
    def _is_vector_figure_page(page: fitz.Page) -> bool:
        """Return True if the page looks like a vector diagram."""
        drawings = page.get_drawings()
        return len(drawings) >= VECTOR_PAGE_MIN_DRAWINGS

    def _extract_vector_page(
        self, page: fitz.Page, page_num: int, paper_id: str
    ) -> Optional[FigureRecord]:
        """Rasterise a vector-heavy page and save as PNG."""
        try:
            mat = fitz.Matrix(2, 2)   # 2× zoom → ~144 DPI
            pix = page.get_pixmap(matrix=mat, alpha=False)
            fname     = f"{paper_id}_vecfig_p{page_num + 1}.png"
            save_path = self.output_dir / fname
            pix.save(str(save_path))

            # Try to find any caption-like text on the page
            caption = ""
            for block in page.get_text("blocks"):
                if len(block) >= 5:
                    text = block[4].strip().replace("\n", " ")
                    if CAPTION_PATTERN.match(text):
                        caption = text
                        break

            logger.debug(f"  Saved vector page figure: {fname}")
            return FigureRecord(
                caption=caption,
                image_path=str(save_path),
                page=page_num + 1,
            )
        except Exception as exc:
            logger.warning(f"Vector page rasterisation failed p{page_num}: {exc}")
            return None
