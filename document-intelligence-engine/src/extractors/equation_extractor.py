"""
src/extractors/equation_extractor.py
======================================
Detects and extracts mathematical equations from parsed PDF text.

Three-layer detection strategy (applied in order, results deduplicated):

  Layer 1 — LaTeX delimiters
    Matches inline ($...$, \\(...\\)) and display ($$...$$, \\[...\\],
    \\begin{equation}...\\end{equation}) LaTeX environments.
    Highest precision — if the PDF was compiled from LaTeX source,
    GROBID preserves these markers in the extracted text.

  Layer 2 — Unicode math symbols
    Regex over Unicode mathematical operators, Greek letters, superscripts,
    subscripts, and common math symbols (∑ ∫ ∂ α β γ etc.).
    Catches equations in papers that don't use LaTeX delimiters.

  Layer 3 — Structural heuristics
    Short text blocks (≤ MAX_HEURISTIC_LEN chars) that:
      • contain operator patterns (= with surrounding terms, fractions)
      • are visually isolated (standalone block, centred-ish on the page)
      • match common equation forms: "X = Y", "f(x) = ...", numbered "(1)"
    Applied to PyMuPDF block-level text for spatial awareness.

Deduplication:
    Extracted strings are normalised (whitespace collapsed) before
    deduplication. Near-duplicate detection uses simple substring inclusion.

Output:
    A plain list[str] of equation strings.
    ["E = mc^2", "\\nabla \\cdot E = \\rho / \\epsilon_0", ...]

Usage:
    from src.extractors.equation_extractor import EquationExtractor
    extractor = EquationExtractor()
    # From GROBID text
    equations = extractor.extract_from_text(full_text)
    # From PDF (adds spatial heuristics)
    equations = extractor.extract_from_pdf(pdf_path)
    # Combined (recommended)
    equations = extractor.extract(pdf_path, grobid_text=full_text)
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import fitz
from loguru import logger


# ── Layer 1: LaTeX delimiter patterns ─────────────────────────────────────────

_DISPLAY_LATEX = re.compile(
    r"""
    \$\$(.+?)\$\$                         # $$...$$
  | \\\[(.+?)\\\]                         # \[...\]
  | \\begin\{equation\*?\}(.+?)           # \begin{equation}
    \\end\{equation\*?\}
  | \\begin\{align\*?\}(.+?)              # \begin{align}
    \\end\{align\*?\}
  | \\begin\{eqnarray\*?\}(.+?)          # \begin{eqnarray}
    \\end\{eqnarray\*?\}
    """,
    re.VERBOSE | re.DOTALL,
)

_INLINE_LATEX = re.compile(
    r"""
    (?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)  # $...$  (not $$)
  | \\\((.+?)\\\)                         # \(...\)
    """,
    re.VERBOSE | re.DOTALL,
)

# ── Layer 2: Unicode math symbol patterns ─────────────────────────────────────

# Unicode math operators and Greek letters
_UNICODE_MATH = re.compile(
    r"[α-ωΑ-Ωα-ωϕψχ]"                    # Greek
    r"|[∑∏∫∂∇∆∞±×÷≤≥≠≈∈∉⊂⊃∪∩∧∨¬]"       # operators
    r"|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]"               # superscripts
    r"|[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]"               # subscripts
    r"|[ℝℂℤℕℚ𝔼]"                         # blackboard bold
)

# A text fragment counts as a math block if it has enough math symbols
_UNICODE_MATH_LINE = re.compile(
    r"^[^\n]{3,120}$"   # reasonable length
)
UNICODE_MATH_SYMBOL_DENSITY = 0.10   # ≥10% of chars are math symbols


# ── Layer 3: Structural / heuristic patterns ──────────────────────────────────

# Patterns that strongly suggest an equation line
_EQ_PATTERNS = [
    re.compile(r"[A-Za-z]\s*=\s*[^,;]{2,60}"),              # X = expr
    re.compile(r"\d+\s*[+\-*/^]\s*\d"),                      # arithmetic
    re.compile(r"\\frac\{.+?\}\{.+?\}"),                     # LaTeX fraction
    re.compile(r"[A-Za-z_]\^[\{0-9]"),                       # exponent
    re.compile(r"[A-Za-z_]_[\{0-9]"),                        # subscript
    re.compile(r"\([A-Za-z0-9\s\+\-\*/\^]+\)\s*="),         # (expr) =
    re.compile(r"^\s*\(\d+\)\s*$"),                          # isolated "(1)"
    re.compile(r"[A-Za-z]\([A-Za-z]\)\s*="),                 # f(x) =
    re.compile(r"(?:lim|max|min|arg\s*max|arg\s*min)\s*[_\{]"),  # lim_{}
    re.compile(r"(?:log|ln|exp|sin|cos|tan)\s*[(\[{]"),      # trig/log
]

MAX_HEURISTIC_LEN = 200   # ignore very long blocks (paragraphs, not equations)
MIN_HEURISTIC_LEN = 3


# ── Normalisation / deduplication ─────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Collapse whitespace, strip, NFC normalise."""
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def _deduplicate(equations: list[str]) -> list[str]:
    """
    Remove exact duplicates and strings that are substrings of another entry.
    Returns list in original order of first occurrence.
    """
    seen:   list[str] = []
    normed: list[str] = []

    for eq in equations:
        n = _normalise(eq)
        if not n or n in normed:
            continue
        # Skip if already captured as part of a longer equation
        if any(n in existing for existing in normed):
            continue
        seen.append(eq)
        normed.append(n)

    return seen


# ── Main extractor ─────────────────────────────────────────────────────────────

class EquationExtractor:
    """
    Multi-layer equation detector for academic PDF text.
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        pdf_path: Optional[Path] = None,
        grobid_text: str = "",
    ) -> list[str]:
        """
        Combined extraction: run all three layers and deduplicate.

        Parameters
        ----------
        pdf_path : Path, optional
            If provided, Layer 3 spatial heuristics run on raw PDF blocks.
        grobid_text : str
            Full text from GROBID (or OCR) to run Layers 1 & 2 on.
        """
        equations: list[str] = []

        # Layer 1 & 2 on GROBID/OCR text
        if grobid_text:
            equations.extend(self.extract_from_text(grobid_text))

        # Layer 3 on raw PDF block text (spatial context)
        if pdf_path and pdf_path.exists():
            equations.extend(self._layer3_spatial(pdf_path))

        result = _deduplicate(equations)
        logger.info(
            f"EquationExtractor: {len(result)} equation(s) found "
            f"(from {len(equations)} candidates)"
        )
        return result

    def extract_from_text(self, text: str) -> list[str]:
        """Run Layer 1 (LaTeX) + Layer 2 (Unicode) on a text string."""
        eqs: list[str] = []
        eqs.extend(self._layer1_latex(text))
        eqs.extend(self._layer2_unicode(text))
        return eqs

    def extract_from_pdf(self, pdf_path: Path) -> list[str]:
        """Run all three layers directly on a PDF (no pre-extracted text)."""
        try:
            doc  = fitz.open(str(pdf_path))
            text = "\n".join(page.get_text("text") for page in doc)
            doc.close()
        except Exception as exc:
            logger.warning(f"EquationExtractor: cannot read {pdf_path}: {exc}")
            text = ""

        return self.extract(pdf_path=pdf_path, grobid_text=text)

    # ── Layer 1: LaTeX delimiters ──────────────────────────────────────────────

    @staticmethod
    def _layer1_latex(text: str) -> list[str]:
        found: list[str] = []

        for m in _DISPLAY_LATEX.finditer(text):
            content = next((g for g in m.groups() if g is not None), "")
            if content.strip():
                found.append(content.strip())

        for m in _INLINE_LATEX.finditer(text):
            content = next((g for g in m.groups() if g is not None), "")
            s = content.strip()
            # Filter out single variables / numbers — not meaningful equations
            if s and len(s) > 2 and re.search(r"[=+\-*/^\\{}]", s):
                found.append(s)

        return found

    # ── Layer 2: Unicode math density ─────────────────────────────────────────

    @staticmethod
    def _layer2_unicode(text: str) -> list[str]:
        found: list[str] = []

        for line in text.splitlines():
            line = line.strip()
            if not _UNICODE_MATH_LINE.match(line):
                continue
            math_chars = len(_UNICODE_MATH.findall(line))
            if math_chars == 0:
                continue
            density = math_chars / max(len(line), 1)
            if density >= UNICODE_MATH_SYMBOL_DENSITY:
                found.append(line)

        return found

    # ── Layer 3: Spatial heuristics on PDF blocks ──────────────────────────────

    @staticmethod
    def _layer3_spatial(pdf_path: Path) -> list[str]:
        found: list[str] = []
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.debug(f"Layer3 spatial: open failed: {exc}")
            return found

        for page in doc:
            page_width = page.rect.width
            blocks = page.get_text("blocks")

            for block in blocks:
                if len(block) < 5:
                    continue
                x0, y0, x1, y1, text, *_ = block
                text = text.strip().replace("\n", " ")

                if not (MIN_HEURISTIC_LEN <= len(text) <= MAX_HEURISTIC_LEN):
                    continue

                # Centred-ish block (equation display heuristic)
                block_centre = (x0 + x1) / 2
                page_centre  = page_width / 2
                is_centred   = abs(block_centre - page_centre) < page_width * 0.25

                # Check against equation patterns
                pattern_hit = any(p.search(text) for p in _EQ_PATTERNS)

                if pattern_hit and is_centred:
                    found.append(text)

        doc.close()
        return found
