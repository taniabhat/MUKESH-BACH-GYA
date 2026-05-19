"""
tests/test_parsers.py — Stage 2 Parser Tests
=============================================
Tests:
  1.  GrobidClient.is_alive()                    — live network check
  2.  GrobidClient.process_pdf() with a real PDF — full parse pipeline
  3.  GrobidResult shape validation              — required fields present
  4.  needs_ocr() on a text PDF                  — should return False
  5.  needs_ocr() on a synthetic blank PDF       — should return True
  6.  OcrParser.__init__ / lazy model loading    — import-level smoke test
  7.  OcrParser._render_page()                   — rasterisation check
  8.  OcrParser._detect_layout()                 — layout model smoke test
  9.  OcrParser.process_pdf() on blank PDF       — full OCR fallback run
  10. End-to-end routing: digital PDF → GROBID   — orchestration logic

Run:
    python tests/test_parsers.py
    python tests/test_parsers.py --pdf path/to/your.pdf   (test with real PDF)

Exit 0 = all pass.
"""

from __future__ import annotations

import argparse
import io
import sys
import traceback
from pathlib import Path

# ── Formatting helpers ─────────────────────────────────────────────────────────
PASS  = "\033[92m  [PASS]\033[0m"
FAIL  = "\033[91m  [FAIL]\033[0m"
WARN  = "\033[93m  [WARN]\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> bool:
    try:
        detail = fn() or ""
        results.append((name, True, detail))
        print(f"{PASS} {name}" + (f"  — {detail}" if detail else ""))
        return True
    except Exception as exc:
        tb = traceback.format_exc()
        results.append((name, False, str(exc)))
        print(f"{FAIL} {name}")
        print(f"       ↳ {exc}")
        lines = [l for l in tb.splitlines() if "File" in l or "Error" in l]
        for l in lines[-3:]:
            print(f"         {l.strip()}")
        return False


# ── Synthetic PDF helpers ──────────────────────────────────────────────────────

def _make_text_pdf(tmp_path: Path) -> Path:
    """Create a minimal text-layer PDF using PyMuPDF."""
    import fitz
    doc  = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World — this is a digital text PDF.", fontsize=12)
    out = tmp_path / "text_sample.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_blank_pdf(tmp_path: Path) -> Path:
    """Create a PDF with no text layer (simulates scanned)."""
    import fitz
    doc = fitz.open()
    doc.new_page()          # blank page, no text
    out = tmp_path / "blank_sample.pdf"
    doc.save(str(out))
    doc.close()
    return out


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_grobid_alive():
    from src.parsers.grobid_client import GrobidClient
    client = GrobidClient()
    assert client.is_alive(), "GROBID is not reachable — is docker compose up?"
    return "GROBID /api/isalive → true"


def test_grobid_process_real_pdf(pdf_path: Path):
    from src.parsers.grobid_client import GrobidClient, GrobidResult
    client = GrobidClient()
    result = client.process_pdf(pdf_path)
    assert isinstance(result, GrobidResult)
    assert result.success, f"GROBID returned error: {result.error}"
    assert isinstance(result.sections, list)
    assert isinstance(result.citations, list)
    return (
        f"title={result.title[:50]!r}, "
        f"sections={len(result.sections)}, "
        f"citations={len(result.citations)}"
    )


def test_grobid_result_shape(pdf_path: Path):
    from src.parsers.grobid_client import GrobidClient
    client = GrobidClient()
    result = client.process_pdf(pdf_path)
    # Validate all fields exist with correct types
    assert hasattr(result, "success")   and isinstance(result.success,   bool)
    assert hasattr(result, "title")     and isinstance(result.title,     str)
    assert hasattr(result, "abstract")  and isinstance(result.abstract,  str)
    assert hasattr(result, "sections")  and isinstance(result.sections,  list)
    assert hasattr(result, "citations") and isinstance(result.citations, list)
    assert hasattr(result, "raw_tei")

    if result.sections:
        s = result.sections[0]
        assert hasattr(s, "heading") and isinstance(s.heading, str)
        assert hasattr(s, "body")    and isinstance(s.body,    str)

    if result.citations:
        c = result.citations[0]
        assert hasattr(c, "ref_id")  and isinstance(c.ref_id,  str)
        assert hasattr(c, "authors") and isinstance(c.authors, list)

    return "GrobidResult shape validated"


def test_needs_ocr_text_pdf(tmp_path: Path):
    from src.parsers.ocr_fallback import needs_ocr
    pdf = _make_text_pdf(tmp_path)
    result = needs_ocr(pdf)
    # A text-layer PDF should NOT need OCR
    assert result is False, f"Expected False for text PDF, got {result}"
    return "text PDF correctly identified as digital (no OCR needed)"


def test_needs_ocr_blank_pdf(tmp_path: Path):
    from src.parsers.ocr_fallback import needs_ocr
    pdf = _make_blank_pdf(tmp_path)
    result = needs_ocr(pdf)
    assert result is True, f"Expected True for blank/scanned PDF, got {result}"
    return "blank PDF correctly identified as scanned (OCR needed)"


def test_ocr_import():
    from src.parsers.ocr_fallback import OcrParser, needs_ocr  # noqa
    return "OcrParser and needs_ocr imported successfully"


def test_ocr_render_page(tmp_path: Path):
    import numpy as np
    from src.parsers.ocr_fallback import OcrParser
    import fitz

    pdf = _make_text_pdf(tmp_path)
    doc  = fitz.open(str(pdf))
    page = doc[0]
    img  = OcrParser._render_page(page)
    doc.close()

    assert isinstance(img, np.ndarray)
    assert img.ndim == 3          # H × W × C
    assert img.shape[2] == 3     # RGB
    h, w = img.shape[:2]
    assert h > 50 and w > 50
    return f"Rendered page: {w}×{h} px"


def test_ocr_detect_layout(tmp_path: Path):
    from src.parsers.ocr_fallback import OcrParser
    import fitz

    pdf  = _make_text_pdf(tmp_path)
    doc  = fitz.open(str(pdf))
    page = doc[0]
    img  = OcrParser._render_page(page)
    doc.close()

    regions = OcrParser._detect_layout(img)
    assert isinstance(regions, list)
    return f"Layout detection returned {len(regions)} region(s)"


def test_ocr_process_blank_pdf(tmp_path: Path):
    from src.parsers.ocr_fallback import OcrParser, OcrResult
    pdf    = _make_blank_pdf(tmp_path)
    parser = OcrParser()
    result = parser.process_pdf(pdf)
    assert isinstance(result, OcrResult)
    assert result.success is True, f"OcrParser failed: {result.error}"
    assert isinstance(result.sections, list)
    return f"OcrResult: sections={len(result.sections)}, title={result.title!r}"


def test_e2e_routing(pdf_path: Path):
    """
    End-to-end routing test:
      digital PDF  →  GROBID path
      scanned PDF  →  OCR fallback path
    """
    import tempfile, pathlib
    from src.parsers.grobid_client import GrobidClient
    from src.parsers.ocr_fallback  import OcrParser, needs_ocr

    client = GrobidClient()
    ocr    = OcrParser()

    # Route the real PDF
    if needs_ocr(pdf_path):
        result = ocr.process_pdf(pdf_path)
        path_taken = "OCR"
    else:
        result = client.process_pdf(pdf_path)
        path_taken = "GROBID"

    assert result.success, f"Routing via {path_taken} failed: {result.error}"

    # Route a synthetic blank PDF
    with tempfile.TemporaryDirectory() as td:
        blank = _make_blank_pdf(pathlib.Path(td))
        assert needs_ocr(blank) is True
        ocr_result = ocr.process_pdf(blank)
        assert ocr_result.success

    return f"Real PDF → {path_taken}; blank PDF → OCR"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2 parser tests")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to a real academic PDF for GROBID tests",
    )
    args = parser.parse_args()

    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())

    # If no PDF supplied, create a minimal synthetic one for basic tests
    test_pdf = args.pdf if args.pdf else _make_text_pdf(tmp)

    print(f"\n{BOLD}╔══════════════════════════════════════════════╗")
    print(      "║   Document Intelligence Engine — Stage 2    ║")
    print(      "║            Parser & OCR Tests               ║")
    print(      f"╚══════════════════════════════════════════════╝{RESET}")
    print(f"  Test PDF: {test_pdf}\n")

    print(f"{BOLD}── GROBID client ──────────────────────────────{RESET}")
    check("GROBID is_alive()",           test_grobid_alive)
    check("process_pdf() real PDF",      lambda: test_grobid_process_real_pdf(test_pdf))
    check("GrobidResult shape",          lambda: test_grobid_result_shape(test_pdf))

    print(f"\n{BOLD}── OCR fallback ───────────────────────────────{RESET}")
    check("needs_ocr() import",          test_ocr_import)
    check("needs_ocr() → text PDF",      lambda: test_needs_ocr_text_pdf(tmp))
    check("needs_ocr() → blank PDF",     lambda: test_needs_ocr_blank_pdf(tmp))
    check("_render_page()",              lambda: test_ocr_render_page(tmp))
    check("_detect_layout()",            lambda: test_ocr_detect_layout(tmp))
    check("process_pdf() blank PDF",     lambda: test_ocr_process_blank_pdf(tmp))

    print(f"\n{BOLD}── End-to-end routing ─────────────────────────{RESET}")
    check("E2E routing (real + blank)",  lambda: test_e2e_routing(test_pdf))

    # ── Summary ────────────────────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{BOLD}── Results ────────────────────────────────────{RESET}")
    print(f"  {passed}/{total} checks passed", end="")
    if failed:
        print(f"  ·  {failed} FAILED\n")
        print(f"{BOLD}  Failing checks:{RESET}")
        for name, ok, detail in results:
            if not ok:
                print(f"    • {name}: {detail}")
        print()
        sys.exit(1)
    else:
        print(f"\n\n  {BOLD}✓ All checks passed — Stage 2 is solid.{RESET}")
        print("  Ready to proceed to Stage 3: Asset Extraction.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
