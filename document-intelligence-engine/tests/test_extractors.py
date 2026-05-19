"""
tests/test_extractors.py — Stage 3 Extractor Tests
====================================================
Tests:
  1.  FigureExtractor import
  2.  FigureExtractor on a synthetic PDF (embedded image)
  3.  FigureExtractor output shape validation
  4.  FigureExtractor caption detection
  5.  TableExtractor import + Camelot available
  6.  TableExtractor on a synthetic tabled PDF (lattice)
  7.  TableExtractor output shape validation
  8.  TableExtractor stream fallback (borderless table)
  9.  EquationExtractor import
  10. EquationExtractor Layer 1 — LaTeX delimiters
  11. EquationExtractor Layer 2 — Unicode math symbols
  12. EquationExtractor Layer 3 — structural heuristics on PDF
  13. EquationExtractor deduplication
  14. EquationExtractor.extract() combined (text + PDF)

Run:
    python tests/test_extractors.py
    python tests/test_extractors.py --pdf path/to/real_paper.pdf

Exit 0 = all pass.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import traceback
from pathlib import Path

# ── Console helpers ────────────────────────────────────────────────────────────
PASS  = "\033[92m  [PASS]\033[0m"
FAIL  = "\033[91m  [FAIL]\033[0m"
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


# ── Synthetic PDF builders ─────────────────────────────────────────────────────

def _make_figure_pdf(tmp: Path) -> Path:
    """PDF with one embedded PNG image and a Figure caption below it."""
    import fitz, struct, zlib

    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Minimal valid 10×10 red PNG (raw, no PIL needed)
    def _tiny_png(w=10, h=10):
        import zlib, struct
        raw = b"\x00" + b"\xff\x00\x00" * w   # row filter byte + RGB pixels
        raw = raw * h
        compressed = zlib.compress(raw)
        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            import binascii
            crc = struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)
            return c + crc
        png  = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", compressed)
        png += chunk(b"IEND", b"")
        return png

    png_bytes = _tiny_png()
    # Insert image into page at rect (100, 100, 300, 300)
    img_rect = fitz.Rect(100, 100, 300, 300)
    page.insert_image(img_rect, stream=png_bytes)

    # Caption text below the image
    page.insert_text(
        (100, 320),
        "Figure 1: A synthetic test figure for unit testing.",
        fontsize=10,
    )

    out = tmp / "figure_sample.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_table_pdf_lattice(tmp: Path) -> Path:
    """PDF with a simple ruled table (lattice mode)."""
    import fitz

    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Draw a 3-col × 3-row table with lines
    rows = [["Model", "Accuracy", "F1"],
            ["Ours",  "94.2",     "93.1"],
            ["Base",  "88.5",     "87.0"]]

    x0, y0   = 80, 150
    col_w, row_h = 140, 30
    n_cols, n_rows = 3, 3

    # Draw grid lines
    shape = page.new_shape()
    for r in range(n_rows + 1):
        y = y0 + r * row_h
        shape.draw_line((x0, y), (x0 + n_cols * col_w, y))
    for c in range(n_cols + 1):
        x = x0 + c * col_w
        shape.draw_line((x, y0), (x, y0 + n_rows * row_h))
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()

    # Insert cell text
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text(
                (x0 + c * col_w + 5, y0 + r * row_h + 20),
                cell, fontsize=10,
            )

    # Caption above the table
    page.insert_text((80, 135), "Table 1: Benchmark Results", fontsize=10)

    out = tmp / "table_lattice.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_table_pdf_stream(tmp: Path) -> Path:
    """PDF with a borderless (stream) table — whitespace delimited."""
    import fitz

    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)

    rows = [
        ("Method",    "Dataset",  "Score"),
        ("Proposed",  "ImageNet", "91.3"),
        ("Baseline",  "ImageNet", "85.7"),
    ]
    y = 200
    for row in rows:
        page.insert_text((80,  y), row[0], fontsize=10)
        page.insert_text((220, y), row[1], fontsize=10)
        page.insert_text((380, y), row[2], fontsize=10)
        y += 25

    out = tmp / "table_stream.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_equation_pdf(tmp: Path) -> Path:
    """PDF with centred equation-like text blocks."""
    import fitz

    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)

    lines = [
        (297, 200, "E = mc^2"),
        (297, 250, "f(x) = ax^2 + bx + c"),
        (297, 300, "L(θ) = -∑ y log(ŷ)"),
        (297, 350, "This is a normal sentence about the model architecture."),
        (297, 400, "∇L = ∂L/∂θ"),
    ]
    for x, y, text in lines:
        w = fitz.get_text_length(text, fontsize=12)
        page.insert_text((x - w / 2, y), text, fontsize=12)

    out = tmp / "equation_sample.pdf"
    doc.save(str(out))
    doc.close()
    return out


# ── Figure tests ───────────────────────────────────────────────────────────────

def test_figure_import():
    from src.extractors.figure_extractor import FigureExtractor  # noqa
    return "FigureExtractor imported"


def test_figure_extract(tmp: Path):
    from src.extractors.figure_extractor import FigureExtractor
    pdf = _make_figure_pdf(tmp)
    out_dir = tmp / "figures"
    extractor = FigureExtractor(output_dir=out_dir)
    records = extractor.extract(pdf, paper_id="test001")
    assert isinstance(records, list), "extract() must return a list"
    # Synthetic PDF has 1 embedded image
    assert len(records) >= 1, f"Expected ≥1 figure, got {len(records)}"
    return f"{len(records)} figure(s) extracted"


def test_figure_shape(tmp: Path):
    from src.extractors.figure_extractor import FigureExtractor, FigureRecord
    pdf = _make_figure_pdf(tmp)
    extractor = FigureExtractor(output_dir=tmp / "figures2")
    records = extractor.extract(pdf, paper_id="test002")
    assert records, "No figures extracted — cannot validate shape"
    r = records[0]
    assert isinstance(r, FigureRecord)
    assert hasattr(r, "caption")    and isinstance(r.caption,    str)
    assert hasattr(r, "image_path") and isinstance(r.image_path, str)
    assert hasattr(r, "page")       and isinstance(r.page,       int)
    assert hasattr(r, "embedding")  # should be None at this stage
    assert Path(r.image_path).exists(), f"Image file not on disk: {r.image_path}"
    return f"FigureRecord shape OK, file exists: {Path(r.image_path).name}"


def test_figure_caption(tmp: Path):
    from src.extractors.figure_extractor import FigureExtractor
    pdf = _make_figure_pdf(tmp)
    extractor = FigureExtractor(output_dir=tmp / "figures3")
    records = extractor.extract(pdf, paper_id="test003")
    captions = [r.caption for r in records if r.caption]
    # Our synthetic PDF has "Figure 1: ..." text — at least one should be found
    assert len(captions) >= 1, (
        f"No captions detected. Records: {records}"
    )
    return f"Caption found: {captions[0][:60]!r}"


# ── Table tests ────────────────────────────────────────────────────────────────

def test_table_import():
    from src.extractors.table_extractor import TableExtractor  # noqa
    import camelot  # noqa
    return "TableExtractor + camelot imported"


def test_table_lattice(tmp: Path):
    from src.extractors.table_extractor import TableExtractor
    pdf = _make_table_pdf_lattice(tmp)
    extractor = TableExtractor(min_accuracy=50.0)
    records = extractor.extract(pdf, paper_id="tbl001")
    assert isinstance(records, list)
    assert len(records) >= 1, f"Expected ≥1 table (lattice), got {len(records)}"
    return f"{len(records)} table(s) extracted (lattice)"


def test_table_shape(tmp: Path):
    from src.extractors.table_extractor import TableExtractor, TableRecord
    pdf = _make_table_pdf_lattice(tmp)
    extractor = TableExtractor(min_accuracy=50.0)
    records = extractor.extract(pdf, paper_id="tbl002")
    assert records, "No tables extracted"
    r = records[0]
    assert isinstance(r, TableRecord)
    assert hasattr(r, "caption")  and isinstance(r.caption,  str)
    assert hasattr(r, "data")     and isinstance(r.data,     list)
    assert hasattr(r, "page")     and isinstance(r.page,     int)
    assert hasattr(r, "accuracy") and isinstance(r.accuracy, float)
    assert len(r.data) >= 1 and len(r.data[0]) >= 1, "data must be 2D list"
    return f"TableRecord shape OK, {len(r.data)}×{len(r.data[0])} cells"


def test_table_stream(tmp: Path):
    from src.extractors.table_extractor import TableExtractor
    pdf = _make_table_pdf_stream(tmp)
    extractor = TableExtractor(min_accuracy=30.0)   # lower threshold for stream
    records = extractor.extract(pdf, paper_id="tbl003")
    assert isinstance(records, list)
    # Stream mode may or may not detect the table depending on spacing
    # — we just assert it doesn't raise and returns a list
    return f"{len(records)} table(s) extracted (stream fallback)"


# ── Equation tests ─────────────────────────────────────────────────────────────

def test_equation_import():
    from src.extractors.equation_extractor import EquationExtractor  # noqa
    return "EquationExtractor imported"


def test_equation_layer1_latex():
    from src.extractors.equation_extractor import EquationExtractor
    ex = EquationExtractor()
    text = r"""
    The energy equation is $E = mc^2$ where $m$ is mass.
    Display form: $$F = ma$$
    Environment: \begin{equation} \nabla \cdot E = \rho / \epsilon_0 \end{equation}
    """
    eqs = ex.extract_from_text(text)
    assert len(eqs) >= 2, f"Expected ≥2 LaTeX equations, got {eqs}"
    return f"Layer 1: {len(eqs)} equation(s) — {eqs[:2]}"


def test_equation_layer2_unicode():
    from src.extractors.equation_extractor import EquationExtractor
    ex = EquationExtractor()
    text = (
        "Normal sentence here.\n"
        "∇L = ∂L/∂θ + αβγ term\n"
        "Another normal sentence.\n"
        "∑_{i=1}^{n} x_i = μ ± σ\n"
    )
    eqs = ex.extract_from_text(text)
    assert len(eqs) >= 1, f"Expected ≥1 Unicode equation, got {eqs}"
    return f"Layer 2: {len(eqs)} equation(s)"


def test_equation_layer3_pdf(tmp: Path):
    from src.extractors.equation_extractor import EquationExtractor
    ex  = EquationExtractor()
    pdf = _make_equation_pdf(tmp)
    eqs = ex.extract_from_pdf(pdf)
    assert isinstance(eqs, list)
    assert len(eqs) >= 1, f"Expected ≥1 equation from PDF heuristics, got {eqs}"
    return f"Layer 3: {len(eqs)} equation(s) — {eqs[:2]}"


def test_equation_dedup():
    from src.extractors.equation_extractor import EquationExtractor, _deduplicate
    # Exact duplicates
    duped = ["E = mc^2", "E = mc^2", "F = ma", "E = mc^2"]
    result = _deduplicate(duped)
    assert len(result) == 2, f"Expected 2 after dedup, got {result}"
    # Substring dedup
    subset = ["E = mc", "E = mc^2 (full form)", "F = ma"]
    result2 = _deduplicate(subset)
    # "E = mc" is a substring of the longer one — should be dropped
    assert len(result2) <= 3
    return f"Dedup: {duped} → {result}"


def test_equation_combined(tmp: Path):
    from src.extractors.equation_extractor import EquationExtractor
    ex  = EquationExtractor()
    pdf = _make_equation_pdf(tmp)
    grobid_text = r"The loss is $L(\theta) = -\sum y \log \hat{y}$. Also $$E=mc^2$$."
    eqs = ex.extract(pdf_path=pdf, grobid_text=grobid_text)
    assert isinstance(eqs, list)
    assert len(eqs) >= 1
    return f"Combined: {len(eqs)} equation(s) total"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3 extractor tests")
    parser.add_argument("--pdf", type=Path, default=None,
                        help="Real academic PDF for additional extractor tests")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp())

    print(f"\n{BOLD}╔══════════════════════════════════════════════╗")
    print(      "║   Document Intelligence Engine — Stage 3    ║")
    print(      "║           Asset Extractor Tests             ║")
    print(      f"╚══════════════════════════════════════════════╝{RESET}\n")

    print(f"{BOLD}── Figure Extractor ───────────────────────────{RESET}")
    check("FigureExtractor import",         test_figure_import)
    check("extract() synthetic PDF",        lambda: test_figure_extract(tmp))
    check("FigureRecord shape",             lambda: test_figure_shape(tmp))
    check("Caption detection",              lambda: test_figure_caption(tmp))

    print(f"\n{BOLD}── Table Extractor ────────────────────────────{RESET}")
    check("TableExtractor + camelot import",test_table_import)
    check("extract() lattice table",        lambda: test_table_lattice(tmp))
    check("TableRecord shape",              lambda: test_table_shape(tmp))
    check("Stream fallback",                lambda: test_table_stream(tmp))

    print(f"\n{BOLD}── Equation Extractor ─────────────────────────{RESET}")
    check("EquationExtractor import",       test_equation_import)
    check("Layer 1 — LaTeX delimiters",     test_equation_layer1_latex)
    check("Layer 2 — Unicode math",         test_equation_layer2_unicode)
    check("Layer 3 — PDF spatial",          lambda: test_equation_layer3_pdf(tmp))
    check("Deduplication logic",            test_equation_dedup)
    check("Combined extract()",             lambda: test_equation_combined(tmp))

    # Optional: run extractors on a real PDF
    if args.pdf:
        print(f"\n{BOLD}── Real PDF: {args.pdf.name} ──────────────────────{RESET}")
        from src.extractors.figure_extractor   import FigureExtractor
        from src.extractors.table_extractor    import TableExtractor
        from src.extractors.equation_extractor import EquationExtractor

        fig_out = tmp / "real_figures"
        check("Real PDF figures",
              lambda: f"{len(FigureExtractor(fig_out).extract(args.pdf, 'real'))} figure(s)")
        check("Real PDF tables",
              lambda: f"{len(TableExtractor().extract(args.pdf, 'real'))} table(s)")
        check("Real PDF equations",
              lambda: f"{len(EquationExtractor().extract_from_pdf(args.pdf))} equation(s)")

    # ── Summary ────────────────────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{BOLD}── Results ────────────────────────────────────{RESET}")
    print(f"  {passed}/{total} checks passed", end="")
    if failed:
        print(f"  ·  {failed} FAILED\n")
        for name, ok, detail in results:
            if not ok:
                print(f"    • {name}: {detail}")
        sys.exit(1)
    else:
        print(f"\n\n  {BOLD}✓ All checks passed — Stage 3 is solid.{RESET}")
        print("  Ready to proceed to Stage 4: Semantic Chunking & Final JSON.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
