"""
tests/test_orchestrator.py — Stage 4 Full Pipeline Tests
=========================================================
Tests:
  1.  SemanticChunker import
  2.  Chunker — single short section (no split needed)
  3.  Chunker — oversized section triggers paragraph split
  4.  Chunker — oversized paragraph triggers sentence split
  5.  Chunker — deterministic UUID (same paper_id → same chunk_ids)
  6.  Chunker — small chunk merging
  7.  Orchestrator import + instantiation
  8.  Orchestrator.process() on a synthetic digital PDF
  9.  PaperDocument schema validation (all required fields present)
  10. Output JSON written to disk with correct structure
  11. Orchestrator.process() on a synthetic scanned (blank) PDF
  12. process_batch() on a directory of synthetic PDFs
  13. End-to-end: real PDF (if --pdf supplied)

Run:
    python tests/test_orchestrator.py
    python tests/test_orchestrator.py --pdf data/input/paper.pdf

Exit 0 = all pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

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


# ── Synthetic PDF helpers ──────────────────────────────────────────────────────

def _make_digital_pdf(tmp: Path) -> Path:
    """Multi-section academic-style PDF with text, image placeholder, table."""
    import fitz

    doc = fitz.open()

    # Page 1: title + abstract
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((72, 80),  "Attention Is All You Need", fontsize=18)
    p1.insert_text((72, 120), "Abstract", fontsize=13)
    p1.insert_text(
        (72, 145),
        ("The dominant sequence transduction models are based on complex "
         "recurrent or convolutional neural networks. We propose a new simple "
         "network architecture, the Transformer, based solely on attention "
         "mechanisms. Experiments show these models are superior in quality."),
        fontsize=10,
    )

    # Page 2: sections
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((72, 80),  "1. Introduction", fontsize=13)
    p2.insert_text(
        (72, 110),
        ("Recurrent neural networks, long short-term memory and gated recurrent "
         "neural networks in particular, have been firmly established as state of "
         "the art approaches in sequence modelling. Numerous efforts have continued "
         "to push the boundaries of recurrent language models and encoder-decoder "
         "architectures. Attention mechanisms have become an integral part of "
         "compelling sequence modelling. In this work, we propose the Transformer."),
        fontsize=10,
    )
    p2.insert_text((72, 260), "2. Methodology", fontsize=13)
    p2.insert_text(
        (72, 285),
        ("The Transformer uses stacked self-attention and point-wise, fully "
         "connected layers for both the encoder and decoder. The encoder maps "
         "an input sequence of symbol representations to a sequence of continuous "
         "representations z. Given z, the decoder generates an output sequence of "
         "symbols one element at a time. At each step the model is auto-regressive."),
        fontsize=10,
    )
    p2.insert_text((72, 420), "Figure 1: The Transformer model architecture.", fontsize=9)

    out = tmp / "digital_sample.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_blank_pdf(tmp: Path) -> Path:
    """Blank page PDF — simulates a scanned document."""
    import fitz
    doc = fitz.open()
    doc.new_page()
    out = tmp / "blank_sample.pdf"
    doc.save(str(out))
    doc.close()
    return out


# ── Chunker tests ──────────────────────────────────────────────────────────────

def test_chunker_import():
    from src.chunkers.semantic_chunker import SemanticChunker, Chunk  # noqa
    return "SemanticChunker imported"


def test_chunker_short_section():
    from src.chunkers.semantic_chunker import SemanticChunker
    from dataclasses import dataclass

    @dataclass
    class Sec:
        heading: str
        body: str

    sections = [Sec("Intro", "This is a short introduction.")]
    chunks   = SemanticChunker().chunk(sections, paper_id="test")
    assert len(chunks) == 1
    assert chunks[0].heading == "Intro"
    assert "short introduction" in chunks[0].body
    return f"1 chunk, heading={chunks[0].heading!r}"


def test_chunker_oversized_section():
    from src.chunkers.semantic_chunker import SemanticChunker
    from dataclasses import dataclass

    @dataclass
    class Sec:
        heading: str
        body: str

    para     = "This is a test sentence about deep learning. " * 20
    body     = "\n\n".join([para] * 5)   # ~4500 chars
    sections = [Sec("Methods", body)]
    chunker  = SemanticChunker(max_chars=500)
    chunks   = chunker.chunk(sections, paper_id="test")
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    for c in chunks:
        assert len(c.body) <= 650, f"Chunk too large: {len(c.body)}"
    return f"{len(chunks)} chunks from oversized section"


def test_chunker_sentence_split():
    from src.chunkers.semantic_chunker import SemanticChunker
    from dataclasses import dataclass

    @dataclass
    class Sec:
        heading: str
        body: str

    sentences = ["The model achieves state of the art results on the benchmark. "] * 30
    body      = "".join(sentences)
    sections  = [Sec("Results", body)]
    chunks    = SemanticChunker(max_chars=400).chunk(sections, paper_id="test")
    assert len(chunks) >= 2, f"Expected sentence-split chunks, got {len(chunks)}"
    return f"{len(chunks)} chunks via sentence split"


def test_chunker_deterministic_uuid():
    from src.chunkers.semantic_chunker import SemanticChunker
    from dataclasses import dataclass

    @dataclass
    class Sec:
        heading: str
        body: str

    sections = [Sec("A", "Body of section A."), Sec("B", "Body of section B.")]
    chunker  = SemanticChunker()
    ids1 = [c.chunk_id for c in chunker.chunk(sections, paper_id="paper-xyz")]
    ids2 = [c.chunk_id for c in chunker.chunk(sections, paper_id="paper-xyz")]
    ids3 = [c.chunk_id for c in chunker.chunk(sections, paper_id="paper-abc")]
    assert ids1 == ids2, "Same paper_id must produce identical chunk IDs"
    assert ids1 != ids3, "Different paper_id must produce different chunk IDs"
    return f"Deterministic: {ids1[0][:8]}…"


def test_chunker_small_merge():
    from src.chunkers.semantic_chunker import SemanticChunker
    from dataclasses import dataclass

    @dataclass
    class Sec:
        heading: str
        body: str

    sections = [Sec("A", "Very short."), Sec("B", "Also short.")]
    chunks   = SemanticChunker(min_chars=50).chunk(sections, paper_id="test")
    assert len(chunks) == 1, f"Expected 1 merged chunk, got {len(chunks)}"
    return f"Merged into {len(chunks)} chunk"


# ── Orchestrator tests ─────────────────────────────────────────────────────────

def test_orchestrator_import():
    from src.orchestrator import Orchestrator, PaperDocument  # noqa
    return "Orchestrator imported"


def test_orchestrator_instantiation(tmp: Path):
    from src.orchestrator import Orchestrator
    orch = Orchestrator(figures_dir=tmp / "figures", output_dir=tmp / "output")
    assert (tmp / "figures").exists()
    assert (tmp / "output").exists()
    return "dirs created"


def test_orchestrator_process_digital(tmp: Path):
    from src.orchestrator import Orchestrator, PaperDocument
    pdf  = _make_digital_pdf(tmp)
    doc  = Orchestrator(figures_dir=tmp/"figs_d", output_dir=tmp/"out_d").process(pdf)
    assert isinstance(doc, PaperDocument)
    assert len(doc.paper_id) == 36
    assert isinstance(doc.sections, list)
    assert isinstance(doc.equations, list)
    return (
        f"paper_id={doc.paper_id[:8]}…  "
        f"sections={len(doc.sections)}  "
        f"equations={len(doc.equations)}"
    )


def test_schema_validation(tmp: Path):
    from src.orchestrator import (Orchestrator, SectionOut,
                                   FigureOut, TableOut, CitationOut)
    doc = Orchestrator(figures_dir=tmp/"figs_s", output_dir=tmp/"out_s").process(
        _make_digital_pdf(tmp)
    )
    for s in doc.sections:
        assert isinstance(s, SectionOut)
        assert len(s.chunk_id) == 36
    for f in doc.figures:
        assert isinstance(f, FigureOut)
    for t in doc.tables:
        assert isinstance(t, TableOut)
        assert len(t.chunk_id) == 36
    for c in doc.citations:
        assert isinstance(c, CitationOut)
        assert isinstance(c.authors, list)
    return "All nested Pydantic types valid"


def test_json_output(tmp: Path):
    from src.orchestrator import Orchestrator
    out_dir = tmp / "out_json"
    doc     = Orchestrator(figures_dir=tmp/"figs_j", output_dir=out_dir).process(
        _make_digital_pdf(tmp)
    )
    json_file = out_dir / f"{doc.paper_id}.json"
    assert json_file.exists(), f"JSON not written: {json_file}"
    data      = json.loads(json_file.read_text(encoding="utf-8"))
    required  = {"paper_id","title","abstract","sections",
                 "figures","tables","equations","citations"}
    missing   = required - set(data.keys())
    assert not missing, f"Missing keys: {missing}"
    for s in data["sections"]:
        assert "chunk_id" in s and "heading" in s and "body" in s
    return f"JSON valid — {sorted(data.keys())}"


def test_orchestrator_process_scanned(tmp: Path):
    from src.orchestrator import Orchestrator, PaperDocument
    doc = Orchestrator(figures_dir=tmp/"figs_b", output_dir=tmp/"out_b").process(
        _make_blank_pdf(tmp)
    )
    assert isinstance(doc, PaperDocument) and doc.paper_id
    return f"Scanned PDF OK — sections={len(doc.sections)}"


def test_process_batch(tmp: Path):
    from src.orchestrator import Orchestrator
    import fitz
    batch = tmp / "batch"
    batch.mkdir()
    _make_digital_pdf(batch)
    p2 = fitz.open(); pg = p2.new_page()
    pg.insert_text((72, 80), "Second Paper", fontsize=14)
    pg.insert_text((72,120), "Body text for second paper.", fontsize=10)
    p2.save(str(batch / "second.pdf")); p2.close()

    docs = Orchestrator(figures_dir=tmp/"figs_bt", output_dir=tmp/"out_bt").process_batch(batch)
    assert len(docs) == 2, f"Expected 2, got {len(docs)}"
    return f"Batch: {len(docs)}/2 PDFs processed"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4 orchestrator tests")
    parser.add_argument("--pdf", type=Path, default=None,
                        help="Real academic PDF for end-to-end test")
    args = parser.parse_args()
    tmp  = Path(tempfile.mkdtemp())

    print(f"\n{BOLD}╔══════════════════════════════════════════════╗")
    print(      "║   Document Intelligence Engine — Stage 4    ║")
    print(      "║      Chunker & Orchestrator Tests           ║")
    print(      f"╚══════════════════════════════════════════════╝{RESET}\n")

    print(f"{BOLD}── Semantic Chunker ───────────────────────────{RESET}")
    check("SemanticChunker import",         test_chunker_import)
    check("Short section (no split)",       test_chunker_short_section)
    check("Oversized section (para split)", test_chunker_oversized_section)
    check("Oversized para (sent split)",    test_chunker_sentence_split)
    check("Deterministic UUIDs",            test_chunker_deterministic_uuid)
    check("Small chunk merging",            test_chunker_small_merge)

    print(f"\n{BOLD}── Orchestrator ───────────────────────────────{RESET}")
    check("Orchestrator import",            test_orchestrator_import)
    check("Orchestrator instantiation",     lambda: test_orchestrator_instantiation(tmp))
    check("process() digital PDF",         lambda: test_orchestrator_process_digital(tmp))
    check("PaperDocument schema",           lambda: test_schema_validation(tmp))
    check("JSON output on disk",            lambda: test_json_output(tmp))
    check("process() scanned PDF",          lambda: test_orchestrator_process_scanned(tmp))
    check("process_batch() directory",      lambda: test_process_batch(tmp))

    if args.pdf:
        print(f"\n{BOLD}── Real PDF: {args.pdf.name} ─────────────────────{RESET}")
        from src.orchestrator import Orchestrator
        real_out = tmp / "real_out"

        def _run_real():
            doc = Orchestrator(
                figures_dir=tmp / "real_figs",
                output_dir=real_out,
            ).process(args.pdf)
            return (
                f"title={doc.title[:50]!r}  "
                f"sections={len(doc.sections)}  "
                f"figures={len(doc.figures)}  "
                f"tables={len(doc.tables)}  "
                f"equations={len(doc.equations)}  "
                f"citations={len(doc.citations)}"
            )

        check("End-to-end real PDF", _run_real)

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
        print(f"\n\n  {BOLD}✓ All checks passed — Full pipeline complete!{RESET}")
        print(
            "\n  Your Document Intelligence Engine is production-ready.\n"
            "  Drop any PDF into data/input/ and run:\n\n"
            "      python -c \"\n"
            "      from pathlib import Path\n"
            "      from src.orchestrator import Orchestrator\n"
            "      doc = Orchestrator().process(Path('data/input/paper.pdf'))\n"
            "      print(doc.model_dump_json(indent=2))\n"
            "      \"\n"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
