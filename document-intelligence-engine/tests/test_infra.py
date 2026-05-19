"""
test_infra.py — Stage 1 Infrastructure Smoke Tests
===================================================
Verifies:
  1. GROBID REST API is reachable and alive
  2. GROBID /api/version returns a parseable response
  3. PaddleOCR initialises without linking errors
  4. LayoutParser imports cleanly
  5. Camelot + Ghostscript binding works
  6. PyMuPDF (fitz) opens without errors
  7. OpenCV (headless) loads without libGL errors

Run:
  python tests/test_infra.py

All checks print PASS / FAIL with actionable error messages.
Exit code 0 = everything green; non-zero = one or more failures.
"""

import sys
import os
import importlib
import traceback

import requests

# ── Config ────────────────────────────────────────────────────────────────────
GROBID_BASE_URL = os.getenv("GROBID_URL", "http://localhost:8070")
GROBID_TIMEOUT  = int(os.getenv("GROBID_TIMEOUT", "10"))

PASS  = "\033[92m  [PASS]\033[0m"
FAIL  = "\033[91m  [FAIL]\033[0m"
SKIP  = "\033[93m  [SKIP]\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []   # (name, passed, detail)


def check(name: str, fn) -> bool:
    """Run fn(); record and print PASS/FAIL."""
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
        # Print abbreviated traceback for actionable debugging
        lines = [l for l in tb.splitlines() if "File" in l or "Error" in l]
        for l in lines[-3:]:
            print(f"         {l.strip()}")
        return False


# ── 1. GROBID isalive ─────────────────────────────────────────────────────────
def _grobid_alive():
    url = f"{GROBID_BASE_URL}/api/isalive"
    r = requests.get(url, timeout=GROBID_TIMEOUT)
    r.raise_for_status()
    body = r.text.strip()
    assert body.lower() == "true", f"Unexpected body: {body!r}"
    return f"GET {url} → 200 OK, body={body!r}"


# ── 2. GROBID version ─────────────────────────────────────────────────────────
def _grobid_version():
    url = f"{GROBID_BASE_URL}/api/version"
    r = requests.get(url, timeout=GROBID_TIMEOUT)
    r.raise_for_status()
    version = r.json() if "application/json" in r.headers.get("Content-Type","") else r.text.strip()
    return f"version={version}"


# ── 3. PaddleOCR loads ────────────────────────────────────────────────────────
def _paddleocr_import():
    from paddleocr import PaddleOCR  # noqa: F401
    # Instantiate with minimal settings — no GPU, English only
    # lang='en' avoids downloading extra model weights at test time
    ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    assert ocr is not None
    return "PaddleOCR instantiated (use_angle_cls=False, lang=en)"


# ── 4. LayoutParser imports ───────────────────────────────────────────────────
def _layoutparser_import():
    import layoutparser as lp  # noqa: F401
    return f"layoutparser version={lp.__version__}"


# ── 5. Camelot + Ghostscript ──────────────────────────────────────────────────
def _camelot_import():
    import camelot  # noqa: F401
    # Probe Ghostscript binding (camelot uses it for PDF rendering)
    from camelot.utils import get_ghostscript_version  # type: ignore
    gs_ver = get_ghostscript_version()
    return f"camelot ok, ghostscript={gs_ver}"


# ── 6. PyMuPDF (fitz) ─────────────────────────────────────────────────────────
def _pymupdf_import():
    import fitz  # noqa: F401
    return f"PyMuPDF version={fitz.__version__}"


# ── 7. OpenCV headless ────────────────────────────────────────────────────────
def _opencv_import():
    import cv2  # noqa: F401
    return f"OpenCV version={cv2.__version__}"


# ── 8. Pydantic (schema layer) ────────────────────────────────────────────────
def _pydantic_import():
    import pydantic
    return f"pydantic version={pydantic.__version__}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}╔══════════════════════════════════════════════╗")
    print(      "║   Document Intelligence Engine — Stage 1    ║")
    print(      "║            Infrastructure Tests             ║")
    print(      f"╚══════════════════════════════════════════════╝{RESET}\n")

    print(f"{BOLD}── Network / GROBID ───────────────────────────{RESET}")
    check("GROBID isalive endpoint",  _grobid_alive)
    check("GROBID version endpoint",  _grobid_version)

    print(f"\n{BOLD}── Python library imports ─────────────────────{RESET}")
    check("PaddleOCR initialisation", _paddleocr_import)
    check("LayoutParser import",      _layoutparser_import)
    check("Camelot + Ghostscript",    _camelot_import)
    check("PyMuPDF (fitz) import",    _pymupdf_import)
    check("OpenCV headless import",   _opencv_import)
    check("Pydantic v2 import",       _pydantic_import)

    # ── Summary ───────────────────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{BOLD}── Results ────────────────────────────────────{RESET}")
    print(f"  {passed}/{total} checks passed", end="")
    if failed:
        print(f"  ·  {failed} FAILED — see details above")
        print(f"\n{BOLD}  Failing checks:{RESET}")
        for name, ok, detail in results:
            if not ok:
                print(f"    • {name}: {detail}")
        print()
        sys.exit(1)
    else:
        print(f"\n\n  {BOLD}✓ All checks passed — Stage 1 infrastructure is solid.{RESET}")
        print("  Ready to proceed to Stage 2: Core PDF Parsing.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
