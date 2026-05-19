"""
src/parsers/grobid_client.py
============================
Production GROBID REST client.

Responsibilities:
  - POST a PDF to GROBID /api/processFulltextDocument
  - Retry on transient failures with exponential back-off
  - Parse the returned TEI-XML into a clean intermediate dict:
      {title, abstract, sections[], citations[], raw_tei}
  - Never raises on a bad PDF — returns a GrobidResult with
    success=False and an error message so the orchestrator can
    route to the OCR fallback cleanly.

Usage:
    from src.parsers.grobid_client import GrobidClient
    client = GrobidClient()
    result = client.process_pdf(Path("paper.pdf"))
    if result.success:
        print(result.title)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import requests
from loguru import logger

# ── TEI-XML namespace ──────────────────────────────────────────────────────────
TEI_NS = "http://www.tei-c.org/ns/1.0"
NS     = {"tei": TEI_NS}

# ── Default config ─────────────────────────────────────────────────────────────
DEFAULT_URL          = os.getenv("GROBID_URL", "http://localhost:8070")
DEFAULT_TIMEOUT      = int(os.getenv("GROBID_TIMEOUT", "60"))
DEFAULT_MAX_RETRIES  = 3
DEFAULT_BACKOFF_BASE = 2.0   # seconds; doubles each retry


# ── Result dataclasses ─────────────────────────────────────────────────────────
@dataclass
class GrobidSection:
    heading: str
    body: str


@dataclass
class GrobidCitation:
    ref_id: str
    title: str
    authors: list[str]
    year: Optional[int]
    doi: Optional[str]


@dataclass
class GrobidResult:
    success:   bool
    error:     Optional[str]                 = None
    raw_tei:   Optional[str]                 = None
    title:     str                           = ""
    abstract:  str                           = ""
    sections:  list[GrobidSection]           = field(default_factory=list)
    citations: list[GrobidCitation]          = field(default_factory=list)


# ── Client ─────────────────────────────────────────────────────────────────────
class GrobidClient:
    """
    Thin wrapper around the GROBID REST API.

    Parameters
    ----------
    base_url : str
        Base URL of the GROBID server (no trailing slash).
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retry attempts on 5xx / connection errors.
    backoff_base : float
        Base multiplier for exponential back-off between retries.
    """

    def __init__(
        self,
        base_url:     str   = DEFAULT_URL,
        timeout:      int   = DEFAULT_TIMEOUT,
        max_retries:  int   = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ) -> None:
        self.base_url     = base_url.rstrip("/")
        self.timeout      = timeout
        self.max_retries  = max_retries
        self.backoff_base = backoff_base
        self._session     = requests.Session()

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """Return True if GROBID /api/isalive responds with 'true'."""
        try:
            r = self._session.get(
                f"{self.base_url}/api/isalive", timeout=self.timeout
            )
            return r.status_code == 200 and r.text.strip().lower() == "true"
        except requests.RequestException:
            return False

    def process_pdf(self, pdf_path: Path) -> GrobidResult:
        """
        Submit a PDF file to GROBID and parse the TEI-XML response.
        Returns a GrobidResult; never raises.
        """
        if not pdf_path.exists():
            return GrobidResult(success=False, error=f"File not found: {pdf_path}")

        tei_xml = self._post_with_retry(pdf_path)
        if tei_xml is None:
            return GrobidResult(
                success=False,
                error="GROBID returned no content after all retries",
            )

        return self._parse_tei(tei_xml)

    # ── HTTP layer ─────────────────────────────────────────────────────────────

    def _post_with_retry(self, pdf_path: Path) -> Optional[str]:
        """POST the PDF to GROBID; retry up to max_retries on failure."""
        endpoint = f"{self.base_url}/api/processFulltextDocument"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"GROBID attempt {attempt}/{self.max_retries}: {pdf_path.name}")
                with pdf_path.open("rb") as fh:
                    response = self._session.post(
                        endpoint,
                        files={"input": (pdf_path.name, fh, "application/pdf")},
                        data={
                            "consolidateHeader":    "1",
                            "consolidateCitations": "1",
                            "includeRawCitations":  "1",
                            "segmentSentences":     "0",
                        },
                        timeout=self.timeout,
                    )

                if response.status_code == 200:
                    logger.info(f"GROBID processed {pdf_path.name} successfully")
                    return response.text

                if response.status_code == 503:
                    logger.warning("GROBID busy (503) — backing off")
                    self._sleep(attempt)
                    continue

                # 4xx are not retryable
                logger.error(f"GROBID HTTP {response.status_code}: {response.text[:200]}")
                return None

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(f"Connection error on attempt {attempt}: {exc}")
                self._sleep(attempt)

        logger.error(f"All {self.max_retries} GROBID attempts failed. Last: {last_exc}")
        return None

    def _sleep(self, attempt: int) -> None:
        delay = self.backoff_base ** attempt
        logger.debug(f"Backing off {delay:.1f}s")
        time.sleep(delay)

    # ── TEI-XML parsing ────────────────────────────────────────────────────────

    def _parse_tei(self, tei_xml: str) -> GrobidResult:
        """Parse GROBID's TEI-XML into a structured GrobidResult."""
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as exc:
            return GrobidResult(
                success=False,
                error=f"TEI-XML parse error: {exc}",
                raw_tei=tei_xml,
            )

        title     = self._extract_title(root)
        abstract  = self._extract_abstract(root)
        sections  = self._extract_sections(root)
        citations = self._extract_citations(root)

        logger.info(
            f"Parsed TEI: title={title[:60]!r}, "
            f"sections={len(sections)}, citations={len(citations)}"
        )

        return GrobidResult(
            success=True,
            raw_tei=tei_xml,
            title=title,
            abstract=abstract,
            sections=sections,
            citations=citations,
        )

    # ── TEI element helpers ────────────────────────────────────────────────────

    @staticmethod
    def _text(element: Optional[ET.Element]) -> str:
        if element is None:
            return ""
        return " ".join(element.itertext()).strip()

    def _extract_title(self, root: ET.Element) -> str:
        el = root.find(".//tei:titleStmt/tei:title[@type='main']", NS)
        return self._text(el)

    def _extract_abstract(self, root: ET.Element) -> str:
        el = root.find(".//tei:profileDesc/tei:abstract", NS)
        return self._text(el)

    def _extract_sections(self, root: ET.Element) -> list[GrobidSection]:
        sections: list[GrobidSection] = []
        body = root.find(".//tei:body", NS)
        if body is None:
            return sections

        for div in body.findall(".//tei:div", NS):
            head = div.find("tei:head", NS)
            heading = self._text(head) if head is not None else ""

            paragraphs = [
                self._text(p)
                for p in div.findall("tei:p", NS)
                if self._text(p)
            ]
            body_text = "\n\n".join(paragraphs)

            if heading or body_text:
                sections.append(GrobidSection(heading=heading, body=body_text))

        return sections

    def _extract_citations(self, root: ET.Element) -> list[GrobidCitation]:
        citations: list[GrobidCitation] = []
        ref_list = root.find(".//tei:listBibl", NS)
        if ref_list is None:
            return citations

        for bib in ref_list.findall("tei:biblStruct", NS):
            ref_id  = bib.get("{http://www.w3.org/XML/1998/namespace}id", "")
            title   = self._text(bib.find(".//tei:title[@level='a']", NS)) or \
                      self._text(bib.find(".//tei:title[@level='m']", NS))
            authors = self._extract_authors(bib)
            year    = self._extract_year(bib)
            doi     = self._extract_doi(bib)

            citations.append(GrobidCitation(
                ref_id=ref_id,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
            ))

        return citations

    def _extract_authors(self, bib: ET.Element) -> list[str]:
        authors: list[str] = []
        for author in bib.findall(".//tei:author", NS):
            forename = self._text(author.find("tei:persName/tei:forename", NS))
            surname  = self._text(author.find("tei:persName/tei:surname", NS))
            name = f"{forename} {surname}".strip()
            if name:
                authors.append(name)
        return authors

    def _extract_year(self, bib: ET.Element) -> Optional[int]:
        date_el = bib.find(".//tei:date[@type='published']", NS)
        when    = (date_el.get("when") or "") if date_el is not None else ""
        try:
            return int(when[:4]) if when else None
        except ValueError:
            return None

    def _extract_doi(self, bib: ET.Element) -> Optional[str]:
        for idno in bib.findall(".//tei:idno[@type='DOI']", NS):
            val = (idno.text or "").strip()
            if val:
                return val
        return None
