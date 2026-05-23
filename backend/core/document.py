import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import httpx
import camelot

from config import get_settings
from core.logging import get_logger


settings = get_settings()

logger = get_logger("core.document")


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------


@dataclass
class DocumentResult:
    paper_id: str
    title: str
    abstract: str
    authors: list[dict]
    sections: list[dict]
    references: list[dict]
    figures: list[dict]
    tables: list[dict]
    equations: list[str]
    chunks: list[dict]
    metadata: dict[str, Any]





# -------------------------------------------------------------------
# GROBID Parsing
# -------------------------------------------------------------------


def parse_tei_xml(xml_content: str) -> dict | None:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_content)
    except Exception:
        return None
        
    ns = {"ns": "http://www.tei-c.org/ns/1.0"}
    
    # Title
    title_elem = root.find(".//ns:titleStmt/ns:title[@type='main']", ns)
    title = title_elem.text if title_elem is not None else ""
    
    # Abstract
    abstract_elem = root.find(".//ns:profileDesc/ns:abstract", ns)
    abstract = ""
    if abstract_elem is not None:
        paragraphs = abstract_elem.findall(".//ns:p", ns)
        abstract = "\n".join("".join(p.itertext()).strip() for p in paragraphs)
        if not abstract:
            abstract = "".join(abstract_elem.itertext()).strip()
            
    # Authors
    authors = []
    author_elems = root.findall(".//ns:analytic/ns:author", ns)
    for auth in author_elems:
        pers = auth.find("ns:persName", ns)
        if pers is not None:
            first = pers.find("ns:forename[@type='first']", ns)
            middle = pers.find("ns:forename[@type='middle']", ns)
            surname = pers.find("ns:surname", ns)
            
            first_name = first.text if first is not None else ""
            mid_name = middle.text if middle is not None else ""
            surname_name = surname.text if surname is not None else ""
            
            name_parts = [first_name, mid_name, surname_name]
            full_name = " ".join(part for part in name_parts if part).strip()
            
            email_elem = auth.find("ns:email", ns)
            email = email_elem.text if email_elem is not None else ""
            
            aff_elem = auth.find("ns:affiliation", ns)
            org = ""
            if aff_elem is not None:
                org_elem = aff_elem.find("ns:orgName[@type='institution']", ns)
                if org_elem is not None:
                    org = org_elem.text
            
            authors.append({
                "name": full_name,
                "email": email,
                "affiliation": org
            })
            
    # Sections
    sections = []
    body_elem = root.find(".//ns:body", ns)
    if body_elem is not None:
        div_elems = body_elem.findall("ns:div", ns)
        for div in div_elems:
            head_elem = div.find("ns:head", ns)
            head = "".join(head_elem.itertext()).strip() if head_elem is not None else ""
            
            p_elems = div.findall("ns:p", ns)
            p_texts = ["".join(p.itertext()).strip() for p in p_elems]
            body_text = "\n\n".join(p for p in p_texts if p)
            
            if head or body_text:
                sections.append({
                    "heading": head,
                    "body": body_text
                })
                
    # References
    references = []
    ref_elems = root.findall(".//ns:div[@type='references']//ns:biblStruct", ns)
    for ref in ref_elems:
        ref_title_elem = ref.find(".//ns:analytic/ns:title", ns)
        if ref_title_elem is None:
            ref_title_elem = ref.find(".//ns:monogr/ns:title", ns)
        ref_title = "".join(ref_title_elem.itertext()).strip() if ref_title_elem is not None else ""
        
        ref_authors = []
        for ref_auth in ref.findall(".//ns:analytic/ns:author", ns):
            ref_pers = ref_auth.find("ns:persName", ns)
            if ref_pers is not None:
                ref_surname = ref_pers.find("ns:surname", ns)
                ref_surname_val = ref_surname.text if ref_surname is not None else ""
                if ref_surname_val:
                    ref_authors.append(ref_surname_val)
                    
        year_elem = ref.find(".//ns:monogr/ns:imprint/ns:date", ns)
        year_val = None
        if year_elem is not None:
            when = year_elem.get("when")
            if when:
                try:
                    year_val = int(when.split("-")[0])
                except:
                    pass
                    
        references.append({
            "title": ref_title,
            "authors": ref_authors,
            "year": year_val
        })
        
    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "sections": sections,
        "references": references,
    }


async def parse_with_grobid(
    pdf_path: str
) -> dict | None:

    logger.info("document.parse_with_grobid.started", pdf_path=pdf_path)
    grobid_url = (
        f"{settings.GROBID_URL}"
        "/api/processFulltextDocument"
    )

    try:
        async with httpx.AsyncClient(timeout=300) as client:

            with open(pdf_path, "rb") as pdf_file:

                files = {
                    "input": (
                        Path(pdf_path).name,
                        pdf_file,
                        "application/pdf"
                    )
                }

                response = await client.post(
                    grobid_url,
                    files=files
                )

        if response.status_code != 200:
            return None

        xml_output = response.text.strip()

        if not xml_output:
            return None

        parsed = parse_tei_xml(xml_output)
        if parsed is None:
            return None

        parsed["metadata"] = {
            "source": "grobid",
            "raw_xml": xml_output
        }
        return parsed

    except Exception:
        return None


# -------------------------------------------------------------------
# OCR Fallback
# -------------------------------------------------------------------


def ocr_fallback(
    pdf_path: str
) -> str:
    import pytesseract
    from PIL import Image

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n\n".join(pages)


# -------------------------------------------------------------------
# Figure Extraction
# -------------------------------------------------------------------


def extract_figures(
    pdf_path: str,
    paper_id: str
) -> list[dict]:

    figures_dir = Path(settings.DATA_DIR) / "figures" / paper_id
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    extracted_figures = []
    
    for page_idx, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            path = figures_dir / f"fig_{page_idx}_{img_idx}.{base['ext']}"
            path.write_bytes(base["image"])
            extracted_figures.append({"page": page_idx + 1, "image_path": str(path), "bbox": None})
            
    doc.close()
    return extracted_figures


# -------------------------------------------------------------------
# Table Extraction
# -------------------------------------------------------------------


def extract_tables(
    pdf_path: str
) -> list[dict]:

    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, flavor=flavor, pages="all")
            return [
                {"page": t.page, "caption": "", "data": t.data, "df": t.df.to_dict()}
                for t in tables
            ]
        except Exception as exc:
            if flavor == "stream":
                logger.warning("document.extract_tables.failed", error=str(exc))
    return []


# -------------------------------------------------------------------
# Equation Detection
# -------------------------------------------------------------------


def detect_equations(
    text: str
) -> list[str]:

    patterns = [
        r"\$\$(.*?)\$\$",
        r"\$(.*?)\$",
        r"\\begin\{equation\}(.*?)\\end\{equation\}"
    ]

    equations = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.DOTALL
        )

        equations.extend(matches)

    return equations


# -------------------------------------------------------------------
# Semantic Chunking
# -------------------------------------------------------------------


def semantic_chunk(
    paper_id: str,
    grobid_output: dict,
    figures: list,
    tables: list,
    equations: list
) -> list[dict]:

    chunks = []

    sections = grobid_output.get(
        "sections",
        []
    )

    for section in sections:

        heading = section.get(
            "heading",
            "Unknown"
        )

        body = section.get(
            "body",
            ""
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            body
        )

        current_chunk = []

        current_length = 0

        for sentence in sentences:

            current_chunk.append(sentence)

            current_length += len(sentence)

            if current_length >= 1200:

                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "paper_id": paper_id,
                    "chunk_type": "text",
                    "section": heading,
                    "content": " ".join(current_chunk),
                    "metadata": {
                        "source": "section_text"
                    }
                })

                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "paper_id": paper_id,
                "chunk_type": "text",
                "section": heading,
                "content": " ".join(current_chunk),
                "metadata": {
                    "source": "section_text"
                }
            })

    for figure in figures:

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "paper_id": paper_id,
            "chunk_type": "figure",
            "section": "Figures",
            "content": figure["image_path"],
            "metadata": figure
        })

    for table in tables:

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "paper_id": paper_id,
            "chunk_type": "table",
            "section": "Tables",
            "content": str(table["data"]),
            "metadata": table
        })

    for equation in equations:

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "paper_id": paper_id,
            "chunk_type": "equation",
            "section": "Equations",
            "content": equation,
            "metadata": {}
        })

    return chunks


# -------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------


async def process_paper(
    pdf_path: str,
    paper_id: str
) -> DocumentResult:

    logger.info("document.process_paper.started", paper_id=paper_id, pdf_path=pdf_path)
    parsed = await parse_with_grobid(pdf_path)

    fallback_text = ""

    if (
        not parsed
        or len(
            parsed.get(
                "abstract",
                ""
            )
        ) < 200
    ):
        logger.warning("document.process_paper.ocr_fallback", paper_id=paper_id, pdf_path=pdf_path)
        fallback_text = ocr_fallback(pdf_path)

        parsed = {
            "title": "",
            "abstract": fallback_text[:2000],
            "authors": [],
            "sections": [{
                "heading": "OCR Content",
                "body": fallback_text
            }],
            "references": [],
            "metadata": {
                "source": "ocr_fallback"
            }
        }

    figures = extract_figures(
        pdf_path,
        paper_id
    )

    tables = extract_tables(pdf_path)

    combined_text = "\n".join([
        section.get("body", "")
        for section in parsed.get("sections", [])
    ])

    equations = detect_equations(combined_text)

    chunks = semantic_chunk(
        paper_id=paper_id,
        grobid_output=parsed,
        figures=figures,
        tables=tables,
        equations=equations
    )

    result = DocumentResult(
        paper_id=paper_id,
        title=parsed.get("title", ""),
        abstract=parsed.get("abstract", ""),
        authors=parsed.get("authors", []),
        sections=parsed.get("sections", []),
        references=parsed.get("references", []),
        figures=figures,
        tables=tables,
        equations=equations,
        chunks=chunks,
        metadata=parsed.get("metadata", {})
    )

    logger.info("document.process_paper.success", paper_id=paper_id)
    return result