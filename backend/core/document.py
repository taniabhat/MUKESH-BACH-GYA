import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import camelot
import fitz
import httpx
import layoutparser as lp
from paddleocr import PaddleOCR

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
# OCR Engine
# -------------------------------------------------------------------


ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)


# -------------------------------------------------------------------
# Layout Parser Model
# -------------------------------------------------------------------


# layout_model = lp.PaddleDetectionLayoutModel(
#     config_path=(
#         "lp://PubLayNet/"
#         "ppyolov2_r50vd_dcn_365e_publaynet/config"
#     ),
#     label_map={
#         0: "Text",
#         1: "Title",
#         2: "List",
#         3: "Table",
#         4: "Figure"
#     },
#     enforce_cpu=True
# )
layout_model = None

# -------------------------------------------------------------------
# GROBID Parsing
# -------------------------------------------------------------------


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

        return {
            "title": "",
            "abstract": "",
            "authors": [],
            "sections": [],
            "references": [],
            "metadata": {
                "source": "grobid",
                "raw_xml": xml_output
            }
        }

    except Exception:
        return None


# -------------------------------------------------------------------
# OCR Fallback
# -------------------------------------------------------------------


def ocr_fallback(
    pdf_path: str
) -> str:

    document = fitz.open(pdf_path)

    full_text = []

    for page_index in range(len(document)):

        page = document[page_index]

        pixmap = page.get_pixmap()

        image_bytes = pixmap.samples

        results = ocr_engine.ocr(
            image_bytes,
            cls=True
        )

        page_text = []

        for block in results[0]:
            text = block[1][0]
            page_text.append(text)

        full_text.append("\n".join(page_text))

    document.close()

    return "\n\n".join(full_text)


# -------------------------------------------------------------------
# Figure Extraction
# -------------------------------------------------------------------


def extract_figures(
    pdf_path: str,
    paper_id: str
) -> list[dict]:

    figures_dir = (
        Path(settings.DATA_DIR)
        / "figures"
        / paper_id
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    document = fitz.open(pdf_path)

    extracted_figures = []

    for page_index in range(len(document)):

        page = document[page_index]

        pixmap = page.get_pixmap()

        image = pixmap.samples

        layout = layout_model.detect(image)

        figure_blocks = [
            block
            for block in layout
            if block.type == "Figure"
        ]

        for idx, block in enumerate(figure_blocks):

            bbox = block.coordinates

            rect = fitz.Rect(
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3]
            )

            clipped_pixmap = page.get_pixmap(
                clip=rect
            )

            image_path = (
                figures_dir
                / f"fig_{page_index}_{idx}.png"
            )

            clipped_pixmap.save(str(image_path))

            extracted_figures.append({
                "page": page_index + 1,
                "image_path": str(image_path),
                "bbox": bbox
            })

    document.close()

    return extracted_figures


# -------------------------------------------------------------------
# Table Extraction
# -------------------------------------------------------------------


def extract_tables(
    pdf_path: str
) -> list[dict]:

    extracted_tables = []

    try:
        tables = camelot.read_pdf(
            pdf_path,
            flavor="lattice",
            pages="all"
        )

    except Exception:
        tables = camelot.read_pdf(
            pdf_path,
            flavor="stream",
            pages="all"
        )

    for table in tables:

        extracted_tables.append({
            "page": table.page,
            "caption": "",
            "data": table.data,
            "df": table.df.to_dict()
        })

    return extracted_tables


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