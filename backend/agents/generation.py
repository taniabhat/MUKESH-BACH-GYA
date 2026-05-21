import ast
import json
import subprocess
import tempfile
from pathlib import Path

import docker
from sqlalchemy import desc
from sqlalchemy import select

from core.llm import build_system_message
from core.llm import build_user_message
from core.llm import chat
from core.llm import get_model
from core.logging import get_logger
from models.db import Citation
from models.db import GeneratedAsset
from models.db import PaperDraft
from models.db import AsyncSessionLocal
from models.db import Project
from config import settings
from prompts.templates import DIAGRAM_GENERATOR


logger = get_logger("agents.generation")


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------


BASE_GENERATED_DIR = Path(
    "/data/generated"
)

EXPORT_DIR = settings.exports_dir


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


async def load_latest_draft(
    project_id: str
) -> PaperDraft:

    async with AsyncSessionLocal() as db:

        query = (
            select(PaperDraft)
            .where(
                PaperDraft.project_id
                == project_id
            )
            .order_by(
                desc(PaperDraft.version)
            )
        )

        result = await db.execute(query)

        draft = result.scalar_one_or_none()

        if not draft:
            raise ValueError(
                "Draft not found"
            )

        return draft


async def load_citations(
    project_id: str
) -> list[Citation]:

    async with AsyncSessionLocal() as db:

        query = select(Citation).where(
            Citation.project_id
            == project_id
        )

        result = await db.execute(query)

        return list(
            result.scalars().all()
        )


async def save_asset(
    project_id: str,
    asset_type: str,
    content: str | None,
    file_path: str
) -> None:

    async with AsyncSessionLocal() as db:

        asset = GeneratedAsset(
            project_id=project_id,
            asset_type=asset_type,
            content=content,
            file_path=file_path
        )

        db.add(asset)

        await db.commit()


# -------------------------------------------------------------------
# Code Generation
# -------------------------------------------------------------------


async def design_experiment(
    methodology_text: str
) -> dict:

    prompt = f"""
Analyze this methodology section.

Identify:
- task type
- framework
- baselines
- metrics
- datasets
- training requirements

Methodology:
{methodology_text}
"""

    response = await chat(
        messages=[
            build_system_message(
                """
Return a structured JSON experiment design.
"""
            ),
            build_user_message(prompt)
        ],
        model=get_model("research"),
        temperature=0.2,
        max_tokens=2048
    )

    try:
        return json.loads(response)

    except Exception:

        return {
            "framework": "PyTorch",
            "task": "classification",
            "metrics": ["accuracy"]
        }


async def generate_model_code(
    design: dict
) -> str:

    prompt = f"""
Generate model.py

Experiment Design:
{design}
"""

    return await chat(
        messages=[
            build_system_message(
                """
Generate production-grade PyTorch model code.
"""
            ),
            build_user_message(prompt)
        ],
        model=get_model("code"),
        temperature=0.2,
        max_tokens=4096
    )


async def generate_train_code(
    design: dict
) -> str:

    prompt = f"""
Generate train.py

Requirements:
- optimizer
- scheduler
- checkpointing
- early stopping

Design:
{design}
"""

    return await chat(
        messages=[
            build_system_message(
                """
Generate robust training pipeline code.
"""
            ),
            build_user_message(prompt)
        ],
        model=get_model("code"),
        temperature=0.2,
        max_tokens=4096
    )


async def generate_eval_code(
    design: dict
) -> str:

    prompt = f"""
Generate evaluate.py

Design:
{design}
"""

    return await chat(
        messages=[
            build_system_message(
                """
Generate evaluation script with metrics.
"""
            ),
            build_user_message(prompt)
        ],
        model=get_model("code"),
        temperature=0.2,
        max_tokens=4096
    )


async def generate_dataloader(
    design: dict
) -> str:

    prompt = f"""
Generate data_loader.py

Design:
{design}
"""

    return await chat(
        messages=[
            build_system_message(
                """
Generate PyTorch DataLoader setup.
"""
            ),
            build_user_message(prompt)
        ],
        model=get_model("code"),
        temperature=0.2,
        max_tokens=4096
    )


# -------------------------------------------------------------------
# Sandbox
# -------------------------------------------------------------------


def run_in_sandbox(
    code_files: dict,
    run_cmd: str
) -> dict:

    client = docker.from_env()

    with tempfile.TemporaryDirectory() as temp_dir:

        workspace = Path(temp_dir)

        for filename, code in code_files.items():

            file_path = workspace / filename

            file_path.write_text(code)

        container = client.containers.run(
            image="research-os-sandbox:latest",
            command=run_cmd,
            volumes={
                str(workspace): {
                    "bind": "/workspace",
                    "mode": "rw"
                }
            },
            network_disabled=True,
            read_only=True,
            working_dir="/workspace",
            detach=True
        )

        result = container.wait()

        stdout = container.logs(
            stdout=True,
            stderr=False
        ).decode()

        stderr = container.logs(
            stdout=False,
            stderr=True
        ).decode()

        container.remove(force=True)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result["StatusCode"],
            "output_files": [
                file.name
                for file in workspace.iterdir()
            ]
        }


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------


def validate_code_syntax(
    code: str,
    filename: str
) -> bool:

    if not filename.endswith(".py"):
        return True

    try:

        ast.parse(code)

        return True

    except SyntaxError:

        return False


# -------------------------------------------------------------------
# Code Generation Pipeline
# -------------------------------------------------------------------


async def run_code_generation(
    project_id: str
) -> dict:

    logger.info("generation.code.started", project_id=project_id)
    draft = await load_latest_draft(
        project_id
    )

    methodology = draft.sections.get(
        "Methodology",
        ""
    )

    design = await design_experiment(
        methodology
    )

    model_code = await generate_model_code(
        design
    )

    train_code = await generate_train_code(
        design
    )

    eval_code = await generate_eval_code(
        design
    )

    dataloader_code = (
        await generate_dataloader(
            design
        )
    )

    code_files = {
        "model.py": model_code,
        "train.py": train_code,
        "evaluate.py": eval_code,
        "data_loader.py": dataloader_code
    }

    output_dir = (
        BASE_GENERATED_DIR
        / project_id
        / "code"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for filename, code in code_files.items():

        if not validate_code_syntax(
            code,
            filename
        ):
            continue

        file_path = output_dir / filename

        file_path.write_text(code)

        await save_asset(
            project_id=project_id,
            asset_type="code",
            content=code,
            file_path=str(file_path)
        )

    sandbox_result = run_in_sandbox(
        code_files,
        "python train.py --help"
    )

    logger.info("generation.code.success", project_id=project_id, files=list(code_files.keys()))
    return {
        "generated_files":
            list(code_files.keys()),

        "sandbox_result":
            sandbox_result
    }


# -------------------------------------------------------------------
# Diagram Generation
# -------------------------------------------------------------------


async def generate_mermaid_diagram(
    description: str,
    diagram_type: str
) -> str:

    layout_hint = {
        "architecture": "flowchart LR",
        "pipeline": "flowchart TD",
        "sequence": "sequenceDiagram"
    }

    prompt = f"""
Generate Mermaid diagram.

Type:
{diagram_type}

Layout:
{layout_hint.get(diagram_type)}

Description:
{description}
"""

    response = await chat(
        messages=[
            build_system_message(
                DIAGRAM_GENERATOR
            ),
            build_user_message(prompt)
        ],
        model=get_model("writing"),
        temperature=0.3,
        max_tokens=2048
    )

    return response.strip()


def validate_diagram_syntax(
    mermaid_code: str
) -> bool:

    try:

        process = subprocess.run(
            [
                "mmdc",
                "--input",
                "/dev/stdin",
                "--output",
                "/dev/null"
            ],
            input=mermaid_code.encode(),
            capture_output=True,
            check=False
        )

        return process.returncode == 0

    except Exception:
        return False


def render_diagram(
    mermaid_code: str,
    output_path: str
) -> str:

    temp_input = (
        Path(output_path)
        .with_suffix(".mmd")
    )

    temp_input.write_text(
        mermaid_code
    )

    subprocess.run(
        [
            "mmdc",
            "-i",
            str(temp_input),
            "-o",
            output_path
        ],
        check=True
    )

    return output_path


# -------------------------------------------------------------------
# Diagram Pipeline
# -------------------------------------------------------------------


async def run_diagram_generation(
    project_id: str
) -> dict:

    logger.info("generation.diagram.started", project_id=project_id)
    draft = await load_latest_draft(
        project_id
    )

    methodology = draft.sections.get(
        "Methodology",
        ""
    )

    requested_diagrams = [
        {
            "name":
                "architecture",

            "description":
                methodology,

            "type":
                "architecture"
        }
    ]

    output_dir = (
        BASE_GENERATED_DIR
        / project_id
        / "diagrams"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    generated = []

    for diagram in requested_diagrams:

        mermaid = (
            await generate_mermaid_diagram(
                diagram["description"],
                diagram["type"]
            )
        )

        valid = validate_diagram_syntax(
            mermaid
        )

        if not valid:

            mermaid = (
                await generate_mermaid_diagram(
                    diagram["description"],
                    diagram["type"]
                )
            )

        output_path = (
            output_dir
            / f"{diagram['name']}.svg"
        )

        rendered = render_diagram(
            mermaid,
            str(output_path)
        )

        await save_asset(
            project_id=project_id,
            asset_type="diagram",
            content=mermaid,
            file_path=rendered
        )

        generated.append(rendered)

    logger.info("generation.diagram.success", project_id=project_id, diagrams=generated)
    return {
        "diagrams": generated
    }


# -------------------------------------------------------------------
# IEEE Export
# -------------------------------------------------------------------


def assemble_ieee_sections(
    draft_sections: dict,
    figures: list,
    tables: list
) -> str:

    ordered_sections = [
        "Abstract",
        "Introduction",
        "Related Work",
        "Methodology",
        "Experiments",
        "Results",
        "Discussion",
        "Conclusion"
    ]

    assembled = []

    section_mapping = {
        "Introduction":
            "I. Introduction",

        "Related Work":
            "II. Related Work",

        "Methodology":
            "III. Proposed Methodology",

        "Experiments":
            "IV. Experimental Setup",

        "Results":
            "V. Results",

        "Discussion":
            "VI. Discussion",

        "Conclusion":
            "VII. Conclusion"
    }

    for section in ordered_sections:

        content = draft_sections.get(
            section,
            ""
        )

        heading = (
            section_mapping.get(
                section,
                section
            )
        )

        assembled.append(
            f"# {heading}\n\n{content}"
        )

    for figure in figures:

        assembled.append(
            f"""
![Figure]({figure})

Figure: Generated figure.
"""
        )

    for table in tables:

        assembled.append(
            f"""
Table:
{table}
"""
        )

    return "\n\n".join(assembled)


def build_bibtex(
    citations: list[dict]
) -> str:

    entries = []

    for index, citation in enumerate(
        citations,
        start=1
    ):

        title = citation.get(
            "title",
            "Untitled"
        )

        authors = citation.get(
            "authors",
            []
        )

        year = citation.get(
            "year",
            "2025"
        )

        venue = citation.get(
            "venue",
            "Unknown"
        )

        entry = f"""
@article{{ref{index},
  title={{{title}}},
  author={{{' and '.join(authors)}}},
  journal={{{venue}}},
  year={{{year}}}
}}
"""

        entries.append(entry)

    return "\n".join(entries)


# -------------------------------------------------------------------
# Export Pipeline
# -------------------------------------------------------------------


async def run_ieee_export(
    project_id: str,
    fmt: str = "pdf"
) -> str:

    logger.info("generation.export.started", project_id=project_id, format=fmt)
    draft = await load_latest_draft(
        project_id
    )

    citations = await load_citations(
        project_id
    )

    figures = []

    tables = []

    markdown = assemble_ieee_sections(
        draft.sections,
        figures,
        tables
    )

    bibtex = build_bibtex([
        {
            "title": citation.bibtex,
            "authors": [],
            "year": "2025",
            "venue": "Unknown"
        }
        for citation in citations
    ])

    export_dir = (
        EXPORT_DIR
        / project_id
    )

    export_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    markdown_path = (
        export_dir
        / "paper.md"
    )

    bib_path = (
        export_dir
        / "references.bib"
    )

    markdown_path.write_text(
        markdown
    )

    bib_path.write_text(
        bibtex
    )

    output_path = (
        export_dir
        / f"paper.{fmt}"
    )

    if fmt == "tex":

        subprocess.run(
            [
                "pandoc",
                str(markdown_path),
                "-o",
                str(output_path)
            ],
            check=True
        )

    elif fmt == "pdf":

        latex_path = (
            export_dir
            / "paper.tex"
        )

        subprocess.run(
            [
                "pandoc",
                str(markdown_path),
                "-o",
                str(latex_path)
            ],
            check=True
        )

        subprocess.run(
            [
                "pdflatex",
                "-output-directory",
                str(export_dir),
                str(latex_path)
            ],
            check=True
        )

        output_path = (
            export_dir
            / "paper.pdf"
        )

    elif fmt == "docx":

        subprocess.run(
            [
                "pandoc",
                str(markdown_path),
                "-o",
                str(output_path)
            ],
            check=True
        )

    else:

        raise ValueError(
            f"Unsupported format: {fmt}"
        )

    await save_asset(
        project_id=project_id,
        asset_type="paper",
        content=None,
        file_path=str(output_path)
    )

    logger.info("generation.export.success", project_id=project_id, format=fmt, output_path=str(output_path))
    return str(output_path)