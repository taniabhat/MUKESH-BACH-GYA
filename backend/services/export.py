import shutil
import subprocess
import tempfile
from pathlib import Path


class ExportError(Exception):
    pass


class CompileError(Exception):
    pass


# -------------------------------------------------------------------
# Markdown -> LaTeX
# -------------------------------------------------------------------


def to_latex(
    markdown_content: str,
    output_path: str
) -> str:

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with tempfile.NamedTemporaryFile(
        suffix=".md",
        delete=False,
        mode="w",
        encoding="utf-8"
    ) as temp_markdown:

        temp_markdown.write(
            markdown_content
        )

        temp_markdown_path = (
            temp_markdown.name
        )

    command = [
        "pandoc",
        "--from",
        "markdown",
        "--to",
        "latex",
        temp_markdown_path,
        "-o",
        str(output)
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    Path(temp_markdown_path).unlink(
        missing_ok=True
    )

    if process.returncode != 0:

        raise ExportError(
            process.stderr
        )

    return str(output)


# -------------------------------------------------------------------
# Template Injection
# -------------------------------------------------------------------


def inject_template(
    latex_content: str,
    template_path: str
) -> str:

    template = Path(
        template_path
    ).read_text(
        encoding="utf-8"
    )

    document_marker = (
        "\\begin{document}"
    )

    if document_marker not in template:

        raise ExportError(
            "Invalid IEEE template"
        )

    preamble = """
\\documentclass[conference]{IEEEtran}

\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{cite}
\\usepackage{hyperref}
\\usepackage{float}
"""

    before = template.split(
        document_marker
    )[0]

    after = template.split(
        document_marker
    )[1]

    final_document = (
        before
        + preamble
        + "\n"
        + document_marker
        + "\n"
        + latex_content
        + "\n"
        + after
    )

    return final_document


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def run_command(
    command: list[str],
    cwd: str
) -> subprocess.CompletedProcess:

    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True
    )


# -------------------------------------------------------------------
# Compile LaTeX
# -------------------------------------------------------------------


def compile_latex(
    tex_path: str,
    output_dir: str
) -> str:

    tex_file = Path(tex_path)

    if not tex_file.exists():

        raise CompileError(
            "TeX file does not exist"
        )

    output_directory = Path(
        output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    tex_filename = tex_file.name

    base_name = tex_file.stem

    working_tex = (
        output_directory
        / tex_filename
    )

    shutil.copy2(
        tex_file,
        working_tex
    )

    commands = [
        [
            "pdflatex",
            "-interaction=nonstopmode",
            tex_filename
        ],

        [
            "bibtex",
            base_name
        ],

        [
            "pdflatex",
            "-interaction=nonstopmode",
            tex_filename
        ],

        [
            "pdflatex",
            "-interaction=nonstopmode",
            tex_filename
        ]
    ]

    logs = []

    for command in commands:

        process = run_command(
            command,
            str(output_directory)
        )

        logs.append(process.stdout)
        logs.append(process.stderr)

        if process.returncode != 0:

            raise CompileError(
                "\n".join(logs)
            )

    pdf_path = (
        output_directory
        / f"{base_name}.pdf"
    )

    if not pdf_path.exists():

        raise CompileError(
            "Failed to generate PDF"
        )

    return str(pdf_path)


# -------------------------------------------------------------------
# LaTeX -> DOCX
# -------------------------------------------------------------------


def to_docx(
    tex_path: str,
    output_path: str
) -> str:

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "pandoc",
        "--from",
        "latex",
        "--to",
        "docx",
        tex_path,
        "-o",
        str(output)
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if process.returncode != 0:

        raise ExportError(
            process.stderr
        )

    return str(output)


# -------------------------------------------------------------------
# BibTeX Generation
# -------------------------------------------------------------------


def determine_entry_type(
    citation: dict
) -> str:

    venue = (
        citation.get(
            "venue",
            ""
        )
        .lower()
    )

    if any(
        keyword in venue
        for keyword in [
            "conference",
            "proceedings",
            "symposium",
            "workshop"
        ]
    ):
        return "inproceedings"

    if any(
        keyword in venue
        for keyword in [
            "journal",
            "transactions",
            "letters"
        ]
    ):
        return "article"

    if any(
        keyword in venue
        for keyword in [
            "report",
            "technical report"
        ]
    ):
        return "techreport"

    return "misc"


def sanitize_bibtex_key(
    title: str
) -> str:

    cleaned = "".join(
        character
        for character in title.lower()
        if character.isalnum()
    )

    return cleaned[:40]


def generate_bibtex_entry(
    citation: dict
) -> str:

    entry_type = determine_entry_type(
        citation
    )

    title = citation.get(
        "title",
        "Untitled"
    )

    authors = citation.get(
        "authors",
        []
    )

    venue = citation.get(
        "venue",
        "Unknown Venue"
    )

    year = citation.get(
        "year",
        "2025"
    )

    doi = citation.get(
        "doi",
        ""
    )

    publisher = citation.get(
        "publisher",
        ""
    )

    bibtex_key = sanitize_bibtex_key(
        title
    )

    author_string = " and ".join(
        authors
    )

    fields = [
        f"title = {{{title}}}",
        f"author = {{{author_string}}}",
        f"year = {{{year}}}"
    ]

    if venue:

        if entry_type == "article":

            fields.append(
                f"journal = {{{venue}}}"
            )

        elif entry_type == "inproceedings":

            fields.append(
                f"booktitle = {{{venue}}}"
            )

        else:

            fields.append(
                f"howpublished = {{{venue}}}"
            )

    if doi:

        fields.append(
            f"doi = {{{doi}}}"
        )

    if publisher:

        fields.append(
            f"publisher = {{{publisher}}}"
        )

    joined_fields = ",\n  ".join(
        fields
    )

    return (
        f"@{entry_type}"
        f"{{{bibtex_key},\n"
        f"  {joined_fields}\n"
        f"}}"
    )