"""
CLI runtime for Research Discovery Platform.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from research_discovery.api import (
    DiscoveryAPIService,
)
from research_discovery.core.runtime import (
    get_logger,
)
from research_discovery.models.paper import (
    DiscoveryResult,
    Paper,
)
from research_discovery.services.storage.service import (
    JSONStorageService,
)

logger = get_logger("cli")


# ---------------------------------------------------------------------------
# Terminal Rendering
# ---------------------------------------------------------------------------

class TerminalRenderer:
    """Terminal output rendering."""

    @staticmethod
    def print_banner() -> None:

        print(
            "\n"
            "Research Discovery Platform\n"
            "===========================\n"
        )

    @staticmethod
    def print_summary(
        result: DiscoveryResult,
    ) -> None:

        metadata = result.metadata

        print(
            "\n"
            f"Results: {result.total_papers} papers\n"
            f"Elapsed: {result.processing_time_seconds:.2f}s\n"
        )

        print(
            f"Queries Generated: "
            f"{metadata.get('num_expansion_queries', 0)}"
        )

        print(
            f"Raw Retrieved: "
            f"{metadata.get('raw_retrieved', 0)}"
        )

        print(
            f"After Dedup: "
            f"{metadata.get('after_dedup', 0)}"
        )

        print(
            f"Embedding Dimension: "
            f"{metadata.get('embedding_dimension', 0)}"
        )

    @staticmethod
    def print_tier(
        tier_name: str,
        papers: list[Paper],
    ) -> None:

        if not papers:
            return

        title = (
            tier_name.upper()
            .replace("_", " ")
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"{title} ({len(papers)} papers)"
        )

        print(
            "=" * 80
        )

        for index, paper in enumerate(
            papers,
            start=1,
        ):

            TerminalRenderer._print_paper(
                index,
                paper,
            )

    @staticmethod
    def _print_paper(
        index: int,
        paper: Paper,
    ) -> None:

        year = (
            f" ({paper.year})"
            if paper.year
            else ""
        )

        venue = (
            f" | {paper.venue}"
            if paper.venue
            else ""
        )

        identifier = (
            paper.external_ids.doi
            or paper.external_ids.arxiv
            or "—"
        )

        print(
            f"\n"
            f"{index:3}. "
            f"[{paper.final_score:.3f}] "
            f"{paper.title}"
            f"{year}"
            f"{venue}"
        )

        print(
            f"     DOI/arXiv: {identifier}"
        )

        print(
            f"     Citations: "
            f"{paper.citation_count:,}"
        )

        if paper.abstract:

            preview = (
                paper.abstract[:160]
                .replace("\n", " ")
            )

            print(
                f"     Abstract: "
                f"{preview}..."
            )


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@dataclass
class DiscoverCommand:

    idea: str

    output_path: Optional[str]

    quiet: bool


# ---------------------------------------------------------------------------
# CLI Runtime
# ---------------------------------------------------------------------------

class CLIApplication:
    """
    Research Discovery CLI runtime.
    """

    def __init__(self):

        self.discovery_service = (
            DiscoveryAPIService()
        )

        self.storage = (
            JSONStorageService()
        )

    async def run_discover(
        self,
        command: DiscoverCommand,
    ) -> int:

        try:

            TerminalRenderer.print_banner()

            print(
                f"Research Idea:\n"
                f"{command.idea}\n"
            )

            result = (
                await self.discovery_service.discover(
                    command.idea
                )
            )

            TerminalRenderer.print_summary(
                result
            )

            if not command.quiet:

                self._render_tiers(
                    result
                )

            if command.output_path:

                self.storage.save_result(
                    result,
                    filepath=command.output_path
                )

                print(
                    "\n"
                    f"Saved results to "
                    f"{command.output_path}"
                )

            return 0

        except KeyboardInterrupt:

            logger.warning(
                "CLI interrupted by user"
            )

            return 130

        except Exception:

            logger.exception(
                "CLI execution failed"
            )

            return 1

    @staticmethod
    def _render_tiers(
        result: DiscoveryResult,
    ) -> None:

        TerminalRenderer.print_tier(
            "highly_relevant",
            result.highly_relevant,
        )

        TerminalRenderer.print_tier(
            "relevant_background",
            result.relevant_background,
        )

        TerminalRenderer.print_tier(
            "adjacent_work",
            result.adjacent_work,
        )

        TerminalRenderer.print_tier(
            "historical_foundations",
            result.historical_foundations,
        )


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def build_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Research Discovery Platform CLI"
        )
    )

    subcommands = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    discover = (
        subcommands.add_parser(
            "discover",
            help=(
                "Run discovery pipeline"
            ),
        )
    )

    discover.add_argument(
        "idea",
        help=(
            "Research topic or idea"
        ),
    )

    discover.add_argument(
        "--output",
        "-o",
        help=(
            "Save result JSON"
        ),
    )

    discover.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help=(
            "Hide paper listings"
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def async_main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    app = CLIApplication()

    if args.command == "discover":

        command = DiscoverCommand(
            idea=args.idea,
            output_path=args.output,
            quiet=args.quiet,
        )

        return await app.run_discover(
            command
        )

    return 1


def cli() -> None:

    exit_code = asyncio.run(
        async_main()
    )

    sys.exit(exit_code)


if __name__ == "__main__":

    cli()