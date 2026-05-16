"""
CLI entrypoint for the Research Discovery Pipeline.

Usage:
    python -m research_discovery.cli "Using LLMs for automated code review"
    python -m research_discovery.cli "attention mechanism transformers" --no-citation-expansion
    python -m research_discovery.cli "RLHF reward modeling" --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from research_discovery.core.pipeline import ResearchDiscoveryPipeline
from research_discovery.core.utils import get_logger

logger = get_logger("cli")


def _print_tier(name: str, papers: list) -> None:
    if not papers:
        return
    print(f"\n{'='*60}")
    print(f"  {name.upper().replace('_', ' ')} ({len(papers)} papers)")
    print(f"{'='*60}")
    for i, paper in enumerate(papers, 1):
        year = f" ({paper.year})" if paper.year else ""
        venue = f" | {paper.venue}" if paper.venue else ""
        doi = paper.external_ids.doi or paper.external_ids.arxiv or "–"
        print(f"\n{i:3}. [{paper.final_score:.3f}] {paper.title}{year}{venue}")
        print(f"       DOI/arXiv: {doi}")
        print(f"       Citations: {paper.citation_count:,}")
        if paper.abstract:
            print(f"       Abstract: {paper.abstract[:150]}...")


async def main(args: argparse.Namespace) -> None:
    pipeline = ResearchDiscoveryPipeline(
        num_expansion_queries=args.num_queries,
        results_per_query=args.results_per_query,
        use_semantic_scholar=not args.no_s2,
        use_arxiv=not args.no_arxiv,
        use_crossref_enrichment=not args.no_crossref,
        use_citation_expansion=not args.no_citation_expansion,
        max_final_papers=args.max_papers,
    )

    print(f"\nResearch Discovery Module")
    print(f"Research Idea: '{args.idea}'")
    print(f"Running pipeline...\n")

    result = await pipeline.run(args.idea)

    print(f"\n{'*'*60}")
    print(f"  RESULTS: {result.total_papers} papers in {result.processing_time_seconds}s")
    print(f"{'*'*60}")
    print(f"  Queries generated:      {result.metadata.get('num_expansion_queries', 0)}")
    print(f"  Raw papers retrieved:   {result.metadata.get('raw_retrieved', 0)}")
    print(f"  After deduplication:    {result.metadata.get('after_dedup', 0)}")
    print(f"  Embedding dimension:    {result.metadata.get('embedding_dimension', 0)}")

    if not args.quiet:
        _print_tier("highly_relevant", result.highly_relevant)
        _print_tier("relevant_background", result.relevant_background)
        _print_tier("adjacent_work", result.adjacent_work)
        _print_tier("historical_foundations", result.historical_foundations)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        print(f"\n✓ Results saved to {args.output}")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Research Discovery Module — find relevant academic papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("idea", help="Research idea or topic to discover papers for")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of expansion queries")
    parser.add_argument("--results-per-query", type=int, default=20, help="Results per API query")
    parser.add_argument("--max-papers", type=int, default=150, help="Max papers in final corpus")
    parser.add_argument("--no-s2", action="store_true", help="Skip Semantic Scholar")
    parser.add_argument("--no-arxiv", action="store_true", help="Skip arXiv")
    parser.add_argument("--no-crossref", action="store_true", help="Skip CrossRef enrichment")
    parser.add_argument("--no-citation-expansion", action="store_true", help="Skip citation expansion")
    parser.add_argument("--output", "-o", help="Save results as JSON to this file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print summary, not paper list")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI server instead")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve mode")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        uvicorn.run(
            "research_discovery.api:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
        )
    else:
        asyncio.run(main(args))


if __name__ == "__main__":
    cli()