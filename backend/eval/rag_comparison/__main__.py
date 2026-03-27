"""CLI for RAG retrieval strategy comparison.

Usage:
    python -m backend.eval.rag_comparison
    python -m backend.eval.rag_comparison --limit 5
    python -m backend.eval.rag_comparison --configs vector_top5,hybrid_top5
    python -m backend.eval.rag_comparison --limit 3 --output results.json
"""

from __future__ import annotations

import asyncio
import platform
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

# Fix Windows asyncio cleanup issues with httpx/RAGAS
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined,unused-ignore]

import logging

import typer

from backend.eval.rag_comparison.output import print_comparison_report, save_comparison_results
from backend.eval.rag_comparison.runner import run_rag_comparison

app = typer.Typer()


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max questions per config"),
    configs: str | None = typer.Option(None, "--configs", "-c", help="Comma-separated config names to test"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
) -> None:
    """Compare RAG retrieval strategies using RAGAS metrics."""
    logging.basicConfig(level=logging.WARNING)

    config_names = [c.strip() for c in configs.split(",")] if configs else None

    try:
        results = run_rag_comparison(config_names=config_names, limit=limit)
    except Exception as e:
        print(f"\nERROR: Comparison failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)

    print_comparison_report(results)

    if output:
        save_comparison_results(results, Path(output))


if __name__ == "__main__":
    app()
