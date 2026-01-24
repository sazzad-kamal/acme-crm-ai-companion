"""CLI entry point for text quality evaluation."""

import logging

import typer

from backend.eval.answer.text.runner import print_summary, run_text_eval


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run text quality evaluation using RAGAS metrics."""
    logging.basicConfig(level=logging.WARNING)
    results = run_text_eval(limit=limit, verbose=verbose)
    print_summary(results)


if __name__ == "__main__":
    typer.run(main)
