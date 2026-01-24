"""CLI entry point for action quality evaluation."""

import logging

import typer

from backend.eval.answer.action.runner import print_summary, run_action_eval


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run action quality evaluation using LLM Judge."""
    logging.basicConfig(level=logging.WARNING)
    results = run_action_eval(limit=limit, verbose=verbose)
    print_summary(results)


if __name__ == "__main__":
    typer.run(main)
