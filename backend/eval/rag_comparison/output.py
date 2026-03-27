"""Output formatting and export for RAG comparison results."""

from __future__ import annotations

import json
from pathlib import Path

from backend.eval.rag_comparison.models import ComparisonResults


def print_comparison_report(results: ComparisonResults) -> None:
    """Print a formatted comparison table to stdout."""
    configs = results.configs
    if not configs:
        print("No results to display.")
        return

    # Header
    print("\n" + "=" * 90)
    print("RAG RETRIEVAL STRATEGY COMPARISON REPORT")
    print("=" * 90)

    # Table header
    header = f"{'Config':<28} {'Correct':>8} {'Relev':>8} {'Faith':>8} {'Composite':>10} {'Latency':>8} {'NaN':>4}"
    print(header)
    print("-" * 90)

    # Sort by composite descending
    sorted_configs = sorted(configs, key=lambda c: c.avg_composite, reverse=True)

    for i, c in enumerate(sorted_configs):
        marker = " *" if i == 0 else "  "
        row = (
            f"{c.config_name:<28}"
            f" {c.avg_correctness:>7.3f}"
            f" {c.avg_relevancy:>7.3f}"
            f" {c.avg_faithfulness:>7.3f}"
            f" {c.avg_composite:>9.3f}{marker}"
            f" {c.avg_latency:>6.1f}s"
            f" {c.total_nan_count:>4}"
        )
        print(row)

    print("-" * 90)

    # Winner summary
    winner = results.winner
    production = results.production_config
    if winner:
        print(f"\nWinner: {winner.config_name} (composite={winner.avg_composite:.4f})")

    if production and winner and production.config_name != winner.config_name:
        delta = winner.avg_composite - production.avg_composite
        sign = "+" if delta > 0 else ""
        print(f"vs Production ({production.config_name}): {sign}{delta:.4f} composite delta")
        print(f"  Correctness:  {production.avg_correctness:.3f} -> {winner.avg_correctness:.3f}")
        print(f"  Relevancy:    {production.avg_relevancy:.3f} -> {winner.avg_relevancy:.3f}")
        print(f"  Faithfulness: {production.avg_faithfulness:.3f} -> {winner.avg_faithfulness:.3f}")
        print(f"  Latency:      {production.avg_latency:.1f}s -> {winner.avg_latency:.1f}s")
    elif production and winner:
        print("Current production config is already optimal.")

    print()


def save_comparison_results(results: ComparisonResults, path: Path) -> None:
    """Save comparison results as JSON for CI/baseline tracking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Results saved to {path}")


__all__ = ["print_comparison_report", "save_comparison_results"]
