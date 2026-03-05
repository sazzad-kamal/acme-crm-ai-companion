"""Regression gate for CI/CD integration.

Provides a CLI command that runs evaluation and exits with non-zero
status if any SLO fails. Use this in CI pipelines to block merges
that degrade model quality.

Usage:
    python -m backend.eval.integration.gate [--limit N] [--baseline path.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from backend.eval.integration.models import ConvoEvalResults
from backend.eval.integration.output import SLO_SPECS, print_summary
from backend.eval.integration.runner import run_convo_eval

app = typer.Typer()


def load_baseline(path: Path) -> dict | None:
    """Load baseline results from JSON file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_regression(results: ConvoEvalResults, baseline: dict) -> list[str]:
    """Check for regressions against baseline.

    Returns list of regression warnings (empty if no regressions).
    """
    regressions = []

    # Check pass rate regression (allow 2% drop)
    baseline_pass_rate = baseline.get("summary", {}).get("pass_rate", 0)
    if results.pass_rate < baseline_pass_rate - 0.02:
        regressions.append(
            f"Pass rate regression: {results.pass_rate:.1%} "
            f"(baseline: {baseline_pass_rate:.1%}, threshold: 2% drop)"
        )

    # Check answer quality regressions (allow 5% drop)
    for metric in ["avg_relevance", "avg_faithfulness", "avg_answer_correctness"]:
        baseline_val = baseline.get("summary", {}).get(metric, 0)
        current_val = getattr(results, metric, 0)
        if current_val < baseline_val - 0.05:
            regressions.append(
                f"{metric} regression: {current_val:.1%} "
                f"(baseline: {baseline_val:.1%}, threshold: 5% drop)"
            )

    # Check latency regressions (allow 20% increase)
    for metric in ["latency_p50_ms", "latency_p95_ms"]:
        baseline_val = baseline.get("summary", {}).get(metric, 0)
        current_val = getattr(results, metric, 0)
        if baseline_val > 0 and current_val > baseline_val * 1.2:
            regressions.append(
                f"{metric} regression: {current_val:.0f}ms "
                f"(baseline: {baseline_val:.0f}ms, threshold: 20% increase)"
            )

    return regressions


def check_slos(results: ConvoEvalResults) -> list[str]:
    """Check all SLOs and return list of failures."""
    failures = []

    for spec in SLO_SPECS:
        value = spec.get_value(results)
        passed = value >= spec.target if spec.compare == ">=" else value <= spec.target

        if not passed:
            if spec.fmt == "pct":
                failures.append(
                    f"{spec.label}: {value:.1%} (SLO: {spec.compare}{spec.target:.1%})"
                )
            else:
                failures.append(
                    f"{spec.label}: {value:.0f}ms (SLO: {spec.compare}{spec.target:.0f}ms)"
                )

    # RAGAS reliability check
    if results.ragas_success_rate < 0.9:
        failures.append(
            f"RAGAS Reliability: {results.ragas_success_rate:.1%} (SLO: >=90%)"
        )

    return failures


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths"),
    baseline: Path | None = typer.Option(None, "--baseline", "-b", help="Baseline JSON file for regression check"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save results to JSON"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Exit 1 on any SLO failure"),
) -> None:
    """Run evaluation gate for CI/CD.

    Exits with status code:
    - 0: All SLOs passed (and no regressions if baseline provided)
    - 1: SLO failures or regressions detected
    """
    print("Running evaluation gate...")
    results = run_convo_eval(max_paths=limit)

    # Print summary
    print_summary(results)

    # Check SLOs
    slo_failures = check_slos(results)

    # Check regressions if baseline provided
    regressions: list[str] = []
    if baseline:
        baseline_data = load_baseline(baseline)
        if baseline_data:
            regressions = check_regression(results, baseline_data)
            if regressions:
                print("\nREGRESSIONS DETECTED:")
                for r in regressions:
                    print(f"  - {r}")
        else:
            print(f"\nWARNING: Baseline file not found: {baseline}")

    # Save results if requested
    if output:
        from backend.eval.integration.output import save_results
        save_results(results, output)

    # Determine exit status
    if strict and (slo_failures or regressions):
        print("\nGATE FAILED")
        if slo_failures:
            print("SLO Failures:")
            for f in slo_failures:
                print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nGATE PASSED")
        sys.exit(0)


if __name__ == "__main__":
    app()
