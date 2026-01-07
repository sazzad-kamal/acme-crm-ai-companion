"""
Shared evaluation utilities.

SLO checking, baseline comparison, results saving.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.table import Table

from backend.eval.formatting import console, format_check_mark, print_overall_result_panel

# =============================================================================
# Constants
# =============================================================================

REGRESSION_THRESHOLD = 0.05  # 5% regression threshold


# =============================================================================
# SLO Checking
# =============================================================================


def create_slo_table(
    slo_checks: list[tuple[str, bool, str, str]],
    title: str = "SLO Summary",
) -> Table:
    """
    Create a standard SLO results table.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples
        title: Table title

    Returns:
        Formatted Rich Table
    """
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("SLO", style="bold")
    table.add_column("Actual", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("Status", justify="center")

    for name, passed, actual, target in slo_checks:
        table.add_row(name, actual, target, format_check_mark(passed))

    return table


def print_slo_result(slo_checks: list[tuple[str, bool, str, str]]) -> bool:
    """
    Print SLO summary table and pass/fail status.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples

    Returns:
        True if all SLOs passed, False otherwise
    """
    console.print()
    slo_table = create_slo_table(slo_checks)
    console.print(slo_table)

    total_slos = len(slo_checks)
    failed_slo_names = [name for name, passed, _, _ in slo_checks if not passed]

    if failed_slo_names:
        console.print(f"\n[red bold][!] {len(failed_slo_names)} SLO(s) FAILED:[/red bold]")
        for slo_name in failed_slo_names:
            console.print(f"    [red]X[/red] {slo_name}")
        return False
    else:
        console.print(f"\n[green bold][OK] All {total_slos} SLOs passed[/green bold]")
        return True


def get_failed_slos(slo_checks: list[tuple[str, bool, str, str]]) -> list[str]:
    """
    Extract names of failed SLOs from slo_checks list.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples

    Returns:
        List of SLO names that failed
    """
    return [name for name, passed, _, _ in slo_checks if not passed]


def determine_exit_code(
    all_slos_passed: bool,
    is_regression: bool,
) -> int:
    """
    Determine exit code from SLO and regression status.

    Args:
        all_slos_passed: Whether all SLOs passed
        is_regression: Whether a regression was detected

    Returns:
        0 for success, 1 for failure
    """
    if not all_slos_passed or is_regression:
        return 1
    return 0


# =============================================================================
# Baseline Comparison (merged from baseline.py)
# =============================================================================


def compare_to_baseline(
    current_score: float,
    baseline_path: Path,
    score_key: str = "overall_score",
) -> tuple[bool, float | None]:
    """
    Compare current score to a baseline.

    Args:
        current_score: Current evaluation score
        baseline_path: Path to baseline JSON file
        score_key: Key in baseline JSON containing the score

    Returns:
        Tuple of (is_regression, baseline_score)
    """
    if not baseline_path.exists():
        return False, None

    try:
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        # Handle nested summary structure
        if "summary" in baseline_data:
            baseline_score = baseline_data["summary"].get(score_key, 0.0)
        else:
            baseline_score = baseline_data.get(score_key, 0.0)

        # Regression if we're more than threshold worse
        is_regression = current_score < (baseline_score - REGRESSION_THRESHOLD)

        return is_regression, baseline_score
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load baseline: {e}[/yellow]")
        return False, None


def save_baseline(
    summary: dict[str, Any],
    baseline_path: Path,
) -> None:
    """
    Save current results as new baseline.

    Args:
        summary: Summary dictionary to save
        baseline_path: Path to save baseline JSON
    """
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump({"summary": summary}, f, indent=2, default=str)
    console.print(f"[green]Baseline saved to {baseline_path}[/green]")


def print_baseline_comparison(
    current_score: float,
    baseline_score: float | None,
    is_regression: bool,
) -> None:
    """
    Print baseline comparison results.

    Args:
        current_score: Current evaluation score
        baseline_score: Baseline score (or None if no baseline)
        is_regression: Whether regression was detected
    """
    if baseline_score is None:
        console.print("[dim]No baseline found for comparison[/dim]")
        return

    diff = current_score - baseline_score
    diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"

    if is_regression:
        console.print("\n[red bold]REGRESSION DETECTED[/red bold]")
        console.print(
            f"  Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ({diff_str})"
        )
    else:
        color = "green" if diff >= 0 else "yellow"
        console.print(
            f"\n[dim]Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ([{color}]{diff_str}[/{color}])[/dim]"
        )


# =============================================================================
# Results Saving
# =============================================================================


def save_eval_results(
    output_path: str,
    summary: Any,
    results: list[Any],
    result_mapper: Callable[[Any], dict],
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        output_path: Path to save JSON results
        summary: Summary object (must have model_dump() or be dict)
        results: List of result objects
        result_mapper: Function to convert result object to dict
    """
    summary_dict = summary.model_dump() if hasattr(summary, "model_dump") else summary
    output_data = {
        "summary": summary_dict,
        "results": [result_mapper(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    console.print(f"[dim]Results saved to {output_path}[/dim]")


# =============================================================================
# CLI Finalization
# =============================================================================


def finalize_eval_cli(
    primary_score: float,
    slo_checks: list[tuple[str, bool, str, str]],
    baseline_path: Path,
    score_key: str,
    set_baseline: bool = False,
    baseline_data: dict | None = None,
    extra_failure_check: bool = False,
    extra_failure_reason: str = "",
) -> int:
    """
    Finalize evaluation CLI: handle baseline, SLOs, and exit code.

    Args:
        primary_score: The primary score to compare against baseline
        slo_checks: List of (name, passed, actual_value, target_value) tuples
        baseline_path: Path to baseline JSON file
        score_key: Key for score in baseline JSON
        set_baseline: Whether to save current results as baseline
        baseline_data: Data to save as baseline (required if set_baseline=True)
        extra_failure_check: Additional failure condition
        extra_failure_reason: Reason for extra failure

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    is_regression, baseline_score = compare_to_baseline(
        primary_score,
        baseline_path,
        score_key=score_key,
    )
    print_baseline_comparison(primary_score, baseline_score, is_regression)

    if set_baseline and baseline_data:
        save_baseline(baseline_data, baseline_path)

    failed_slos = get_failed_slos(slo_checks)
    all_slos_passed = not failed_slos
    exit_code = determine_exit_code(all_slos_passed, is_regression)

    if extra_failure_check:
        exit_code = 1

    failure_reasons = []
    if extra_failure_check and extra_failure_reason:
        failure_reasons.append(extra_failure_reason)
    if failed_slos:
        failure_reasons.append(f"{len(failed_slos)} SLOs failed: {', '.join(failed_slos)}")
    if is_regression:
        failure_reasons.append("Regression detected vs baseline")

    console.print()
    print_overall_result_panel(
        all_passed=exit_code == 0,
        failure_reasons=failure_reasons,
        success_message=f"All {len(slo_checks)} SLOs met, no regression detected",
    )

    return exit_code


__all__ = [
    # Constants
    "REGRESSION_THRESHOLD",
    # SLO
    "create_slo_table",
    "print_slo_result",
    "get_failed_slos",
    "determine_exit_code",
    # Baseline
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    # Results
    "save_eval_results",
    "finalize_eval_cli",
]
