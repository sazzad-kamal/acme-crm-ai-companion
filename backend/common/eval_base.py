"""
Shared evaluation utilities for agent and RAG evaluation harnesses.

Provides:
- Rich console formatting helpers
- Common summary table rendering
- Shared metrics computation
- Result saving utilities
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# Shared console instance
console = Console()


# =============================================================================
# Table Creation Helpers
# =============================================================================

def create_summary_table(title: str = "Evaluation Summary") -> Table:
    """Create a standard summary table with consistent styling."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    return table


def create_detail_table(title: str, columns: list[tuple[str, str]]) -> Table:
    """
    Create a detail table with custom columns.

    Args:
        title: Table title
        columns: List of (name, justify) tuples
    """
    table = Table(title=title, show_header=True, header_style="bold")
    for name, justify in columns:
        table.add_column(name, justify=justify)
    return table


def create_comparison_table(title: str, columns: list[tuple[str, str, str]]) -> Table:
    """
    Create a comparison table with custom columns and styles.

    Args:
        title: Table title
        columns: List of (name, justify, style) tuples
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for name, justify, style in columns:
        table.add_column(name, justify=justify, style=style if style else None)
    return table


# =============================================================================
# Formatting Helpers
# =============================================================================

def format_check_mark(value: bool) -> str:
    """Format boolean as colored check/cross mark."""
    return "[green]Y[/green]" if value else "[red]X[/red]"


def format_percentage(value: float, thresholds: tuple[float, float] = (0.9, 0.7)) -> str:
    """
    Format percentage with color based on thresholds.

    Args:
        value: Float between 0 and 1
        thresholds: Tuple of (green_threshold, yellow_threshold)

    Returns:
        Colored percentage string
    """
    green_thresh, yellow_thresh = thresholds
    if value >= green_thresh:
        color = "green"
    elif value >= yellow_thresh:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{value:.1%}[/{color}]"


def format_latency(ms: float, slo_ms: float | None = None) -> str:
    """
    Format latency with optional SLO comparison.

    Args:
        ms: Latency in milliseconds
        slo_ms: Optional SLO threshold for coloring

    Returns:
        Formatted latency string
    """
    if slo_ms is not None:
        color = "green" if ms <= slo_ms else "red"
        return f"[{color}]{ms:.0f}ms[/{color}]"
    return f"{ms:.0f}ms"


def format_delta(value: float, is_positive_good: bool = True) -> str:
    """
    Format a delta value with color.

    Args:
        value: Delta value (can be positive or negative)
        is_positive_good: If True, positive is green, negative is red

    Returns:
        Colored delta string
    """
    if is_positive_good:
        color = "green" if value >= 0 else "red"
    else:
        color = "red" if value > 0 else "green"
    return f"[{color}]{value:+.1%}[/{color}]" if abs(value) < 1 else f"[{color}]{value:+.0f}[/{color}]"


# =============================================================================
# Panel Helpers
# =============================================================================

def print_eval_header(title: str, subtitle: str) -> None:
    """Print evaluation header panel."""
    console.print(Panel(
        subtitle,
        title=title,
        border_style="blue",
    ))


def print_issues_panel(title: str, issues: list[str]) -> None:
    """Print issues panel if there are any issues."""
    if issues:
        console.print(Panel(
            "\n\n".join(issues),
            title=f"[red]{title}[/red]",
            border_style="red",
        ))


def print_success_panel(title: str, message: str) -> None:
    """Print success panel."""
    console.print(Panel(
        message,
        title=f"[green]{title}[/green]",
        border_style="green",
    ))


# =============================================================================
# Table Row Helpers
# =============================================================================

def add_separator_row(table: Table) -> None:
    """Add a visual separator row to a table."""
    table.add_row("─" * 20, "─" * 10)


def add_metric_row(
    table: Table,
    metric: str,
    value: float,
    format_type: str = "percent",
    threshold: float | None = None,
    slo_label: str | None = None,
) -> None:
    """
    Add a metric row to a summary table.

    Args:
        table: Rich Table to add to
        metric: Metric name
        value: Metric value
        format_type: "percent", "latency", or "count"
        threshold: Optional threshold for coloring
        slo_label: Optional SLO label to append
    """
    if format_type == "percent":
        if threshold is not None:
            color = "green" if value >= threshold else "red"
            formatted = f"[{color}]{value:.1%}[/{color}]"
        else:
            formatted = f"{value:.1%}"
    elif format_type == "latency":
        if threshold is not None:
            color = "green" if value <= threshold else "red"
            formatted = f"[{color}]{value:.0f}ms[/{color}]"
        else:
            formatted = f"{value:.0f}ms"
    else:  # count
        formatted = f"{value:,.0f}"

    if slo_label:
        formatted = f"{formatted} {slo_label}"

    table.add_row(metric, formatted)


# =============================================================================
# Results I/O
# =============================================================================

def save_results_json(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: List of result dictionaries
        summary: Summary dictionary
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "results": results,
        "summary": summary,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


def load_results_json(input_path: Path) -> tuple[list[dict], dict]:
    """
    Load evaluation results from JSON file.

    Args:
        input_path: Path to input file

    Returns:
        Tuple of (results list, summary dict)
    """
    with open(input_path) as f:
        data = json.load(f)
    return data.get("results", []), data.get("summary", {})


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_p95(values: list[float]) -> float:
    """Compute P95 of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    p95_index = int(len(sorted_values) * 0.95)
    return sorted_values[min(p95_index, len(sorted_values) - 1)]


def compute_pass_rate(
    values: list[int | bool],
    threshold: int = 1,
) -> float:
    """
    Compute pass rate for binary values.

    Args:
        values: List of 0/1 or True/False values
        threshold: Value considered as pass (default 1)

    Returns:
        Pass rate as float between 0 and 1
    """
    if not values:
        return 0.0
    passes = sum(1 for v in values if v >= threshold)
    return passes / len(values)


# =============================================================================
# Baseline Comparison
# =============================================================================

REGRESSION_THRESHOLD = 0.05  # 5% regression threshold


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
        console.print(f"\n[red bold]REGRESSION DETECTED[/red bold]")
        console.print(f"  Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ({diff_str})")
    else:
        color = "green" if diff >= 0 else "yellow"
        console.print(f"\n[dim]Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ([{color}]{diff_str}[/{color}])[/dim]")


__all__ = [
    "console",
    "create_summary_table",
    "create_detail_table",
    "create_comparison_table",
    "format_check_mark",
    "format_percentage",
    "format_latency",
    "format_delta",
    "print_eval_header",
    "print_issues_panel",
    "print_success_panel",
    "add_separator_row",
    "add_metric_row",
    "save_results_json",
    "load_results_json",
    "compute_p95",
    "compute_pass_rate",
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    "REGRESSION_THRESHOLD",
]
