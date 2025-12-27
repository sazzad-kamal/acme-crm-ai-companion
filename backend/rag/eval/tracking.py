"""
Evaluation Tracking: Regression Detection & Latency Budget Monitoring.

Provides:
- Comparison against previous evaluation runs
- Latency budget violation tracking
- Rich console output for both
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from backend.rag.pipeline.constants import LATENCY_BUDGETS, TOTAL_LATENCY_BUDGET_MS
from backend.rag.eval.models import DocsEvalSummary, EvalResult


console = Console()


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PREVIOUS_RESULTS_PATH = DATA_DIR / "eval_results_previous.json"


# =============================================================================
# Regression Detection
# =============================================================================

def load_previous_summary() -> DocsEvalSummary | None:
    """Load summary from previous evaluation run."""
    if not PREVIOUS_RESULTS_PATH.exists():
        return None

    try:
        with open(PREVIOUS_RESULTS_PATH) as f:
            data = json.load(f)
            if "summary" in data:
                return DocsEvalSummary(**data["summary"])
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load previous results: {e}[/yellow]")
    return None


def save_as_previous(results: list[EvalResult], summary: DocsEvalSummary) -> None:
    """Save current results as previous for next comparison."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "results": [r.model_dump() for r in results],
        "summary": summary.model_dump(),
    }
    with open(PREVIOUS_RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def compare_with_previous(
    current: DocsEvalSummary,
    previous: DocsEvalSummary | None,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Compare current results with previous run.

    Args:
        current: Current evaluation summary
        previous: Previous evaluation summary (or None)
        threshold: Minimum delta to flag as regression (default 5%)

    Returns:
        Dict with comparison details and regression flags
    """
    if previous is None:
        return {
            "has_previous": False,
            "regressions": [],
            "improvements": [],
        }

    metrics = [
        ("RAG Triad", "rag_triad_success"),
        ("Context Relevance", "context_relevance"),
        ("Answer Relevance", "answer_relevance"),
        ("Groundedness", "groundedness"),
        ("Doc Recall", "avg_doc_recall"),
    ]

    regressions = []
    improvements = []

    for label, attr in metrics:
        prev_val = getattr(previous, attr)
        curr_val = getattr(current, attr)
        delta = curr_val - prev_val

        if delta < -threshold:
            regressions.append({
                "metric": label,
                "previous": prev_val,
                "current": curr_val,
                "delta": delta,
            })
        elif delta > threshold:
            improvements.append({
                "metric": label,
                "previous": prev_val,
                "current": curr_val,
                "delta": delta,
            })

    # Latency comparison (inverse - increase is regression)
    latency_delta = current.p95_latency_ms - previous.p95_latency_ms
    latency_threshold = 500  # 500ms threshold for latency

    if latency_delta > latency_threshold:
        regressions.append({
            "metric": "P95 Latency",
            "previous": previous.p95_latency_ms,
            "current": current.p95_latency_ms,
            "delta": latency_delta,
            "unit": "ms",
        })
    elif latency_delta < -latency_threshold:
        improvements.append({
            "metric": "P95 Latency",
            "previous": previous.p95_latency_ms,
            "current": current.p95_latency_ms,
            "delta": latency_delta,
            "unit": "ms",
        })

    return {
        "has_previous": True,
        "regressions": regressions,
        "improvements": improvements,
        "previous_summary": previous,
    }


def print_regression_report(comparison: dict[str, Any]) -> None:
    """Print regression comparison report."""
    if not comparison["has_previous"]:
        console.print("[dim]No previous run to compare against. This run will be saved as baseline.[/dim]")
        return

    previous = comparison["previous_summary"]

    # Build comparison table
    table = Table(title="Quality Comparison (vs last run)", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Previous", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")

    metrics = [
        ("RAG Triad", "rag_triad_success", "%"),
        ("Context Relevance", "context_relevance", "%"),
        ("Answer Relevance", "answer_relevance", "%"),
        ("Groundedness", "groundedness", "%"),
        ("Doc Recall", "avg_doc_recall", "%"),
        ("P95 Latency", "p95_latency_ms", "ms"),
    ]

    for label, attr, unit in metrics:
        prev_val = getattr(previous, attr)
        # Get current value from comparison data
        curr_val = None
        delta = None

        # Find in regressions or improvements
        for item in comparison["regressions"] + comparison["improvements"]:
            if item["metric"] == label:
                curr_val = item["current"]
                delta = item["delta"]
                break

        # If not found in changes, compute manually
        if curr_val is None:
            continue

        # Format based on unit
        if unit == "%":
            prev_str = f"{prev_val:.1%}"
            curr_str = f"{curr_val:.1%}"
            delta_str = f"{delta:+.1%}"
        else:
            prev_str = f"{prev_val:.0f}{unit}"
            curr_str = f"{curr_val:.0f}{unit}"
            delta_str = f"{delta:+.0f}{unit}"

        # Color code delta
        if label == "P95 Latency":
            # For latency, lower is better
            delta_color = "green" if delta < 0 else "red" if delta > 0 else "white"
        else:
            # For quality metrics, higher is better
            delta_color = "green" if delta > 0 else "red" if delta < 0 else "white"

        table.add_row(label, prev_str, curr_str, f"[{delta_color}]{delta_str}[/{delta_color}]")

    console.print(table)

    # Regression/improvement summary
    if comparison["regressions"]:
        console.print(f"\n[red bold]⚠ REGRESSION DETECTED: {len(comparison['regressions'])} metrics declined[/red bold]")
        for r in comparison["regressions"]:
            console.print(f"  • {r['metric']}: {r['delta']:+.1%}" if r.get("unit") != "ms" else f"  • {r['metric']}: {r['delta']:+.0f}ms")

    if comparison["improvements"]:
        console.print(f"\n[green bold]✓ IMPROVEMENT: {len(comparison['improvements'])} metrics improved[/green bold]")


# =============================================================================
# Latency Budget Tracking
# =============================================================================

def extract_step_latencies(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """
    Extract per-step latencies from evaluation results.

    Returns:
        Dict mapping step_id to dict of {question_id: latency_ms}
    """
    step_latencies: dict[str, dict[str, float]] = {}

    for result in results:
        for step_id, elapsed_ms in result.step_timings.items():
            if step_id not in step_latencies:
                step_latencies[step_id] = {}
            step_latencies[step_id][result.question_id] = elapsed_ms

    return step_latencies


def analyze_budget_violations(
    results: list[EvalResult],
    step_timings: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Analyze latency budget violations.

    Args:
        results: List of evaluation results
        step_timings: Optional pre-computed step timings per question

    Returns:
        Dict with budget analysis
    """
    violations: list[dict[str, Any]] = []
    step_stats: dict[str, dict[str, float]] = {}

    # If we have detailed step timings
    if step_timings:
        for step_id, budget in LATENCY_BUDGETS.items():
            timings = step_timings.get(step_id, {})
            if not timings:
                continue

            values = list(timings.values())
            avg = sum(values) / len(values) if values else 0
            p95_idx = int(len(values) * 0.95)
            p95 = sorted(values)[min(p95_idx, len(values) - 1)] if values else 0

            exceeded_count = sum(1 for v in values if v > budget)

            step_stats[step_id] = {
                "budget": budget,
                "avg": avg,
                "p95": p95,
                "exceeded_count": exceeded_count,
                "total_count": len(values),
            }

            if exceeded_count > 0:
                # Find which questions exceeded
                exceeded_questions = [qid for qid, v in timings.items() if v > budget]
                violations.append({
                    "step": step_id,
                    "budget": budget,
                    "exceeded_count": exceeded_count,
                    "questions": exceeded_questions[:3],  # First 3
                })

    # Also check total latency against P95 SLO
    total_violations = []
    for result in results:
        if result.latency_ms > TOTAL_LATENCY_BUDGET_MS:
            total_violations.append({
                "question_id": result.question_id,
                "latency_ms": result.latency_ms,
                "over_by": result.latency_ms - TOTAL_LATENCY_BUDGET_MS,
            })

    return {
        "step_stats": step_stats,
        "step_violations": violations,
        "total_violations": total_violations,
        "total_budget": TOTAL_LATENCY_BUDGET_MS,
    }


def print_budget_report(
    results: list[EvalResult],
    step_timings: dict[str, dict[str, float]] | None = None,
) -> None:
    """Print latency budget report."""
    analysis = analyze_budget_violations(results, step_timings)

    console.print("\n")

    # Step budget table (if we have step timings)
    if analysis["step_stats"]:
        table = Table(title="Latency Budget Summary", show_header=True, header_style="bold cyan")
        table.add_column("Step", style="bold")
        table.add_column("Budget", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("Exceeded", justify="right")

        for step_id in LATENCY_BUDGETS:
            stats = analysis["step_stats"].get(step_id)
            if not stats:
                continue

            budget = stats["budget"]
            avg = stats["avg"]
            p95 = stats["p95"]
            exceeded = stats["exceeded_count"]
            total = stats["total_count"]

            # Color code
            status = "[green]✓[/green]" if exceeded == 0 else f"[yellow]⚠ {exceeded}/{total}[/yellow]"

            table.add_row(
                step_id,
                f"{budget}ms",
                f"{avg:.0f}ms",
                f"{p95:.0f}ms",
                status,
            )

        console.print(table)

    # Total latency violations
    if analysis["total_violations"]:
        console.print(f"\n[yellow bold]⚠ {len(analysis['total_violations'])} questions exceeded total latency budget ({analysis['total_budget']}ms)[/yellow bold]")

        for v in analysis["total_violations"][:5]:  # Show first 5
            console.print(f"  • {v['question_id']}: {v['latency_ms']:.0f}ms (+{v['over_by']:.0f}ms over)")
    else:
        console.print(f"\n[green]✓ All questions within total latency budget ({analysis['total_budget']}ms)[/green]")


# =============================================================================
# Combined Report
# =============================================================================

def print_full_tracking_report(
    results: list[EvalResult],
    summary: DocsEvalSummary,
    step_timings: dict[str, dict[str, float]] | None = None,
) -> None:
    """
    Print full tracking report with regression detection and budget analysis.

    Args:
        results: Current evaluation results
        summary: Current evaluation summary
        step_timings: Optional per-step timing data (auto-extracted if None)
    """
    console.print(Panel(
        "[bold]Evaluation Tracking Report[/bold]",
        border_style="blue",
    ))

    # 1. Regression detection
    previous = load_previous_summary()
    comparison = compare_with_previous(summary, previous)
    print_regression_report(comparison)

    # 2. Extract step timings from results if not provided
    if step_timings is None:
        step_timings = extract_step_latencies(results)

    # 3. Budget analysis with per-step breakdown
    print_budget_report(results, step_timings)

    # 4. Add to history and show trends
    from backend.rag.eval.history import add_to_history, print_trend_report
    add_to_history(summary)
    print_trend_report(num_runs=10)

    # 5. Save current as previous for next comparison
    save_as_previous(results, summary)
    console.print("\n[dim]Current results saved for next comparison.[/dim]")
