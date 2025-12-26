"""
Combined agent evaluation runner.

Runs all agent evaluation suites:
1. Tool evaluation (correctness)
2. Router evaluation (mode selection)
3. End-to-end evaluation (full pipeline)

SLOs (Service Level Objectives):
- Tool accuracy ≥ 90%
- Router accuracy ≥ 90%
- E2E relevance ≥ 80%
- E2E groundedness ≥ 80%
- P95 latency ≤ 5000ms

Usage:
    python -m backend.agent.eval.run_all
    python -m backend.agent.eval.run_all --skip-e2e  # Skip expensive E2E tests
    python -m backend.agent.eval.run_all --verbose
    python -m backend.agent.eval.run_all --baseline path/to/baseline.json  # Compare to baseline
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.agent.eval.tool_eval import run_tool_eval, print_tool_eval_results
from backend.agent.eval.router_eval import run_router_eval, print_router_eval_results
from backend.agent.eval.e2e_eval import run_e2e_eval, print_e2e_eval_results
from backend.agent.eval.models import (
    AgentEvalSummary,
    SLO_TOOL_ACCURACY,
    SLO_ROUTER_ACCURACY,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
    SLO_LATENCY_P95_MS,
)


console = Console()

OUTPUT_PATH = Path("data/processed/agent_eval_results.json")
BASELINE_PATH = Path("data/processed/agent_eval_baseline.json")


def check_slos(summary: AgentEvalSummary) -> tuple[bool, list[str]]:
    """
    Check if all SLOs are met.
    
    Returns:
        Tuple of (all_passed, list_of_failures)
    """
    failures = []
    
    # Tool accuracy SLO
    if summary.tool_eval.accuracy < SLO_TOOL_ACCURACY:
        failures.append(
            f"Tool accuracy {summary.tool_eval.accuracy:.1%} < {SLO_TOOL_ACCURACY:.1%}"
        )
    
    # Router accuracy SLO (combine mode + company)
    router_combined = (
        summary.router_eval.mode_accuracy + 
        summary.router_eval.company_extraction_accuracy
    ) / 2
    if router_combined < SLO_ROUTER_ACCURACY:
        failures.append(
            f"Router accuracy {router_combined:.1%} < {SLO_ROUTER_ACCURACY:.1%}"
        )
    
    # E2E SLOs (only if run)
    if summary.e2e_eval:
        if summary.e2e_eval.answer_relevance_rate < SLO_ANSWER_RELEVANCE:
            failures.append(
                f"E2E relevance {summary.e2e_eval.answer_relevance_rate:.1%} < {SLO_ANSWER_RELEVANCE:.1%}"
            )
        if summary.e2e_eval.groundedness_rate < SLO_GROUNDEDNESS:
            failures.append(
                f"E2E groundedness {summary.e2e_eval.groundedness_rate:.1%} < {SLO_GROUNDEDNESS:.1%}"
            )
        if summary.e2e_eval.p95_latency_ms > SLO_LATENCY_P95_MS:
            failures.append(
                f"P95 latency {summary.e2e_eval.p95_latency_ms:.0f}ms > {SLO_LATENCY_P95_MS}ms"
            )
    
    return len(failures) == 0, failures


def compare_to_baseline(
    summary: AgentEvalSummary, 
    baseline_path: Path | None = None,
) -> tuple[bool, float | None]:
    """
    Compare current results to a baseline.
    
    Returns:
        Tuple of (is_regression, baseline_score)
    """
    path = baseline_path or BASELINE_PATH
    if not path.exists():
        return False, None
    
    try:
        with open(path) as f:
            baseline_data = json.load(f)
        
        baseline_score = baseline_data.get("overall_score", 0.0)
        current_score = summary.overall_score
        
        # Regression if we're more than 5% worse
        regression_threshold = 0.05
        is_regression = current_score < (baseline_score - regression_threshold)
        
        return is_regression, baseline_score
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load baseline: {e}[/yellow]")
        return False, None


def compute_overall_score(
    tool_accuracy: float,
    router_mode_accuracy: float,
    router_company_accuracy: float,
    e2e_relevance: float,
    e2e_groundedness: float,
    e2e_tool_selection: float,
) -> float:
    """
    Compute weighted overall score.
    
    Weights:
    - Tool correctness: 20%
    - Router accuracy: 20%
    - E2E answer quality: 60%
    """
    tool_score = tool_accuracy * 0.20
    router_score = ((router_mode_accuracy + router_company_accuracy) / 2) * 0.20
    e2e_score = ((e2e_relevance + e2e_groundedness + e2e_tool_selection) / 3) * 0.60
    
    return tool_score + router_score + e2e_score


def run_all_evals(
    skip_e2e: bool = False,
    e2e_limit: int | None = None,
    verbose: bool = False,
    save_results: bool = True,
    baseline_path: Path | None = None,
) -> AgentEvalSummary:
    """
    Run all agent evaluation suites.
    
    Args:
        skip_e2e: Skip end-to-end evaluation (expensive)
        e2e_limit: Limit E2E tests
        verbose: Print detailed progress
        save_results: Save results to JSON
        baseline_path: Path to baseline JSON for regression comparison
        
    Returns:
        Complete AgentEvalSummary
    """
    console.print(Panel(
        "[bold]Agent Evaluation Suite[/bold]\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="blue",
    ))
    
    # 1. Tool Evaluation
    console.print("\n[bold cyan]1/3 Tool Evaluation[/bold cyan]")
    tool_results, tool_summary = run_tool_eval(verbose=verbose)
    print_tool_eval_results(tool_results, tool_summary)
    
    # 2. Router Evaluation
    console.print("\n[bold cyan]2/3 Router Evaluation[/bold cyan]")
    router_results, router_summary = run_router_eval(verbose=verbose)
    print_router_eval_results(router_results, router_summary)
    
    # 3. End-to-End Evaluation
    e2e_summary = None
    if not skip_e2e:
        console.print("\n[bold cyan]3/3 End-to-End Evaluation[/bold cyan]")
        e2e_results, e2e_summary = run_e2e_eval(limit=e2e_limit, verbose=verbose)
        print_e2e_eval_results(e2e_results, e2e_summary)
    else:
        console.print("\n[dim]3/3 End-to-End Evaluation - SKIPPED[/dim]")
    
    # Compute overall score
    overall_score = compute_overall_score(
        tool_accuracy=tool_summary.accuracy,
        router_mode_accuracy=router_summary.mode_accuracy,
        router_company_accuracy=router_summary.company_extraction_accuracy,
        e2e_relevance=e2e_summary.answer_relevance_rate if e2e_summary else 0.8,
        e2e_groundedness=e2e_summary.groundedness_rate if e2e_summary else 0.8,
        e2e_tool_selection=e2e_summary.tool_selection_accuracy if e2e_summary else 0.8,
    )
    
    # Build initial summary
    summary = AgentEvalSummary(
        tool_eval=tool_summary,
        router_eval=router_summary,
        e2e_eval=e2e_summary,
        overall_score=overall_score,
    )
    
    # Check SLOs
    all_slos_passed, failed_slos = check_slos(summary)
    summary.all_slos_passed = all_slos_passed
    summary.failed_slos = failed_slos
    
    # Check for regression
    is_regression, baseline_score = compare_to_baseline(summary, baseline_path)
    summary.regression_detected = is_regression
    summary.baseline_score = baseline_score
    
    # Print final summary
    console.print("\n")
    score_status = "green" if overall_score >= 0.8 and all_slos_passed else "yellow" if overall_score >= 0.6 else "red"
    console.print(Panel(
        f"[bold]Overall Agent Score: {overall_score:.1%}[/bold]",
        border_style=score_status,
    ))
    
    final_table = Table(title="Evaluation Summary", show_header=True)
    final_table.add_column("Component", style="dim")
    final_table.add_column("Score", justify="right")
    final_table.add_column("SLO", justify="right")
    final_table.add_column("Status", justify="center")
    
    def status_icon(score: float, threshold: float = 0.8) -> str:
        return "[green]✓ PASS[/green]" if score >= threshold else "[red]✗ FAIL[/red]"
    
    final_table.add_row(
        "Tool Correctness",
        f"{tool_summary.accuracy:.1%}",
        f"≥{SLO_TOOL_ACCURACY:.0%}",
        status_icon(tool_summary.accuracy, SLO_TOOL_ACCURACY),
    )
    final_table.add_row(
        "Router Mode Selection",
        f"{router_summary.mode_accuracy:.1%}",
        f"≥{SLO_ROUTER_ACCURACY:.0%}",
        status_icon(router_summary.mode_accuracy, SLO_ROUTER_ACCURACY),
    )
    final_table.add_row(
        "Router Company Extraction",
        f"{router_summary.company_extraction_accuracy:.1%}",
        f"≥{SLO_ROUTER_ACCURACY:.0%}",
        status_icon(router_summary.company_extraction_accuracy, SLO_ROUTER_ACCURACY),
    )
    final_table.add_row(
        "Router Intent Accuracy",
        f"{router_summary.intent_accuracy:.1%}",
        "≥80%",
        status_icon(router_summary.intent_accuracy),
    )
    
    if e2e_summary:
        final_table.add_row(
            "E2E Answer Relevance",
            f"{e2e_summary.answer_relevance_rate:.1%}",
            f"≥{SLO_ANSWER_RELEVANCE:.0%}",
            status_icon(e2e_summary.answer_relevance_rate, SLO_ANSWER_RELEVANCE),
        )
        final_table.add_row(
            "E2E Groundedness",
            f"{e2e_summary.groundedness_rate:.1%}",
            f"≥{SLO_GROUNDEDNESS:.0%}",
            status_icon(e2e_summary.groundedness_rate, SLO_GROUNDEDNESS),
        )
        final_table.add_row(
            "E2E Tool Selection",
            f"{e2e_summary.tool_selection_accuracy:.1%}",
            "≥80%",
            status_icon(e2e_summary.tool_selection_accuracy),
        )
        final_table.add_row(
            "E2E P95 Latency",
            f"{e2e_summary.p95_latency_ms:.0f}ms",
            f"≤{SLO_LATENCY_P95_MS}ms",
            "[green]✓ PASS[/green]" if e2e_summary.latency_slo_pass else "[red]✗ FAIL[/red]",
        )
    
    final_table.add_row("─" * 25, "─" * 8, "─" * 6, "─" * 10)
    final_table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{overall_score:.1%}[/bold]",
        "≥80%",
        status_icon(overall_score),
    )
    
    console.print(final_table)
    
    # SLO Summary
    if not all_slos_passed:
        console.print("\n[red bold]SLO Failures:[/red bold]")
        for failure in failed_slos:
            console.print(f"  • {failure}")
    
    # Regression check
    if baseline_score is not None:
        if is_regression:
            console.print(f"\n[red bold]⚠ REGRESSION DETECTED[/red bold]")
            console.print(f"  Baseline: {baseline_score:.1%}, Current: {overall_score:.1%}")
        else:
            diff = overall_score - baseline_score
            diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
            console.print(f"\n[dim]Baseline comparison: {baseline_score:.1%} → {overall_score:.1%} ({diff_str})[/dim]")
    
    # Save results
    if save_results:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
        console.print(f"\n[dim]Results saved to {OUTPUT_PATH}[/dim]")
    
    return summary


# =============================================================================
# CLI
# =============================================================================

app = typer.Typer()


@app.command()
def main(
    skip_e2e: bool = typer.Option(False, "--skip-e2e", help="Skip E2E tests (faster)"),
    e2e_limit: int | None = typer.Option(None, "--e2e-limit", help="Limit E2E tests"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_save: bool = typer.Option(False, "--no-save", help="Don't save results"),
    baseline: str | None = typer.Option(None, "--baseline", help="Path to baseline JSON for regression comparison"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Set current results as new baseline"),
):
    """Run complete agent evaluation suite."""
    baseline_path = Path(baseline) if baseline else None
    
    summary = run_all_evals(
        skip_e2e=skip_e2e,
        e2e_limit=e2e_limit,
        verbose=verbose,
        save_results=not no_save,
        baseline_path=baseline_path,
    )
    
    # Optionally set this run as the new baseline
    if set_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
        console.print(f"\n[green]✓ Set as new baseline: {BASELINE_PATH}[/green]")
    
    # Exit code based on results
    exit_code = 0
    
    if summary.overall_score < 0.8:
        console.print("\n[red bold]FAIL: Overall score below 80%[/red bold]")
        exit_code = 1
    
    if not summary.all_slos_passed:
        console.print("\n[red bold]FAIL: One or more SLOs not met[/red bold]")
        exit_code = 1
    
    if summary.regression_detected:
        console.print("\n[red bold]FAIL: Regression detected compared to baseline[/red bold]")
        exit_code = 1
    
    if exit_code == 0:
        console.print("\n[green bold]✓ PASS: Agent evaluation successful[/green bold]")
    
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
