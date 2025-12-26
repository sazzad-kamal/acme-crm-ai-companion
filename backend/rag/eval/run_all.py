"""
Combined RAG evaluation runner.

Runs all RAG evaluation suites:
1. Docs evaluation (public documentation RAG)
2. Account evaluation (private account-scoped RAG with privacy checks)

SLOs (Service Level Objectives):
- Context relevance ≥ 80%
- Answer relevance ≥ 80%
- Groundedness ≥ 80%
- RAG triad success ≥ 70%
- Privacy leakage = 0%
- P95 latency ≤ 5000ms

Usage:
    python -m backend.rag.eval.run_all
    python -m backend.rag.eval.run_all --skip-account  # Skip account eval (faster)
    python -m backend.rag.eval.run_all --verbose
    python -m backend.rag.eval.run_all --set-baseline  # Set current as baseline
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

from backend.rag.eval.docs_eval import run_evaluation as run_docs_eval, print_summary as print_docs_summary
from backend.rag.eval.account_eval import run_evaluation as run_account_eval, print_summary as print_account_summary
from backend.rag.eval.models import (
    RAGEvalSummary,
    SLO_CONTEXT_RELEVANCE,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
    SLO_RAG_TRIAD,
    SLO_PRIVACY_LEAKAGE,
    SLO_LATENCY_P95_MS,
)


console = Console()

OUTPUT_PATH = Path("data/processed/rag_eval_results.json")
BASELINE_PATH = Path("data/processed/rag_eval_baseline.json")


def compute_overall_score(
    docs_triad: float,
    account_triad: float,
    account_privacy: float,
) -> float:
    """
    Compute weighted overall score.
    
    Weights:
    - Docs RAG triad: 40%
    - Account RAG triad: 40%
    - Privacy (inverted): 20%
    """
    docs_score = docs_triad * 0.40
    account_score = account_triad * 0.40
    # Privacy: 0% leakage = 1.0, any leakage = 0.0
    privacy_score = (1.0 if account_privacy == 0 else 0.0) * 0.20
    
    return docs_score + account_score + privacy_score


def compare_to_baseline(
    summary: RAGEvalSummary,
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


def run_all_evals(
    skip_account: bool = False,
    verbose: bool = False,
    save_results: bool = True,
    baseline_path: Path | None = None,
) -> RAGEvalSummary:
    """
    Run all RAG evaluation suites.
    
    Args:
        skip_account: Skip account evaluation (expensive)
        verbose: Print detailed progress
        save_results: Save results to JSON
        baseline_path: Path to baseline JSON for regression comparison
        
    Returns:
        Complete RAGEvalSummary
    """
    console.print(Panel(
        "[bold]RAG Evaluation Suite[/bold]\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="blue",
    ))
    
    # 1. Docs Evaluation
    console.print("\n[bold cyan]1/2 Docs RAG Evaluation[/bold cyan]")
    docs_results, docs_summary = run_docs_eval(verbose=verbose)
    print_docs_summary(docs_results, docs_summary)
    
    # 2. Account Evaluation
    account_summary = None
    if not skip_account:
        console.print("\n[bold cyan]2/2 Account RAG Evaluation[/bold cyan]")
        account_results, account_summary = run_account_eval(verbose=verbose)
        print_account_summary(account_results, account_summary)
    else:
        console.print("\n[dim]2/2 Account RAG Evaluation - SKIPPED[/dim]")
    
    # Compute overall score
    overall_score = compute_overall_score(
        docs_triad=docs_summary.rag_triad_success,
        account_triad=account_summary.rag_triad_success if account_summary else 0.7,
        account_privacy=account_summary.privacy_leakage_rate if account_summary else 0.0,
    )
    
    # Combine failed SLOs
    all_failed_slos = list(docs_summary.failed_slos)
    if account_summary:
        all_failed_slos.extend(account_summary.failed_slos)
    
    all_slos_passed = len(all_failed_slos) == 0
    
    # Build summary
    summary = RAGEvalSummary(
        docs_eval=docs_summary,
        account_eval=account_summary,
        overall_score=overall_score,
        all_slos_passed=all_slos_passed,
        failed_slos=all_failed_slos,
    )
    
    # Check for regression
    is_regression, baseline_score = compare_to_baseline(summary, baseline_path)
    summary.regression_detected = is_regression
    summary.baseline_score = baseline_score
    
    # Print final summary
    console.print("\n")
    score_status = "green" if overall_score >= 0.8 and all_slos_passed else "yellow" if overall_score >= 0.6 else "red"
    console.print(Panel(
        f"[bold]Overall RAG Score: {overall_score:.1%}[/bold]",
        border_style=score_status,
    ))
    
    final_table = Table(title="RAG Evaluation Summary", show_header=True)
    final_table.add_column("Component", style="dim")
    final_table.add_column("Score", justify="right")
    final_table.add_column("SLO", justify="right")
    final_table.add_column("Status", justify="center")
    
    def status_icon(score: float, threshold: float, higher_better: bool = True) -> str:
        if higher_better:
            passed = score >= threshold
        else:
            passed = score <= threshold
        return "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
    
    # Docs metrics
    final_table.add_row(
        "Docs Context Relevance",
        f"{docs_summary.context_relevance:.1%}",
        f"≥{SLO_CONTEXT_RELEVANCE:.0%}",
        status_icon(docs_summary.context_relevance, SLO_CONTEXT_RELEVANCE),
    )
    final_table.add_row(
        "Docs RAG Triad",
        f"{docs_summary.rag_triad_success:.1%}",
        f"≥{SLO_RAG_TRIAD:.0%}",
        status_icon(docs_summary.rag_triad_success, SLO_RAG_TRIAD),
    )
    final_table.add_row(
        "Docs P95 Latency",
        f"{docs_summary.p95_latency_ms:.0f}ms",
        f"≤{SLO_LATENCY_P95_MS}ms",
        status_icon(docs_summary.p95_latency_ms, SLO_LATENCY_P95_MS, higher_better=False),
    )
    
    if account_summary:
        final_table.add_row("─" * 22, "─" * 8, "─" * 6, "─" * 10)
        final_table.add_row(
            "Account RAG Triad",
            f"{account_summary.rag_triad_success:.1%}",
            f"≥{SLO_RAG_TRIAD:.0%}",
            status_icon(account_summary.rag_triad_success, SLO_RAG_TRIAD),
        )
        final_table.add_row(
            "Account Privacy Leakage",
            f"{account_summary.privacy_leakage_rate:.1%}",
            f"={SLO_PRIVACY_LEAKAGE:.0%}",
            status_icon(account_summary.privacy_leakage_rate, SLO_PRIVACY_LEAKAGE, higher_better=False),
        )
        final_table.add_row(
            "Account P95 Latency",
            f"{account_summary.p95_latency_ms:.0f}ms",
            f"≤{SLO_LATENCY_P95_MS}ms",
            status_icon(account_summary.p95_latency_ms, SLO_LATENCY_P95_MS, higher_better=False),
        )
    
    final_table.add_row("─" * 22, "─" * 8, "─" * 6, "─" * 10)
    final_table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{overall_score:.1%}[/bold]",
        "≥80%",
        status_icon(overall_score, 0.8),
    )
    
    console.print(final_table)
    
    # SLO Summary
    if not all_slos_passed:
        console.print("\n[red bold]SLO Failures:[/red bold]")
        for failure in all_failed_slos:
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
    skip_account: bool = typer.Option(False, "--skip-account", help="Skip account eval (faster)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_save: bool = typer.Option(False, "--no-save", help="Don't save results"),
    baseline: str | None = typer.Option(None, "--baseline", help="Path to baseline JSON for regression comparison"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Set current results as new baseline"),
):
    """Run complete RAG evaluation suite."""
    baseline_path = Path(baseline) if baseline else None
    
    summary = run_all_evals(
        skip_account=skip_account,
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
        console.print("\n[green bold]✓ PASS: RAG evaluation successful[/green bold]")
    
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
