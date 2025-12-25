"""
Combined agent evaluation runner.

Runs all agent evaluation suites:
1. Tool evaluation (correctness)
2. Router evaluation (mode selection)
3. End-to-end evaluation (full pipeline)

Usage:
    python -m backend.agent.eval.run_all
    python -m backend.agent.eval.run_all --skip-e2e  # Skip expensive E2E tests
    python -m backend.agent.eval.run_all --verbose
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
from backend.agent.eval.models import AgentEvalSummary


console = Console()

OUTPUT_PATH = Path("data/processed/agent_eval_results.json")


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
    e2e_limit: Optional[int] = None,
    verbose: bool = False,
    save_results: bool = True,
) -> AgentEvalSummary:
    """
    Run all agent evaluation suites.
    
    Args:
        skip_e2e: Skip end-to-end evaluation (expensive)
        e2e_limit: Limit E2E tests
        verbose: Print detailed progress
        save_results: Save results to JSON
        
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
    
    summary = AgentEvalSummary(
        tool_eval=tool_summary,
        router_eval=router_summary,
        e2e_eval=e2e_summary,
        overall_score=overall_score,
    )
    
    # Print final summary
    console.print("\n")
    console.print(Panel(
        f"[bold]Overall Agent Score: {overall_score:.1%}[/bold]",
        border_style="green" if overall_score >= 0.8 else "yellow" if overall_score >= 0.6 else "red",
    ))
    
    final_table = Table(title="Evaluation Summary", show_header=True)
    final_table.add_column("Component", style="dim")
    final_table.add_column("Score", justify="right")
    final_table.add_column("Status", justify="center")
    
    def status_icon(score: float, threshold: float = 0.8) -> str:
        return "[green]✓ PASS[/green]" if score >= threshold else "[red]✗ FAIL[/red]"
    
    final_table.add_row(
        "Tool Correctness",
        f"{tool_summary.accuracy:.1%}",
        status_icon(tool_summary.accuracy),
    )
    final_table.add_row(
        "Router Mode Selection",
        f"{router_summary.mode_accuracy:.1%}",
        status_icon(router_summary.mode_accuracy),
    )
    final_table.add_row(
        "Router Company Extraction",
        f"{router_summary.company_extraction_accuracy:.1%}",
        status_icon(router_summary.company_extraction_accuracy),
    )
    
    if e2e_summary:
        final_table.add_row(
            "E2E Answer Relevance",
            f"{e2e_summary.answer_relevance_rate:.1%}",
            status_icon(e2e_summary.answer_relevance_rate),
        )
        final_table.add_row(
            "E2E Groundedness",
            f"{e2e_summary.groundedness_rate:.1%}",
            status_icon(e2e_summary.groundedness_rate),
        )
        final_table.add_row(
            "E2E Tool Selection",
            f"{e2e_summary.tool_selection_accuracy:.1%}",
            status_icon(e2e_summary.tool_selection_accuracy),
        )
    
    final_table.add_row("─" * 25, "─" * 8, "─" * 10)
    final_table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{overall_score:.1%}[/bold]",
        status_icon(overall_score),
    )
    
    console.print(final_table)
    
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
    e2e_limit: Optional[int] = typer.Option(None, "--e2e-limit", help="Limit E2E tests"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_save: bool = typer.Option(False, "--no-save", help="Don't save results"),
):
    """Run complete agent evaluation suite."""
    summary = run_all_evals(
        skip_e2e=skip_e2e,
        e2e_limit=e2e_limit,
        verbose=verbose,
        save_results=not no_save,
    )
    
    # Exit code based on overall score
    if summary.overall_score < 0.8:
        console.print("\n[red bold]FAIL: Overall score below 80%[/red bold]")
        raise typer.Exit(code=1)
    else:
        console.print("\n[green bold]PASS: Agent evaluation successful[/green bold]")


if __name__ == "__main__":
    app()
