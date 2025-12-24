"""
Base evaluation utilities shared between docs and account eval.

Provides:
- Rich console formatting helpers
- Common summary table rendering
- Shared metrics computation
"""

from typing import Callable, TypeVar
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track


# Type variable for result types
T = TypeVar("T")

# Shared console instance
console = Console()


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


def run_with_progress(
    items: list,
    evaluate_fn: Callable,
    description: str = "Evaluating...",
) -> list:
    """
    Run evaluation function on items with progress bar.
    
    Args:
        items: Items to evaluate
        evaluate_fn: Function to call for each item
        description: Progress bar description
        
    Returns:
        List of results
    """
    results = []
    for item in track(items, description=description):
        result = evaluate_fn(item)
        results.append(result)
    return results


def compute_aggregate_metrics(results: list, metrics: dict[str, Callable]) -> dict:
    """
    Compute aggregate metrics from results.
    
    Args:
        results: List of result objects
        metrics: Dict mapping metric name to extraction function
        
    Returns:
        Dict of computed averages
    """
    n = len(results)
    if n == 0:
        return {}
    
    return {
        name: sum(fn(r) for r in results) / n
        for name, fn in metrics.items()
    }


def format_check_mark(value: bool) -> str:
    """Format boolean as colored check/cross mark."""
    return "[green]✓[/green]" if value else "[red]✗[/red]"


def format_percentage(value: float, threshold: float = 0.0) -> str:
    """Format percentage with color based on threshold."""
    if value > threshold:
        return f"[red]{value:.1%}[/red]"
    return f"[green]{value:.1%}[/green]"


def add_separator_row(table: Table) -> None:
    """Add a visual separator row to a table."""
    table.add_row("─" * 20, "─" * 10)
