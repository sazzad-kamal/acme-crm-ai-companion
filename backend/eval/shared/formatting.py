"""
Formatting utilities for eval output.

Rich console formatting helpers for tables and styled output.
"""

from rich.console import Console
from rich.table import Table

# Shared console instance
console = Console()


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


def print_error(label: str, message: str, indent: int = 2) -> None:
    """Print labeled error in red."""
    console.print(f"{' ' * indent}[red]{label}[/red]: {message}")


def print_warning(message: str, indent: int = 2) -> None:
    """Print warning message in yellow."""
    console.print(f"{' ' * indent}[yellow]{message}[/yellow]")


def print_status(passed: bool, details: str = "") -> None:
    """Print pass/fail status."""
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"  {status} {details}")


def print_case(num: int, total: int, text: str, difficulty: int) -> None:
    """Print case header."""
    console.print(f"\n[bold]Case {num}/{total}:[/bold] {text} [dim](d={difficulty})[/dim]")


def print_section_header(text: str) -> None:
    """Print bold red section header."""
    console.print(f"\n[bold red]{text}[/bold red]\n")


def print_failed_case(num: int, text: str, difficulty: int) -> None:
    """Print failed case header."""
    console.print(f"[bold cyan]{num}. {text}[/bold cyan] [dim](d={difficulty})[/dim]")


def print_dim(text: str, indent: int = 0) -> None:
    """Print dim text."""
    console.print(f"{' ' * indent}[dim]{text}[/dim]")


def build_eval_table(
    title: str,
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]],
    aggregate_row: tuple[str, str, str, bool] | None = None,
) -> Table:
    """
    Build a standardized evaluation summary table.

    Args:
        title: Table title
        sections: List of (section_name, rows) where rows are
                  (label, value, slo_target, slo_passed)
                  slo_target=None means "tracked", slo_passed=None means no status
        aggregate_row: Optional (label, value, slo_target, slo_passed) for bottom row with border

    Returns:
        Rich Table
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("SLO", justify="right", style="dim")

    for section_name, rows in sections:
        if section_name:
            # Color section header based on whether all SLOs in section pass
            section_passed = all(slo_passed for _, _, _, slo_passed in rows if slo_passed is not None)
            color = "green" if section_passed else "red"
            table.add_row(f"[bold {color}]{section_name}[/bold {color}]", "", "")

        for label, value, slo_target, _slo_passed in rows:
            slo_display = slo_target if slo_target else ""
            table.add_row(label, value, slo_display)

        table.add_row("", "", "")  # Spacer

    # Add aggregate row with separator if provided
    if aggregate_row:
        label, value, slo_target, slo_passed = aggregate_row
        color = "green" if slo_passed else "red"
        table.add_row("─" * 20, "─" * 10, "─" * 10, end_section=True)
        table.add_row(
            f"[bold {color}]{label}[/bold {color}]",
            f"[bold {color}]{value}[/bold {color}]",
            slo_target,
        )

    return table


__all__ = [
    "console",
    "format_percentage",
    "print_error",
    "print_warning",
    "print_status",
    "print_case",
    "print_section_header",
    "print_failed_case",
    "print_dim",
    "build_eval_table",
]
