"""
CLI for question tree utilities.

Usage:
    python -m backend.agent.followup.tree validate
    python -m backend.agent.followup.tree tree [--depth N]
    python -m backend.agent.followup.tree stats
    python -m backend.agent.followup.tree paths [--limit N]
"""

from collections import defaultdict

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import get_all_paths, get_tree_stats, print_tree, validate_tree

app = typer.Typer(help="Question tree utilities for demo reliability.")
console = Console()


@app.command()
def validate() -> None:
    """Validate the question tree for consistency."""
    issues = validate_tree()

    if issues:
        rprint("[red]Validation failed![/red]")
        for issue in issues:
            rprint(f"  [red]x[/red] {issue}")
        raise typer.Exit(1)
    else:
        rprint("[green]OK[/green] Question tree is valid")


@app.command()
def tree(
    depth: int | None = typer.Option(None, "--depth", "-d", help="Max depth to display"),
) -> None:
    """Print the question tree in a top-down format."""
    tree_output = print_tree(max_depth=depth)
    console.print(tree_output)


@app.command()
def stats() -> None:
    """Show statistics about the question tree."""
    s = get_tree_stats()
    table = Table(title="Question Tree Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Starter questions", str(s["num_starters"]))
    table.add_row("Total questions", str(s["num_questions"]))
    table.add_row("Edges (follow-up links)", str(s["num_edges"]))
    table.add_row("Max depth", str(s["max_depth"]))
    table.add_row("Total paths", str(s["num_paths"]))
    table.add_row("Min path length", str(s["path_lengths"]["min"]))
    table.add_row("Max path length", str(s["path_lengths"]["max"]))

    console.print(table)


@app.command()
def paths(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths shown"),
) -> None:
    """List all conversation paths for auditing workflows."""
    all_paths = get_all_paths()

    # Group paths by depth
    paths_by_depth: dict[int, list] = defaultdict(list)
    for path in all_paths:
        paths_by_depth[len(path)].append(path)

    console.print(f"\n[bold]Question Tree[/bold] - {len(all_paths)} paths\n")

    path_num = 0
    for depth in sorted(paths_by_depth.keys()):
        console.print(f"[dim]-- Depth {depth} ({len(paths_by_depth[depth])} paths) --[/dim]\n")

        for path in paths_by_depth[depth]:
            path_num += 1
            if limit and path_num > limit:
                remaining = len(all_paths) - limit
                console.print(f"\n[dim]... and {remaining} more paths (use --limit to see more)[/dim]")
                return

            # Format path as numbered steps
            steps = []
            for i, question in enumerate(path, 1):
                steps.append(f"[cyan]{i}.[/cyan] {question}")

            path_content = "\n".join(steps)
            console.print(Panel(path_content, title=f"Path {path_num}", border_style="dim"))
            console.print()


if __name__ == "__main__":
    app()
