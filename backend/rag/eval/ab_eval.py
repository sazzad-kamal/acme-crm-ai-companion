"""
A/B Evaluation for RAG Pipeline Configuration.

Runs the same evaluation questions with different pipeline configurations
to quantify the quality vs latency tradeoffs of HyDE, query rewrite, and reranking.

Usage:
    python -m backend.rag.eval.ab_eval
    python -m backend.rag.eval.ab_eval --limit 5
"""

import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from backend.rag.retrieval.base import create_backend
from backend.rag.pipeline.docs import answer_question
from backend.rag.eval.docs_eval import load_eval_questions
from backend.rag.eval.judge import judge_response


console = Console()
app = typer.Typer(help="A/B Evaluation for pipeline configurations")


# =============================================================================
# Configuration Matrix (2x2x2 = 8 configs)
# =============================================================================

AB_CONFIGS: dict[str, dict[str, bool]] = {
    # All on (quality)
    "full_quality": {"use_hyde": True, "use_rewrite": True, "use_reranker": True},

    # Single feature off
    "no_reranker": {"use_hyde": True, "use_rewrite": True, "use_reranker": False},
    "no_hyde": {"use_hyde": False, "use_rewrite": True, "use_reranker": True},
    "no_rewrite": {"use_hyde": True, "use_rewrite": False, "use_reranker": True},

    # Two features off
    "hyde_only": {"use_hyde": True, "use_rewrite": False, "use_reranker": False},
    "rewrite_only": {"use_hyde": False, "use_rewrite": True, "use_reranker": False},
    "reranker_only": {"use_hyde": False, "use_rewrite": False, "use_reranker": True},

    # All off (fast)
    "fast": {"use_hyde": False, "use_rewrite": False, "use_reranker": False},
}


# =============================================================================
# A/B Evaluation
# =============================================================================

def run_config_eval(
    config_name: str,
    config: dict[str, bool],
    questions: list[dict],
    backend,
) -> dict[str, Any]:
    """
    Run evaluation with a specific configuration.

    Returns:
        Dict with metrics for this configuration
    """
    results = []

    for q in questions:
        question = q["question"]
        target_doc_ids = q["target_doc_ids"]

        start_time = time.time()

        try:
            result = answer_question(
                question,
                backend,
                k=8,
                use_hyde=config["use_hyde"],
                use_rewrite=config["use_rewrite"],
                verbose=False,
            )
            latency = (time.time() - start_time) * 1000

            # Build context for judge
            context = "\n\n".join([
                f"[{c.doc_id}] {c.text[:500]}"
                for c in result["used_chunks"]
            ])

            # Judge the response
            judge_result = judge_response(
                question=question,
                context=context,
                answer=result["answer"],
                doc_ids=result["doc_ids_used"],
            )

            # Compute doc recall
            target_set = set(target_doc_ids)
            retrieved_set = set(result["doc_ids_used"])
            doc_recall = len(target_set & retrieved_set) / len(target_set) if target_set else 1.0

            results.append({
                "latency_ms": latency,
                "context_relevance": judge_result.context_relevance,
                "answer_relevance": judge_result.answer_relevance,
                "groundedness": judge_result.groundedness,
                "doc_recall": doc_recall,
                "rag_triad": int(
                    judge_result.context_relevance == 1 and
                    judge_result.answer_relevance == 1 and
                    judge_result.groundedness == 1
                ),
            })

        except Exception as e:
            console.print(f"[red]Error on question: {e}[/red]")
            results.append({
                "latency_ms": (time.time() - start_time) * 1000,
                "context_relevance": 0,
                "answer_relevance": 0,
                "groundedness": 0,
                "doc_recall": 0,
                "rag_triad": 0,
            })

    # Aggregate metrics
    n = len(results)
    latencies = sorted([r["latency_ms"] for r in results])
    p95_idx = int(n * 0.95) if n > 0 else 0

    return {
        "config_name": config_name,
        "use_hyde": config["use_hyde"],
        "use_rewrite": config["use_rewrite"],
        "use_reranker": config["use_reranker"],
        "num_questions": n,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n if n else 0,
        "p95_latency_ms": latencies[min(p95_idx, n - 1)] if n else 0,
        "context_relevance": sum(r["context_relevance"] for r in results) / n if n else 0,
        "answer_relevance": sum(r["answer_relevance"] for r in results) / n if n else 0,
        "groundedness": sum(r["groundedness"] for r in results) / n if n else 0,
        "rag_triad": sum(r["rag_triad"] for r in results) / n if n else 0,
        "doc_recall": sum(r["doc_recall"] for r in results) / n if n else 0,
    }


def run_ab_evaluation(
    limit: int | None = None,
    configs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Run A/B evaluation across all configurations.

    Args:
        limit: Limit number of questions per config
        configs: Specific configs to run (default: all)

    Returns:
        List of results per configuration
    """
    console.print(Panel(
        "Running A/B comparison across pipeline configurations",
        title="[bold blue]A/B Evaluation[/bold blue]",
        border_style="blue",
    ))

    # Load questions
    questions = load_eval_questions()
    if limit:
        questions = questions[:limit]

    console.print(f"Questions per config: {len(questions)}")

    # Initialize backend once
    with console.status("[bold green]Loading backend..."):
        backend = create_backend()

    # Select configs
    configs_to_run = configs or list(AB_CONFIGS.keys())

    all_results = []

    for config_name in track(configs_to_run, description="Running configs..."):
        config = AB_CONFIGS[config_name]
        console.print(f"\n[cyan]Config: {config_name}[/cyan]")
        console.print(f"  HyDE={config['use_hyde']}, Rewrite={config['use_rewrite']}, Reranker={config['use_reranker']}")

        result = run_config_eval(config_name, config, questions, backend)
        all_results.append(result)

        console.print(f"  RAG Triad: {result['rag_triad']:.1%}, Latency: {result['avg_latency_ms']:.0f}ms")

    return all_results


def print_ab_results(results: list[dict[str, Any]]) -> None:
    """Print A/B comparison results."""
    # Sort by RAG triad (quality)
    results_sorted = sorted(results, key=lambda x: x["rag_triad"], reverse=True)

    # Main comparison table
    table = Table(title="A/B Configuration Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Config", style="bold")
    table.add_column("HyDE", justify="center")
    table.add_column("Rewrite", justify="center")
    table.add_column("Rerank", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("RAG Triad", justify="right")
    table.add_column("Recall", justify="right")

    for r in results_sorted:
        hyde = "[green]✓[/green]" if r["use_hyde"] else "[red]✗[/red]"
        rewrite = "[green]✓[/green]" if r["use_rewrite"] else "[red]✗[/red]"
        reranker = "[green]✓[/green]" if r["use_reranker"] else "[red]✗[/red]"

        # Color code quality
        triad_color = "green" if r["rag_triad"] >= 0.8 else "yellow" if r["rag_triad"] >= 0.6 else "red"
        recall_color = "green" if r["doc_recall"] >= 0.7 else "yellow" if r["doc_recall"] >= 0.5 else "red"

        table.add_row(
            r["config_name"],
            hyde,
            rewrite,
            reranker,
            f"{r['avg_latency_ms']:.0f}ms",
            f"[{triad_color}]{r['rag_triad']:.1%}[/{triad_color}]",
            f"[{recall_color}]{r['doc_recall']:.1%}[/{recall_color}]",
        )

    console.print(table)

    # Feature impact analysis
    console.print("\n[bold]Feature Impact Analysis[/bold]")

    # Find baseline (full_quality)
    baseline = next((r for r in results if r["config_name"] == "full_quality"), results[0])

    impact_table = Table(show_header=True, header_style="bold")
    impact_table.add_column("Feature Disabled")
    impact_table.add_column("Quality Impact", justify="right")
    impact_table.add_column("Latency Saved", justify="right")
    impact_table.add_column("Efficiency", justify="right")

    for r in results:
        if r["config_name"] in ["no_hyde", "no_rewrite", "no_reranker"]:
            quality_delta = r["rag_triad"] - baseline["rag_triad"]
            latency_delta = baseline["avg_latency_ms"] - r["avg_latency_ms"]

            # Efficiency = latency saved per % quality lost
            efficiency = latency_delta / abs(quality_delta * 100) if quality_delta != 0 else 0

            feature = r["config_name"].replace("no_", "").upper()

            impact_table.add_row(
                feature,
                f"[red]{quality_delta:+.1%}[/red]" if quality_delta < 0 else f"[green]{quality_delta:+.1%}[/green]",
                f"[green]{latency_delta:+.0f}ms[/green]" if latency_delta > 0 else f"{latency_delta:.0f}ms",
                f"{efficiency:.1f}ms/%",
            )

    console.print(impact_table)

    # Recommendation
    console.print("\n[bold]Recommendation[/bold]")

    # Find most efficient feature to disable
    feature_impacts = []
    for r in results:
        if r["config_name"] in ["no_hyde", "no_rewrite", "no_reranker"]:
            quality_delta = abs(baseline["rag_triad"] - r["rag_triad"])
            latency_delta = baseline["avg_latency_ms"] - r["avg_latency_ms"]
            if quality_delta > 0 and latency_delta > 0:
                efficiency = latency_delta / quality_delta
                feature_impacts.append((r["config_name"].replace("no_", ""), efficiency, quality_delta, latency_delta))

    if feature_impacts:
        # Sort by efficiency (latency saved per quality lost)
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        best = feature_impacts[0]
        console.print(f"If you need to reduce latency, disable [bold]{best[0].upper()}[/bold] first:")
        console.print(f"  • Saves {best[3]:.0f}ms latency")
        console.print(f"  • Costs {best[2]:.1%} quality")


def save_ab_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save A/B results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


# =============================================================================
# CLI
# =============================================================================

@app.command()
def run(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit questions per config"),
    output: Path = typer.Option(
        Path("backend/data/processed/ab_eval_results.json"),
        "--output", "-o",
        help="Output file for results",
    ),
    configs: str | None = typer.Option(
        None,
        "--configs", "-c",
        help="Comma-separated configs to run (default: all)",
    ),
):
    """Run A/B evaluation across pipeline configurations."""
    config_list = configs.split(",") if configs else None

    results = run_ab_evaluation(limit=limit, configs=config_list)
    print_ab_results(results)
    save_ab_results(results, output)


@app.command()
def quick():
    """Quick A/B test with just 3 questions and 4 main configs."""
    results = run_ab_evaluation(
        limit=3,
        configs=["full_quality", "no_reranker", "no_hyde", "fast"],
    )
    print_ab_results(results)


if __name__ == "__main__":
    app()
