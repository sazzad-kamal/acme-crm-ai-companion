"""
RAG Evaluation Harness for Acme CRM docs.

Evaluates the RAG pipeline using:
- Context relevance: Did we retrieve the right documents?
- Answer relevance: Does the answer address the question?
- Groundedness: Is the answer grounded in the context (no hallucinations)?

Uses LLM-as-judge for relevance and groundedness scoring.

Usage:
    python -m backend.rag.eval.docs_eval
    python -m backend.rag.eval.docs_eval --verbose
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import typer
from rich.progress import track

from backend.rag.retrieval.base import create_backend
from backend.rag.pipeline.docs import answer_question
from backend.rag.eval.models import (
    EvalResult,
    DocsEvalSummary,
    SLO_CONTEXT_RELEVANCE,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
    SLO_RAG_TRIAD,
    SLO_DOC_RECALL,
    SLO_LATENCY_P95_MS,
)
from backend.rag.eval.judge import judge_response, compute_doc_recall
from backend.rag.eval.base import (
    console,
    create_summary_table,
    create_detail_table,
    print_eval_header,
    print_issues_panel,
    format_check_mark,
    add_separator_row,
)
from backend.rag.eval.tracking import print_full_tracking_report


# =============================================================================
# Configuration
# =============================================================================

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_eval_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    """Load evaluation questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def evaluate_question(
    question_data: dict,
    backend,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a single question through the RAG pipeline.
    
    Args:
        question_data: Dict with id, question, target_doc_ids
        backend: Initialized RetrievalBackend
        verbose: Print progress
        
    Returns:
        EvalResult with all metrics
    """
    question_id = question_data["id"]
    question = question_data["question"]
    target_doc_ids = question_data["target_doc_ids"]
    
    if verbose:
        print(f"\nEvaluating: {question_id}")
        print(f"  Question: {question[:60]}...")
    
    # Run RAG pipeline
    start_time = time.time()
    result = answer_question(question, backend, k=8, verbose=False)
    total_latency = (time.time() - start_time) * 1000
    
    # Build context string for judge
    context = "\n\n".join([
        f"[{c.doc_id}] {c.text[:500]}"
        for c in result["used_chunks"]
    ])
    
    # Compute doc recall
    doc_recall = compute_doc_recall(target_doc_ids, result["doc_ids_used"])
    
    if verbose:
        print(f"  Retrieved: {result['doc_ids_used']}")
        print(f"  Doc recall: {doc_recall:.2f}")
    
    # Judge the response
    judge_result = judge_response(
        question=question,
        context=context,
        answer=result["answer"],
        doc_ids=result["doc_ids_used"],
    )
    
    if verbose:
        print(f"  Judge: context={judge_result.context_relevance}, "
              f"answer={judge_result.answer_relevance}, "
              f"grounded={judge_result.groundedness}")
    
    # Extract step timings from pipeline result
    step_timings = {}
    for step in result.get("steps", []):
        step_id = step.get("id", "unknown")
        elapsed = step.get("elapsed_ms", 0)
        step_timings[step_id] = elapsed

    return EvalResult(
        question_id=question_id,
        question=question,
        target_doc_ids=target_doc_ids,
        retrieved_doc_ids=result["doc_ids_used"],
        answer=result["answer"],
        judge_result=judge_result,
        doc_recall=doc_recall,
        latency_ms=total_latency,
        total_tokens=result["metrics"]["total_tokens"],
        step_timings=step_timings,
    )


def run_evaluation(
    questions: list[dict] | None = None,
    verbose: bool = True,
) -> tuple[list[EvalResult], DocsEvalSummary]:
    """
    Run full evaluation over all questions.
    
    Args:
        questions: List of question dicts (or load from file)
        verbose: Print progress
        
    Returns:
        Tuple of (results list, summary)
    """
    if questions is None:
        questions = load_eval_questions()
    
    print_eval_header(
        "RAG Evaluation Harness",
        f"Evaluating [bold]{len(questions)}[/bold] questions",
    )
    
    # Initialize backend
    with console.status("[bold green]Loading backend..."):
        backend = create_backend()
    
    results = []
    for q in track(questions, description="Evaluating..."):
        result = evaluate_question(q, backend, verbose=verbose)
        results.append(result)
    
    # Compute summary
    summary = compute_summary(results)
    
    return results, summary


def compute_summary(results: list[EvalResult]) -> DocsEvalSummary:
    """Compute summary statistics and check SLOs."""
    n = len(results)
    
    if n == 0:
        return DocsEvalSummary(
            total_tests=0,
            context_relevance=0.0,
            answer_relevance=0.0,
            groundedness=0.0,
            rag_triad_success=0.0,
            avg_doc_recall=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_tokens=0,
            estimated_cost=0.0,
        )
    
    # Compute aggregates
    context_relevance = sum(r.judge_result.context_relevance for r in results) / n
    answer_relevance = sum(r.judge_result.answer_relevance for r in results) / n
    groundedness = sum(r.judge_result.groundedness for r in results) / n
    
    # RAG triad success (all three = 1)
    triad_success = sum(
        1 for r in results
        if r.judge_result.context_relevance == 1
        and r.judge_result.answer_relevance == 1
        and r.judge_result.groundedness == 1
    ) / n
    
    avg_doc_recall = sum(r.doc_recall for r in results) / n
    avg_latency = sum(r.latency_ms for r in results) / n
    
    # P95 latency
    latencies = sorted([r.latency_ms for r in results])
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[min(p95_index, len(latencies) - 1)]
    
    total_tokens = sum(r.total_tokens for r in results)
    estimated_cost = (total_tokens * 0.8 * 0.40 + total_tokens * 0.2 * 1.60) / 1_000_000
    
    # Check SLOs
    failed_slos = []
    if context_relevance < SLO_CONTEXT_RELEVANCE:
        failed_slos.append(f"Context relevance {context_relevance:.1%} < {SLO_CONTEXT_RELEVANCE:.1%}")
    if answer_relevance < SLO_ANSWER_RELEVANCE:
        failed_slos.append(f"Answer relevance {answer_relevance:.1%} < {SLO_ANSWER_RELEVANCE:.1%}")
    if groundedness < SLO_GROUNDEDNESS:
        failed_slos.append(f"Groundedness {groundedness:.1%} < {SLO_GROUNDEDNESS:.1%}")
    if triad_success < SLO_RAG_TRIAD:
        failed_slos.append(f"RAG triad {triad_success:.1%} < {SLO_RAG_TRIAD:.1%}")
    if avg_doc_recall < SLO_DOC_RECALL:
        failed_slos.append(f"Doc recall {avg_doc_recall:.1%} < {SLO_DOC_RECALL:.1%}")
    if p95_latency > SLO_LATENCY_P95_MS:
        failed_slos.append(f"P95 latency {p95_latency:.0f}ms > {SLO_LATENCY_P95_MS}ms")
    
    return DocsEvalSummary(
        total_tests=n,
        context_relevance=context_relevance,
        answer_relevance=answer_relevance,
        groundedness=groundedness,
        rag_triad_success=triad_success,
        avg_doc_recall=avg_doc_recall,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
        all_slos_passed=len(failed_slos) == 0,
        failed_slos=failed_slos,
    )


def print_summary(results: list[EvalResult], summary: DocsEvalSummary) -> None:
    """Print summary statistics from evaluation results using Rich."""
    n = len(results)
    
    if n == 0:
        console.print("[yellow]No results to summarize[/yellow]")
        return
    
    # Summary table using shared helper
    summary_table = create_summary_table()
    
    summary_table.add_row("Questions evaluated", str(summary.total_tests))
    
    ctx_style = "[green]" if summary.context_relevance >= SLO_CONTEXT_RELEVANCE else "[red]"
    summary_table.add_row("Context Relevance", f"{ctx_style}{summary.context_relevance:.1%}[/] (SLO: ≥{SLO_CONTEXT_RELEVANCE:.0%})")
    
    ans_style = "[green]" if summary.answer_relevance >= SLO_ANSWER_RELEVANCE else "[red]"
    summary_table.add_row("Answer Relevance", f"{ans_style}{summary.answer_relevance:.1%}[/] (SLO: ≥{SLO_ANSWER_RELEVANCE:.0%})")
    
    gnd_style = "[green]" if summary.groundedness >= SLO_GROUNDEDNESS else "[red]"
    summary_table.add_row("Groundedness", f"{gnd_style}{summary.groundedness:.1%}[/] (SLO: ≥{SLO_GROUNDEDNESS:.0%})")
    
    triad_style = "[green]" if summary.rag_triad_success >= SLO_RAG_TRIAD else "[red]"
    summary_table.add_row("RAG Triad Success", f"[bold {triad_style[1:-1]}]{summary.rag_triad_success:.1%}[/bold {triad_style[1:-1]}] (SLO: ≥{SLO_RAG_TRIAD:.0%})")
    
    recall_style = "[green]" if summary.avg_doc_recall >= SLO_DOC_RECALL else "[yellow]"
    summary_table.add_row("Avg Doc Recall", f"{recall_style}{summary.avg_doc_recall:.1%}[/] (SLO: ≥{SLO_DOC_RECALL:.0%})")
    
    add_separator_row(summary_table)
    summary_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    
    p95_style = "[green]" if summary.p95_latency_ms <= SLO_LATENCY_P95_MS else "[red]"
    summary_table.add_row("P95 Latency", f"{p95_style}{summary.p95_latency_ms:.0f}ms[/] (SLO: ≤{SLO_LATENCY_P95_MS}ms)")
    
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Est. Cost", f"${summary.estimated_cost:.4f}")
    
    # SLO status
    add_separator_row(summary_table)
    if summary.all_slos_passed:
        summary_table.add_row("SLO Status", "[bold green]✓ ALL PASSED[/bold green]")
    else:
        summary_table.add_row("SLO Status", f"[bold red]✗ {len(summary.failed_slos)} FAILED[/bold red]")
    
    console.print(summary_table)
    
    # Per-question results using shared helper
    detail_table = create_detail_table("Per-Question Results", [
        ("ID", "left"),
        ("Ctx", "center"),
        ("Ans", "center"),
        ("Gnd", "center"),
        ("Recall", "right"),
        ("Latency", "right"),
    ])
    
    for r in results:
        detail_table.add_row(
            r.question_id,
            format_check_mark(r.judge_result.context_relevance == 1),
            format_check_mark(r.judge_result.answer_relevance == 1),
            format_check_mark(r.judge_result.groundedness == 1),
            f"{r.doc_recall:.1%}",
            f"{r.latency_ms:.0f}ms",
        )
    
    console.print(detail_table)
    
    # Failed questions using shared helper
    failed = [r for r in results if r.judge_result.groundedness == 0 or r.judge_result.answer_relevance == 0]
    print_issues_panel(
        "Questions Needing Attention",
        [f"[bold]{r.question_id}[/bold]: {r.question}\n  [dim]{r.judge_result.explanation}[/dim]" for r in failed],
    )


# =============================================================================
# CLI Entrypoint
# =============================================================================

app = typer.Typer(help="RAG Evaluation Harness")


@app.command()
def run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    output: Path = typer.Option(
        Path("data/processed/eval_results.json"),
        "--output", "-o",
        help="Output file for results",
    ),
) -> None:
    """Run evaluation on all test questions."""
    results, summary = run_evaluation(verbose=verbose)
    print_summary(results, summary)
    
    # Per-question detail table
    detail_table = create_detail_table("Per-Question Results", [
        ("ID", "left"),
        ("Ctx", "center"),
        ("Ans", "center"),
        ("Gnd", "center"),
        ("Recall", "right"),
        ("Latency", "right"),
    ])
    
    for r in results:
        detail_table.add_row(
            r.question_id,
            format_check_mark(r.judge_result.context_relevance == 1),
            format_check_mark(r.judge_result.answer_relevance == 1),
            format_check_mark(r.judge_result.groundedness == 1),
            f"{r.doc_recall:.1%}",
            f"{r.latency_ms:.0f}ms",
        )
    
    console.print(detail_table)
    
    # Failed questions
    failed = [r for r in results if r.judge_result.groundedness == 0 or r.judge_result.answer_relevance == 0]
    print_issues_panel(
        "Questions Needing Attention",
        [f"[bold]{r.question_id}[/bold]: {r.question}\n  [dim]{r.judge_result.explanation}[/dim]" for r in failed],
    )
    
    # Save results to file
    output.parent.mkdir(parents=True, exist_ok=True)
    results_data = {
        "results": [r.model_dump() for r in results],
        "summary": summary.model_dump(),
    }
    with open(output, "w") as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output}[/dim]")

    # Print tracking report (regression detection + budget analysis)
    print_full_tracking_report(results, summary)

    # Exit code based on SLOs
    if not summary.all_slos_passed:
        console.print("\n[red bold]FAIL: One or more SLOs not met[/red bold]")
        for slo in summary.failed_slos:
            console.print(f"  • {slo}")
        raise typer.Exit(code=1)
    else:
        console.print("\n[green bold]✓ PASS: All SLOs met[/green bold]")


def main() -> None:
    """Main entrypoint for evaluation."""
    app()


if __name__ == "__main__":
    main()
