"""Evaluation runner - tests conversation paths."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

# Fix Windows asyncio cleanup issues with httpx/RAGAS
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined,unused-ignore]

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from backend.eval.answer.text.ragas import evaluate_single
from backend.eval.integration.langsmith import get_latency_percentages
from backend.eval.integration.models import (
    SLO_FLOW_ANSWER_CORRECTNESS,
    SLO_FLOW_AVG_LATENCY_MS,
    SLO_FLOW_FAITHFULNESS,
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    FlowEvalResults,
    FlowResult,
    FlowStepResult,
)
from backend.eval.integration.tree import get_all_paths, get_expected_answer, get_tree_stats

_console = Console()
logger = logging.getLogger(__name__)

# Constants
MIN_ANSWER_LENGTH = 10
AGENT_TIMEOUT_SECONDS = 120
RAGAS_TIMEOUT_SECONDS = 180


# =============================================================================
# SLO Definitions
# =============================================================================


class SloSpec(NamedTuple):
    """Single SLO metric specification."""

    key: str
    label: str
    section: str
    get_value: Callable[[FlowEvalResults], float]
    target: float
    compare: str
    fmt: str


SLO_SPECS: list[SloSpec] = [
    SloSpec("path_pass_rate", "Path Pass Rate", "Pass Rates",
            lambda r: r.path_pass_rate, SLO_FLOW_PATH_PASS_RATE, ">=", "pct"),
    SloSpec("question_pass_rate", "Question Pass Rate", "Pass Rates",
            lambda r: r.question_pass_rate, SLO_FLOW_QUESTION_PASS_RATE, ">=", "pct"),
    SloSpec("relevance", "Relevance", "Answer Quality",
            lambda r: r.avg_relevance, SLO_FLOW_RELEVANCE, ">=", "pct"),
    SloSpec("faithfulness", "Faithfulness", "Answer Quality",
            lambda r: r.avg_faithfulness, SLO_FLOW_FAITHFULNESS, ">=", "pct"),
    SloSpec("answer_correctness", "Answer Correctness", "Answer Quality",
            lambda r: r.avg_answer_correctness, SLO_FLOW_ANSWER_CORRECTNESS, ">=", "pct"),
    SloSpec("avg_latency_ms", "Avg Latency/Question", "Latency",
            lambda r: r.avg_latency_per_question_ms, SLO_FLOW_AVG_LATENCY_MS, "<=", "ms"),
]


def _slo_passed(spec: SloSpec, results: FlowEvalResults) -> bool:
    """Check if an SLO spec passes for the given results."""
    value = spec.get_value(results)
    return value >= spec.target if spec.compare == ">=" else value <= spec.target


def _format_slo(spec: SloSpec, value: float) -> tuple[str, str]:
    """Format a value and its SLO target for display."""
    if spec.fmt == "pct":
        return f"{value:.1%}", f"{spec.compare}{spec.target:.1%}"
    return f"{value:.0f}ms", f"{spec.compare}{spec.target:.0f}ms"


def _count_slo_failures(step: FlowStepResult) -> int:
    """Count how many SLO metrics failed for a step."""
    count = 0
    if step.relevance_score < SLO_FLOW_RELEVANCE:
        count += 1
    if step.faithfulness_score < SLO_FLOW_FAITHFULNESS:
        count += 1
    if step.answer_correctness_score < SLO_FLOW_ANSWER_CORRECTNESS:
        count += 1
    return count


def _print_slo_failures(results: FlowEvalResults) -> None:
    """Print details of SLO failures."""
    failures: list[tuple[int, FlowStepResult]] = []
    for flow_result in results.all_results:
        for step in flow_result.steps:
            if _count_slo_failures(step) > 0:
                failures.append((flow_result.path_id, step))

    if not failures:
        return

    failures.sort(key=lambda x: _count_slo_failures(x[1]), reverse=True)
    shown = failures[:5]

    print()
    print(f"SLO Failures ({len(shown)} of {len(failures)} shown, sorted by severity)")
    print(f"  {'Path':<5} {'Question':<40} {'R':>3} {'F':>3} {'A':>3}")
    print(f"  {'-'*5} {'-'*40} {'-'*3} {'-'*3} {'-'*3}")

    def fmt(passed: bool) -> str:
        return "Y" if passed else "X"

    for path_id, step in shown:
        q = step.question[:38] + "..." if len(step.question) > 38 else step.question
        r = fmt(step.relevance_score >= SLO_FLOW_RELEVANCE)
        f = fmt(step.faithfulness_score >= SLO_FLOW_FAITHFULNESS)
        a = fmt(step.answer_correctness_score >= SLO_FLOW_ANSWER_CORRECTNESS)
        print(f"  {path_id+1:<5} {q:<40} {r:>3} {f:>3} {a:>3}")


def print_summary(results: FlowEvalResults, latency_pcts: dict[str, float] | None = None) -> bool:
    """Print evaluation summary with SLO status. Returns True if all SLOs passed."""
    print()
    print("Flow Evaluation Summary")
    print("=" * 50)

    all_passed = True
    current_section = ""

    for spec in SLO_SPECS:
        if spec.section != current_section:
            current_section = spec.section
            print(f"\n{current_section}")

        value = spec.get_value(results)
        passed = _slo_passed(spec, results)
        if not passed:
            all_passed = False

        val_str, target_str = _format_slo(spec, value)
        print(f"  {spec.label}: {val_str} ({target_str} SLO) {'PASS' if passed else 'FAIL'}")

    # RAGAS Reliability
    ragas_ok = results.ragas_metrics_total - results.ragas_metrics_failed
    ragas_passed = results.ragas_success_rate >= 0.9
    if not ragas_passed:
        all_passed = False
    print("\nRAGAS Reliability")
    print(
        f"  Metrics Success: {ragas_ok}/{results.ragas_metrics_total}"
        f" ({results.ragas_success_rate:.1%}) (>=90.0% SLO)"
        f" {'PASS' if ragas_passed else 'FAIL'}"
    )

    if latency_pcts:
        print("\nLangSmith (info)")
        for key in ("fetch", "answer", "followup"):
            print(f"  {key.capitalize()}: {latency_pcts.get(key, 0):.1%}")

    _print_slo_failures(results)
    return all_passed


# =============================================================================
# RAGAS Judging
# =============================================================================


class RagasMetrics(TypedDict):
    """RAGAS evaluation metrics."""

    relevance: float
    faithfulness: float
    answer_correctness: float
    explanation: str
    ragas_metrics_total: int
    ragas_metrics_failed: int


def judge_answer(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    timeout: int = RAGAS_TIMEOUT_SECONDS,
) -> dict:
    """Judge an answer using RAGAS metrics with timeout."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                evaluate_single, question, answer, contexts, reference_answer or ""
            )
            result = future.result(timeout=timeout)

        ragas_error = result.get("error")
        nan_metrics: list[str] = result.get("nan_metrics", [])  # type: ignore[assignment]
        return {
            "relevance": result["answer_relevancy"],
            "faithfulness": result["faithfulness"],
            "answer_correctness": result.get("answer_correctness", 0.0),
            "explanation": f"RAGAS error: {ragas_error}" if ragas_error else "",
            "ragas_failed": ragas_error is not None,
            "nan_metrics": nan_metrics,
        }
    except TimeoutError:
        logger.warning(f"RAGAS judge timed out after {timeout}s")
        return {"relevance": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0,
                "explanation": f"RAGAS timeout after {timeout}s", "ragas_failed": True, "nan_metrics": []}
    except Exception as e:
        logger.warning(f"RAGAS judge failed: {e}")
        return {"relevance": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0,
                "explanation": f"RAGAS error: {e}", "ragas_failed": True, "nan_metrics": []}


def _count_failed_metrics(result: dict, metric_names: tuple[str, ...]) -> int:
    """Count failed metrics from a RAGAS result."""
    if result.get("ragas_failed"):
        return len(metric_names)
    nan_metrics = result.get("nan_metrics", [])
    return sum(1 for m in nan_metrics if m in metric_names)


def _evaluate_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str | None,
) -> RagasMetrics:
    """Run RAGAS evaluation for answer quality metrics."""
    if not contexts:
        return {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "explanation": "No context available",
            "ragas_metrics_total": 0,
            "ragas_metrics_failed": 0,
        }

    result = judge_answer(question, answer, contexts, reference_answer=expected_answer)
    return {
        "relevance": result["relevance"],
        "faithfulness": result["faithfulness"],
        "answer_correctness": result["answer_correctness"],
        "explanation": result["explanation"],
        "ragas_metrics_total": 3,
        "ragas_metrics_failed": _count_failed_metrics(
            result, ("answer_relevancy", "faithfulness", "answer_correctness")
        ),
    }


# =============================================================================
# Evaluation Functions
# =============================================================================


def _invoke_agent(
    question: str,
    session_id: str | None = None,
    timeout: int = AGENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Invoke the agent graph and return result with timeout."""
    from backend.agent.graph import agent_graph, build_thread_config

    state: dict[str, Any] = {"question": question, "session_id": session_id}
    config = build_thread_config(session_id)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(agent_graph.invoke, state, config=config)
        return future.result(timeout=timeout)


def _create_error_step_result(question: str, latency_ms: int, error: str) -> FlowStepResult:
    """Create a FlowStepResult for error cases."""
    return FlowStepResult(
        question=question,
        answer="",
        latency_ms=latency_ms,
        has_answer=False,
        relevance_score=0.0,
        faithfulness_score=0.0,
        answer_correctness_score=0.0,
        judge_explanation=error,
        error=error,
    )


def test_single_question(
    question: str,
    session_id: str,
    use_judge: bool = True,
) -> FlowStepResult:
    """Test a single question and return answer with metrics."""
    start_time = time.time()

    try:
        result = _invoke_agent(question=question, session_id=session_id)
        latency_ms = int((time.time() - start_time) * 1000)

        answer = result.get("answer", "")
        has_answer = bool(answer and len(answer) > MIN_ANSWER_LENGTH)

        contexts = []
        sql_results = result.get("sql_results", {})
        if sql_results:
            contexts.append(json.dumps(sql_results, indent=2, default=str))

        expected_answer = get_expected_answer(question)

        ragas: RagasMetrics = {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "explanation": "",
            "ragas_metrics_total": 0,
            "ragas_metrics_failed": 0,
        }
        if use_judge and has_answer:
            ragas = _evaluate_ragas(question, answer, contexts, expected_answer)

        return FlowStepResult(
            question=question,
            answer=answer,
            latency_ms=latency_ms,
            has_answer=has_answer,
            relevance_score=ragas["relevance"],
            faithfulness_score=ragas["faithfulness"],
            answer_correctness_score=ragas["answer_correctness"],
            judge_explanation=ragas["explanation"],
            error=None,
            ragas_metrics_total=ragas["ragas_metrics_total"],
            ragas_metrics_failed=ragas["ragas_metrics_failed"],
        )

    except TimeoutError:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Timeout testing question '{question}' after {AGENT_TIMEOUT_SECONDS}s")
        return _create_error_step_result(question, latency_ms, f"Timeout after {AGENT_TIMEOUT_SECONDS}s")
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error testing question '{question}': {e}")
        return _create_error_step_result(question, latency_ms, f"Error: {e}")


def test_flow(
    path: list[str],
    path_id: int,
    use_judge: bool = True,
) -> FlowResult:
    """Test a conversation flow (sequential questions with memory)."""
    session_id = f"flow_eval_{path_id}_{int(time.time())}"
    steps: list[FlowStepResult] = []
    total_latency = 0
    success = True
    first_error: str | None = None

    for question in path:
        step_result = test_single_question(question, session_id, use_judge)
        steps.append(step_result)
        total_latency += step_result.latency_ms

        if not step_result.passed:
            success = False
            if first_error is None and step_result.error:
                first_error = step_result.error

    return FlowResult(
        path_id=path_id,
        questions=path,
        steps=steps,
        total_latency_ms=total_latency,
        success=success,
        error=first_error,
    )


def run_flow_eval(
    max_paths: int | None = None,
    use_judge: bool = True,
    concurrency: int = 5,
) -> FlowEvalResults:
    """Run flow evaluation on all paths with parallel execution."""
    eval_start_time = time.time()

    all_paths = get_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths else all_paths

    print(f"Paths to test: {len(paths_to_test)}")
    print(f"Concurrency: {concurrency}")
    print()

    results: list[FlowResult] = []

    with Progress(
        TextColumn("[cyan]Evaluating paths"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    ) as progress:
        task = progress.add_task("", total=len(paths_to_test))

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(test_flow, path, i, use_judge): i
                for i, path in enumerate(paths_to_test)
            }

            for future in as_completed(futures):
                path_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(FlowResult(
                        path_id=path_id, questions=paths_to_test[path_id], steps=[],
                        total_latency_ms=0, success=False, error=str(e),
                    ))
                progress.advance(task)

    results.sort(key=lambda r: r.path_id)

    all_steps = [s for r in results for s in r.steps]
    paths_passed = sum(1 for r in results if r.success)
    total_questions = sum(len(r.steps) for r in results)
    questions_passed = sum(sum(1 for s in r.steps if s.passed) for r in results)
    total_latency = sum(r.total_latency_ms for r in results)

    return FlowEvalResults(
        total_paths=len(all_paths),
        paths_tested=len(results),
        paths_passed=paths_passed,
        paths_failed=len(results) - paths_passed,
        total_questions=total_questions,
        questions_passed=questions_passed,
        questions_failed=total_questions - questions_passed,
        avg_relevance=sum(s.relevance_score for s in all_steps) / len(all_steps) if all_steps else 0.0,
        avg_faithfulness=sum(s.faithfulness_score for s in all_steps) / len(all_steps) if all_steps else 0.0,
        avg_answer_correctness=sum(s.answer_correctness_score for s in all_steps) / len(all_steps) if all_steps else 0.0,
        ragas_metrics_total=sum(s.ragas_metrics_total for s in all_steps),
        ragas_metrics_failed=sum(s.ragas_metrics_failed for s in all_steps),
        total_latency_ms=total_latency,
        avg_latency_per_question_ms=total_latency / total_questions if total_questions > 0 else 0,
        wall_clock_ms=int((time.time() - eval_start_time) * 1000),
        failed_paths=[r for r in results if not r.success],
        all_results=results,
    )


# =============================================================================
# CLI
# =============================================================================


def _run_eval(limit: int | None) -> None:
    """Run the flow evaluation."""
    eval_start_time = time.time()

    stats = get_tree_stats()
    print("\nQuestion Tree Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    try:
        results = run_flow_eval(max_paths=limit, use_judge=True, concurrency=1)
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        traceback.print_exc()
        return

    elapsed_minutes = int((time.time() - eval_start_time) / 60) + 1
    latency_pcts = get_latency_percentages(minutes_ago=max(elapsed_minutes, 5))
    print_summary(results, latency_pcts=latency_pcts)


def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    _run_eval(limit=limit)


if __name__ == "__main__":
    typer.run(main)
