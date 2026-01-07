"""E2E evaluation runner - tests the full orchestrator pipeline."""

from __future__ import annotations

import threading
import time

from rich.progress import track

from backend.agent.graph import agent_graph, build_thread_config, clear_thread
from backend.eval.base import console, print_eval_header
from backend.eval.parallel import run_parallel_evaluation, calculate_p95_latency
from backend.eval.ragas_judge import evaluate_single
from backend.eval.models import E2EEvalResult, E2EEvalSummary
from backend.eval.e2e.test_cases import E2E_TEST_CASES


def _invoke_agent(question: str, session_id: str | None = None) -> dict:
    """Invoke the agent graph and return state for eval."""
    state = {"question": question, "sources": []}
    config = build_thread_config(session_id)
    return agent_graph.invoke(state, config=config)


def judge_e2e_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """Judge an end-to-end response using RAGAS metrics."""
    try:
        result = evaluate_single(question, answer, contexts)
        return {
            "answer_relevance": result["answer_relevancy"],
            "faithfulness": result["faithfulness"],
            "context_precision": result["context_precision"],
            "explanation": "",
        }
    except Exception as e:
        return {
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "explanation": f"RAGAS error: {e}",
        }


def check_refusal_response(
    answer: str,
    expected_refusal: bool,
    refusal_keywords: list[str],
    forbidden_keywords: list[str],
) -> tuple[bool, bool]:
    """
    Check if the response correctly handles refusal.

    Returns:
        (refusal_correct, has_forbidden_content)
    """
    answer_lower = answer.lower()

    # Check for forbidden content
    has_forbidden = any(kw.lower() in answer_lower for kw in forbidden_keywords)

    if expected_refusal:
        # Should refuse - check for refusal keywords
        has_refusal = any(kw.lower() in answer_lower for kw in refusal_keywords)
        refusal_correct = has_refusal
    else:
        # No refusal expected - just check forbidden keywords
        refusal_correct = True

    return refusal_correct, has_forbidden


def run_e2e_test(
    test_case: dict,
    verbose: bool = False,
    agent_lock: threading.Lock | None = None,
) -> E2EEvalResult:
    """Run a single end-to-end test case.

    Args:
        test_case: Test case dictionary with question, expected values, etc.
        verbose: Print detailed progress
        agent_lock: Optional lock for thread-safe agent/RAG access in parallel mode
    """
    test_id = test_case["id"]
    question = test_case["question"]
    category = test_case["category"]
    expected_company = test_case.get("expected_company")
    expected_intent = test_case.get("expected_intent")
    session_id = test_case.get("session_id")

    # Adversarial test fields
    expected_refusal = test_case.get("expected_refusal", False)
    refusal_keywords = test_case.get("refusal_keywords", [])
    forbidden_keywords = test_case.get("forbidden_keywords", [])

    if verbose:
        console.print(f"\n  Testing: {test_id}")
        console.print(f"    Q: {question[:60]}...")
        if session_id:
            console.print(f"    Session: {session_id}")

    start_time = time.time()
    error = None

    try:
        # Run the full agent pipeline
        if agent_lock:
            with agent_lock:
                result = _invoke_agent(question, session_id=session_id)
        else:
            result = _invoke_agent(question, session_id=session_id)
        latency = (time.time() - start_time) * 1000

        answer = result.get("answer", "")
        sources = [s.id if hasattr(s, "id") else s.get("id", "") for s in result.get("sources", [])]

        actual_company = result.get("resolved_company_id")
        actual_intent = result.get("intent", "general")

    except Exception as e:
        error = str(e)
        latency = (time.time() - start_time) * 1000
        return E2EEvalResult(
            test_case_id=test_id,
            question=question,
            category=category,
            expected_company_id=expected_company,
            actual_company_id=None,
            company_correct=expected_company is None,
            expected_intent=expected_intent,
            actual_intent=None,
            intent_correct=expected_intent is None,
            expected_refusal=expected_refusal,
            refusal_correct=False,
            has_forbidden_content=False,
            answer="",
            answer_relevance=0.0,
            faithfulness=0.0,
            context_precision=0.0,
            has_sources=False,
            latency_ms=latency,
            total_tokens=0,
            error=error,
        )

    # Judge the response using RAGAS
    judge_result = judge_e2e_response(question, answer, sources)

    # Check company extraction correctness
    company_correct = expected_company is None or actual_company == expected_company

    # Check intent classification correctness
    intent_correct = expected_intent is None or actual_intent == expected_intent

    # Check refusal correctness
    refusal_correct, has_forbidden = check_refusal_response(
        answer, expected_refusal, refusal_keywords, forbidden_keywords
    )

    if verbose:
        relevance = f"{judge_result['answer_relevance']:.2f}"
        faithful = f"{judge_result['faithfulness']:.2f}"
        ctx_prec = f"{judge_result['context_precision']:.2f}"
        company_mark = "Y" if company_correct else "N"
        intent_mark = "Y" if intent_correct else "N"
        console.print(
            f"    Company: {actual_company} [{company_mark}], Intent: {actual_intent} [{intent_mark}]"
        )
        console.print(
            f"    Relevance: {relevance}, Faithfulness: {faithful}, CtxPrec: {ctx_prec}"
        )
        if expected_refusal:
            refusal_mark = "Y" if refusal_correct else "N"
            console.print(f"    Refusal: [{refusal_mark}], Forbidden: {has_forbidden}")

    return E2EEvalResult(
        test_case_id=test_id,
        question=question,
        category=category,
        expected_company_id=expected_company,
        actual_company_id=actual_company,
        company_correct=company_correct,
        expected_intent=expected_intent,
        actual_intent=actual_intent,
        intent_correct=intent_correct,
        expected_refusal=expected_refusal,
        refusal_correct=refusal_correct,
        has_forbidden_content=has_forbidden,
        answer=answer[:1000],
        answer_relevance=judge_result["answer_relevance"],
        faithfulness=judge_result["faithfulness"],
        context_precision=judge_result["context_precision"],
        judge_explanation=judge_result["explanation"],
        has_sources=len(sources) > 0,
        sources=sources,
        latency_ms=latency,
        total_tokens=0,
        error=error,
    )


def run_e2e_eval(
    limit: int | None = None,
    verbose: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
) -> tuple[list[E2EEvalResult], E2EEvalSummary]:
    """
    Run end-to-end evaluation.

    Args:
        limit: Limit number of tests to run
        verbose: Print detailed progress
        parallel: Run tests in parallel for faster execution
        max_workers: Maximum number of parallel workers (default 4)

    Returns:
        Tuple of (results list, summary)
    """
    print_eval_header(
        "[bold blue]End-to-End Agent Evaluation[/bold blue]",
        "Testing full orchestrator pipeline",
    )

    test_cases = E2E_TEST_CASES[:limit] if limit else E2E_TEST_CASES
    results = []

    # Separate multi-turn tests from regular tests
    multi_turn_tests = [t for t in test_cases if t.get("session_id")]
    regular_tests = [t for t in test_cases if not t.get("session_id")]

    # Clear any existing sessions
    session_ids = set(t.get("session_id") for t in multi_turn_tests if t.get("session_id"))
    for sid in session_ids:
        clear_thread(sid)

    if parallel and regular_tests:
        def evaluate_fn(test_case: dict, lock: threading.Lock | None) -> E2EEvalResult:
            return run_e2e_test(test_case, verbose, lock)

        results = run_parallel_evaluation(
            items=regular_tests,
            evaluate_fn=evaluate_fn,
            max_workers=max_workers,
            description="Running E2E tests",
            id_field="id",
            use_lock=True,
        )
    elif regular_tests:
        for test_case in track(regular_tests, description="Running regular tests..."):
            result = run_e2e_test(test_case, verbose=verbose)
            results.append(result)

    # Run multi-turn tests sequentially
    if multi_turn_tests:
        console.print(
            f"[cyan]Running {len(multi_turn_tests)} multi-turn tests sequentially...[/cyan]"
        )
        for test_case in track(multi_turn_tests, description="Running multi-turn tests..."):
            result = run_e2e_test(test_case, verbose=verbose)
            results.append(result)

    # Compute summary
    total = len(results)

    # Company extraction accuracy
    company_tests = [r for r in results if r.expected_company_id is not None]
    company_correct_count = sum(1 for r in company_tests if r.company_correct)
    company_accuracy = company_correct_count / len(company_tests) if company_tests else 1.0

    # Intent classification accuracy
    intent_tests = [r for r in results if r.expected_intent is not None]
    intent_correct_count = sum(1 for r in intent_tests if r.intent_correct)
    intent_accuracy = intent_correct_count / len(intent_tests) if intent_tests else 1.0

    # Answer quality metrics (RAGAS)
    relevance_rate = sum(r.answer_relevance for r in results) / total if total > 0 else 0
    faithfulness_rate = sum(r.faithfulness for r in results) / total if total > 0 else 0
    context_precision_rate = sum(r.context_precision for r in results) / total if total > 0 else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

    # Compute P95 latency
    latencies = [r.latency_ms for r in results]
    p95_latency = calculate_p95_latency(latencies)

    from backend.eval.models import SLO_LATENCY_P95_MS
    latency_slo_pass = p95_latency <= SLO_LATENCY_P95_MS

    # By category breakdown
    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {
                "count": 0,
                "relevance_sum": 0.0,
                "faithfulness_sum": 0.0,
                "context_precision_sum": 0.0,
            }
        by_category[cat]["count"] += 1
        by_category[cat]["relevance_sum"] += r.answer_relevance
        by_category[cat]["faithfulness_sum"] += r.faithfulness
        by_category[cat]["context_precision_sum"] += r.context_precision

    for cat in by_category:
        count = by_category[cat]["count"]
        by_category[cat]["relevance_rate"] = by_category[cat]["relevance_sum"] / count
        by_category[cat]["faithfulness_rate"] = by_category[cat]["faithfulness_sum"] / count
        by_category[cat]["context_precision_rate"] = by_category[cat]["context_precision_sum"] / count

    summary = E2EEvalSummary(
        total_tests=total,
        company_extraction_accuracy=company_accuracy,
        intent_accuracy=intent_accuracy,
        answer_relevance_rate=relevance_rate,
        faithfulness_rate=faithfulness_rate,
        context_precision_rate=context_precision_rate,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        latency_slo_pass=latency_slo_pass,
        by_category=by_category,
    )

    return results, summary
