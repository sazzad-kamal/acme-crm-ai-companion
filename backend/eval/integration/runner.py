"""Evaluation runner - tests conversation paths."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, cast

from dotenv import load_dotenv

from backend.eval.answer.action.judge import judge_suggested_action
from backend.eval.answer.text.ragas import RAGAS_METRICS_COUNT, evaluate_single
from backend.eval.integration.models import (
    ConvoEvalResults,
    ConvoStepResult,
)
from backend.eval.integration.tree import (
    get_all_paths,
    get_expected_action,
    get_expected_answer,
)
from backend.eval.shared.version import get_eval_metadata

load_dotenv()

logger = logging.getLogger(__name__)


def _invoke_agent(question: str, session_id: str | None = None) -> dict[str, Any]:
    """Invoke the agent graph and return result."""
    from backend.agent.graph import agent_graph, build_thread_config
    from backend.agent.state import AgentState

    state: AgentState = {"question": question}
    config = build_thread_config(session_id)
    return cast(dict[str, Any], agent_graph.invoke(state, config=config))


def _evaluate_ragas(
    question: str, answer: str, sql_results: dict,
) -> dict[str, Any]:
    """Run RAGAS evaluation. Returns ConvoStepResult fields."""
    contexts = [json.dumps(sql_results, default=str)]
    expected = get_expected_answer(question)
    try:
        ragas = evaluate_single(question, answer, contexts, expected or "")
        nan_metrics = cast(list[str], ragas.get("nan_metrics", []))
        return {
            "relevance_score": cast(float, ragas["answer_relevancy"]),
            "faithfulness_score": cast(float, ragas["faithfulness"]),
            "answer_correctness_score": cast(float, ragas["answer_correctness"]),
            "ragas_metrics_total": RAGAS_METRICS_COUNT,
            "ragas_metrics_failed": len(nan_metrics),
        }
    except Exception as e:
        logger.warning(f"RAGAS failed: {e}")
        return {"errors": [f"RAGAS failed: {e}"]}


def _evaluate_action(
    question: str, answer: str, suggested_action: str | None,
) -> dict[str, Any]:
    """Evaluate action quality. Returns ConvoStepResult fields."""
    expected_action = get_expected_action(question)
    action_rel = action_act = action_app = 0.0
    action_passed = True

    if (expected_action is True and suggested_action is None) or (
        expected_action is False and suggested_action is not None
    ):
        action_passed = False
    elif suggested_action:
        try:
            action_passed, action_rel, action_act, action_app, _ = judge_suggested_action(
                question, answer, suggested_action,
            )
        except Exception as e:
            logger.warning(f"Action judge failed: {e}")
            action_passed = False

    return {
        "expected_action": expected_action,
        "suggested_action": suggested_action,
        "action_relevance": action_rel,
        "action_actionability": action_act,
        "action_appropriateness": action_app,
        "action_passed": action_passed,
    }


def test_single_question(question: str, session_id: str) -> ConvoStepResult:
    """Test a single question and return answer with metrics."""
    start_time = time.perf_counter()
    try:
        result = _invoke_agent(question=question, session_id=session_id)
        latency_ms = (time.perf_counter() - start_time) * 1000

        answer = result.get("answer", "")

        kwargs: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "latency_ms": latency_ms,
        }

        sql_results = result.get("sql_results", {})
        if answer and sql_results:
            kwargs.update(_evaluate_ragas(question, answer, sql_results))

        suggested_action = result.get("suggested_action")
        kwargs.update(_evaluate_action(question, answer, suggested_action))

        return ConvoStepResult(**kwargs)

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Error testing question '{question}': {e}")
        return ConvoStepResult(
            question=question, answer="", errors=[str(e)], latency_ms=latency_ms
        )


def run_convo_eval(max_paths: int | None = None) -> ConvoEvalResults:
    """Run conversation evaluation on all paths."""
    # Get eval metadata for versioning
    metadata = get_eval_metadata()

    all_paths = get_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths is not None else all_paths

    total_questions = sum(len(p) for p in paths_to_test)
    results = ConvoEvalResults(
        total=total_questions,
        eval_version=metadata["version"],
        eval_checksum=metadata["checksum"],
    )
    question_num = 0

    for path_idx, path in enumerate(paths_to_test):
        session_id = f"convo_eval_{path_idx}_{int(time.time())}"
        for question in path:
            question_num += 1
            step = test_single_question(question, session_id)
            results.cases.append(step)
            status = "PASS" if step.passed else "FAIL"
            latency_str = f" ({step.latency_ms:.0f}ms)" if step.latency_ms > 0 else ""
            print(f"  [{question_num}/{total_questions}] {status}{latency_str} {question[:45]}...")

    results.compute_aggregates()
    return results
