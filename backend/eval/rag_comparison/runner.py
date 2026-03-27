"""Main runner for RAG retrieval strategy comparison."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml

from backend.agent.rag.indexer import get_index
from backend.eval.answer.text.ragas import evaluate_single
from backend.eval.rag_comparison.configs import build_query_engine, get_configs_by_name
from backend.eval.rag_comparison.models import ComparisonResults, ConfigCaseResult, ConfigResults

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


def _load_questions(limit: int | None = None) -> list[dict]:
    """Load evaluation questions from YAML."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    questions = data["questions"]
    if limit is not None:
        questions = questions[:limit]
    return questions


def _run_single_question(
    query_engine,
    question: str,
    reference_answer: str,
) -> ConfigCaseResult:
    """Run a single question through one query engine and evaluate with RAGAS."""
    start = time.time()
    response = query_engine.query(question)
    latency = time.time() - start

    answer = str(response)
    contexts = [node.text for node in response.source_nodes]
    sources = [node.metadata.get("source", "unknown") for node in response.source_nodes]

    # Evaluate with RAGAS
    ragas_scores = evaluate_single(
        question=question,
        answer=answer,
        contexts=contexts,
        reference_answer=reference_answer,
    )

    return ConfigCaseResult(
        question=question,
        reference_answer=reference_answer,
        answer=answer,
        contexts=contexts,
        sources=sources,
        latency_seconds=latency,
        answer_correctness=ragas_scores["answer_correctness"],
        answer_relevancy=ragas_scores["answer_relevancy"],
        faithfulness=ragas_scores["faithfulness"],
        nan_metrics=ragas_scores.get("nan_metrics", []),
    )


def run_rag_comparison(
    config_names: list[str] | None = None,
    limit: int | None = None,
) -> ComparisonResults:
    """Run the full RAG retrieval comparison.

    Args:
        config_names: Specific config names to test, or None for all 6
        limit: Max number of questions to evaluate per config

    Returns:
        ComparisonResults with per-config and per-question scores
    """
    configs = get_configs_by_name(config_names)
    questions = _load_questions(limit)

    print(f"\nRAG Comparison: {len(configs)} configs × {len(questions)} questions")
    print(f"Configs: {', '.join(c.name for c in configs)}\n")

    index = get_index()
    results = ComparisonResults()

    for config in configs:
        print(f"  Running: {config.name} ", end="", flush=True)
        config_results = ConfigResults(config_name=config.name)

        query_engine = build_query_engine(config, index)

        for i, q in enumerate(questions):
            try:
                case = _run_single_question(
                    query_engine,
                    question=q["question"],
                    reference_answer=q["reference_answer"],
                )
                config_results.cases.append(case)
                print(".", end="", flush=True)
            except Exception as e:
                logger.error(f"[RAG Compare] Failed on Q{i+1} ({config.name}): {e}")
                config_results.cases.append(ConfigCaseResult(
                    question=q["question"],
                    reference_answer=q["reference_answer"],
                    answer=f"ERROR: {e}",
                    contexts=[],
                    sources=[],
                    latency_seconds=0.0,
                    nan_metrics=["answer_correctness", "answer_relevancy", "faithfulness"],
                ))
                print("x", end="", flush=True)

        print(f"  composite={config_results.avg_composite:.3f}")
        results.configs.append(config_results)

    return results


__all__ = ["run_rag_comparison"]
