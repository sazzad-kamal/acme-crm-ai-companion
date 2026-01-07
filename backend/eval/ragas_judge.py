"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, faithfulness

logger = logging.getLogger(__name__)


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
) -> dict[str, float]:
    """
    Evaluate a single Q&A pair using RAGAS metrics.

    Args:
        question: The user's question
        answer: The agent's answer
        contexts: List of retrieved context strings
        reference_answer: Optional ground truth answer for answer_correctness metric

    Returns:
        dict with answer_relevancy, faithfulness, context_precision, answer_correctness (0.0-1.0)
    """
    # RAGAS requires non-empty contexts
    if not contexts:
        contexts = [""]

    dataset_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    # Select metrics - add answer_correctness if reference provided
    metrics = [answer_relevancy, faithfulness, context_precision]
    if reference_answer:
        dataset_dict["ground_truth"] = [reference_answer]
        metrics.append(answer_correctness)

    dataset = Dataset.from_dict(dataset_dict)

    try:
        result = evaluate(dataset, metrics=metrics)

        return {
            "answer_relevancy": float(result["answer_relevancy"]),  # type: ignore[arg-type,index]
            "faithfulness": float(result["faithfulness"]),  # type: ignore[arg-type,index]
            "context_precision": float(result["context_precision"]),  # type: ignore[arg-type,index]
            "answer_correctness": float(result.get("answer_correctness", 0.0) or 0.0),  # type: ignore[union-attr]
        }
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "answer_correctness": 0.0,
        }


__all__ = ["evaluate_single"]
