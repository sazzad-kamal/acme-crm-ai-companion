"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision

logger = logging.getLogger(__name__)


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """
    Evaluate a single Q&A pair using RAGAS metrics.

    Args:
        question: The user's question
        answer: The agent's answer
        contexts: List of retrieved context strings

    Returns:
        dict with answer_relevancy, faithfulness, context_precision (0.0-1.0)
    """
    # RAGAS requires non-empty contexts
    if not contexts:
        contexts = [""]

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    try:
        result = evaluate(
            dataset,
            metrics=[answer_relevancy, faithfulness, context_precision],
        )

        return {
            "answer_relevancy": float(result["answer_relevancy"]),
            "faithfulness": float(result["faithfulness"]),
            "context_precision": float(result["context_precision"]),
        }
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
        }


__all__ = ["evaluate_single"]
