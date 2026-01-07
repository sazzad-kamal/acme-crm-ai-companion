"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import os
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections.answer_correctness.metric import AnswerCorrectness
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevancy
from ragas.metrics.collections.context_precision.metric import ContextPrecision
from ragas.metrics.collections.faithfulness.metric import Faithfulness

logger = logging.getLogger(__name__)


def _get_openai_client() -> AsyncOpenAI:
    """Get OpenAI async client."""
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _get_ragas_llm() -> Any:
    """Get LLM for RAGAS using native factory."""
    return llm_factory("gpt-4o-mini", client=_get_openai_client())


def _get_ragas_embeddings() -> Any:
    """Get embeddings for RAGAS using native OpenAI embeddings."""
    return RagasOpenAIEmbeddings(client=_get_openai_client())


def _create_metrics(
    llm: Any, embeddings: Any, include_correctness: bool = False
) -> list[Any]:
    """Create RAGAS metric instances with LLM and embeddings."""
    metrics: list[Any] = [
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
    ]
    if include_correctness:
        metrics.append(AnswerCorrectness(llm=llm, embeddings=embeddings))
    return metrics


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
        contexts = ["No context provided"]

    dataset_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    # Get LLM and embeddings for RAGAS 0.4.x
    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    # Create metric instances with LLM and embeddings
    include_correctness = reference_answer is not None
    if reference_answer:
        dataset_dict["ground_truth"] = [reference_answer]

    metrics = _create_metrics(ragas_llm, ragas_embeddings, include_correctness)
    dataset = Dataset.from_dict(dataset_dict)

    try:
        # Pass metrics to evaluate() - RAGAS 0.4.x API
        result = evaluate(dataset, metrics=metrics)

        # RAGAS 0.4.x returns EvaluationResult - convert to pandas DataFrame
        df = result.to_pandas()  # type: ignore[union-attr]

        def get_score(name: str) -> float:
            if name in df.columns and len(df) > 0:
                val = df[name].iloc[0]
                if val is None or (isinstance(val, float) and val != val):  # Check for NaN
                    return 0.0
                return float(val)
            return 0.0

        return {
            "answer_relevancy": get_score("answer_relevancy"),
            "faithfulness": get_score("faithfulness"),
            "context_precision": get_score("context_precision"),
            "answer_correctness": get_score("answer_correctness"),
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
