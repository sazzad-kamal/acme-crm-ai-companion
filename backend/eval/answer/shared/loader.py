"""Shared loader utilities for answer evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import duckdb

from backend.agent.answer.answerer import call_answer_chain, extract_suggested_action
from backend.agent.fetch.sql.executor import execute_sql
from backend.eval.answer.shared.models import Question

QUESTIONS_PATH = Path(__file__).parent.parent.parent / "shared" / "questions.yaml"


def load_questions() -> list[Question]:
    """Load questions from shared YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return [Question(**item) for item in data.get("questions", [])]


def generate_answer(
    question: Question, conn: duckdb.DuckDBPyConnection
) -> tuple[str, str | None, list[dict], int, str | None]:
    """Execute SQL and generate answer.

    Returns:
        tuple: (answer_text, suggested_action, sql_results, latency_ms, error)
    """
    start = time.time()
    try:
        # Step 1: Execute expected SQL
        sql_results, sql_error = execute_sql(question.expected_sql, conn)
        if sql_error:
            return "", None, [], int((time.time() - start) * 1000), f"SQL error: {sql_error}"

        # Step 2: Call answer chain
        raw_answer = call_answer_chain(question.text, sql_results={"rows": sql_results})

        # Step 3: Extract suggested action
        answer, suggested_action = extract_suggested_action(raw_answer)

        latency_ms = int((time.time() - start) * 1000)
        return answer, suggested_action, sql_results, latency_ms, None

    except Exception as e:
        return "", None, [], int((time.time() - start) * 1000), f"Error: {e}"


__all__ = ["load_questions", "generate_answer", "QUESTIONS_PATH"]
