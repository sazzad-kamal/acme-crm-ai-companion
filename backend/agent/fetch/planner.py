"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from backend.agent.core.config import get_config
from backend.agent.core.llm import load_prompt

logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent


class SQLPlan(BaseModel):
    """LLM output containing SQL query and RAG flag."""

    sql: str = Field(description="The SQL query to execute")
    needs_rag: bool = Field(default=False, description="Whether RAG context is needed")


@lru_cache
def _get_client() -> OpenAI:
    """Get OpenAI client (cached)."""
    return OpenAI()


def get_sql_plan(question: str, conversation_history: str = "") -> SQLPlan:
    """
    Get SQL directly from LLM using SQL Sorcerer approach.

    Returns SQLPlan with SQL string and needs_rag flag.
    """
    config = get_config()

    prompt = load_prompt(_DIR / "prompt.txt").format(
        today=datetime.now().strftime("%Y-%m-%d"),
        conversation_history=conversation_history or "",
        question=question,
    )

    response = _get_client().beta.chat.completions.parse(
        model=config.router_model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        response_format=SQLPlan,
    )

    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse SQL plan from LLM response")
    logger.info("SQL Planner: %s (needs_rag=%s)", result.sql[:80], result.needs_rag)
    return result


__all__ = ["SQLPlan", "get_sql_plan"]
