"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from backend.agent.core.config import get_config
from backend.agent.llm.client import load_prompt

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


def _parse_response(response: str) -> tuple[str, bool]:
    """
    Parse LLM response to extract SQL and needs_rag flag.

    Returns:
        Tuple of (sql_string, needs_rag_bool)
    """
    # Extract SQL from ```sql ... ``` block
    sql = ""
    match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
    else:
        # Try to extract from ``` ... ``` block
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            sql = match.group(1).strip()
        else:
            # No code blocks - take everything before needs_rag line
            lines = response.strip().split("\n")
            sql_lines = [l for l in lines if not l.strip().lower().startswith("needs_rag")]
            sql = "\n".join(sql_lines).strip()

    # Extract needs_rag flag
    needs_rag = False
    rag_match = re.search(r"needs_rag:\s*(true|false)", response, re.IGNORECASE)
    if rag_match:
        needs_rag = rag_match.group(1).lower() == "true"

    return sql, needs_rag


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

    response = _get_client().chat.completions.create(
        model=config.router_model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    content = response.choices[0].message.content or ""
    sql, needs_rag = _parse_response(content)

    result = SQLPlan(
        sql=sql,
        needs_rag=needs_rag,
    )

    logger.info("SQL Planner: %s (needs_rag=%s)", sql[:80], needs_rag)
    return result


__all__ = ["SQLPlan", "get_sql_plan"]
