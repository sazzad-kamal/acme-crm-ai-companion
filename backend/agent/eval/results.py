"""
Evaluation results handling - saving, parsing, and LLM judge.

Provides utilities for persisting eval results and LLM-based judgment.
"""

import json
import os
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from rich.panel import Panel

from backend.agent.eval.formatting import console


def parse_json_response(text: str) -> dict[str, Any] | None:
    """
    Parse JSON from LLM response text.

    Handles responses that may have markdown code blocks around JSON.

    Args:
        text: Raw text that may contain JSON

    Returns:
        Parsed dict or None if parsing fails
    """
    # Strip markdown code blocks if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


async def run_llm_judge(
    question: str,
    generated_answer: str,
    expected_answer: str,
    judge_model: str = "openai:gpt-4.1-mini",
) -> tuple[bool, str, float | None]:
    """
    Use an LLM to judge if generated answer matches expected answer.

    Args:
        question: The original question
        generated_answer: The answer from the system
        expected_answer: The expected/reference answer
        judge_model: LLM model to use for judgment

    Returns:
        Tuple of (passed, explanation, confidence_score)
    """
    system_prompt = """You are an evaluation judge. Compare the generated answer
to the expected answer and determine if they are semantically equivalent.

Respond in JSON format:
{
    "passed": true/false,
    "explanation": "Brief explanation of your judgment",
    "confidence": 0.0-1.0
}"""

    human_prompt = f"""Question: {question}

Generated Answer: {generated_answer}

Expected Answer: {expected_answer}

Judge if the generated answer is semantically correct compared to the expected answer.
Consider partial matches and different phrasings that convey the same meaning."""

    try:
        model = init_chat_model(judge_model)
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        result = parse_json_response(response.content)
        if result:
            return (
                result.get("passed", False),
                result.get("explanation", "No explanation provided"),
                result.get("confidence"),
            )
        return False, "Failed to parse judge response", None

    except Exception as e:
        return False, f"Judge error: {e}", None


def save_eval_results(
    results_dir: str,
    filename: str,
    data: dict[str, Any],
) -> str:
    """
    Save evaluation results to JSON file.

    Args:
        results_dir: Directory to save results in
        filename: Name of the output file
        data: Results data to save

    Returns:
        Path to saved file
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


def finalize_eval_cli(
    exit_code: int,
    summary_message: str | None = None,
) -> int:
    """
    Finalize eval CLI output and return exit code.

    Args:
        exit_code: Exit code to return (0 for success)
        summary_message: Optional summary message to print

    Returns:
        The exit code
    """
    if summary_message:
        style = "green" if exit_code == 0 else "red"
        console.print(Panel(summary_message, style=style))

    return exit_code


__all__ = [
    "parse_json_response",
    "run_llm_judge",
    "save_eval_results",
    "finalize_eval_cli",
]
