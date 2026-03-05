"""Eval cases versioning and checksumming.

Provides version tracking for evaluation test cases to detect changes
and ensure reproducibility of evaluation results.
"""

import hashlib
from functools import cache
from pathlib import Path

import yaml

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"

# Semantic version for eval cases
# Bump minor for new questions, major for format changes
EVAL_CASES_VERSION = "1.0.0"


@cache
def get_eval_cases_checksum() -> str:
    """Get SHA256 checksum of the eval cases file.

    This allows detecting when eval cases have changed between runs,
    which is important for regression tracking.
    """
    with open(QUESTIONS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


@cache
def get_eval_cases_stats() -> dict:
    """Get statistics about the eval cases.

    Returns:
        Dict with counts by difficulty and expected_action status
    """
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)

    questions = data.get("questions", [])

    stats = {
        "total_questions": len(questions),
        "by_difficulty": {},
        "with_expected_answer": 0,
        "with_expected_action": 0,
    }

    for q in questions:
        difficulty = q.get("difficulty", 1)
        stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1
        if q.get("expected_answer"):
            stats["with_expected_answer"] += 1
        if q.get("expected_action"):
            stats["with_expected_action"] += 1

    return stats


def get_eval_metadata() -> dict:
    """Get complete metadata for eval run.

    Returns:
        Dict with version, checksum, and stats
    """
    return {
        "version": EVAL_CASES_VERSION,
        "checksum": get_eval_cases_checksum(),
        "stats": get_eval_cases_stats(),
    }


__all__ = [
    "EVAL_CASES_VERSION",
    "get_eval_cases_checksum",
    "get_eval_cases_stats",
    "get_eval_metadata",
]
