"""Shared evaluation utilities."""

import os
from pathlib import Path

from dotenv import load_dotenv

from backend.eval.shared.formatting import console
from backend.eval.shared.ragas import evaluate_single


def is_mock_mode() -> bool:
    """Check if MOCK_LLM mode is enabled."""
    return os.environ.get("MOCK_LLM", "0") == "1"


def load_project_env() -> None:
    """Load .env from project root."""
    project_root = Path(__file__).parent.parent.parent.parent
    load_dotenv(project_root / ".env")


__all__ = [
    "console",
    "evaluate_single",
    "is_mock_mode",
    "load_project_env",
]
