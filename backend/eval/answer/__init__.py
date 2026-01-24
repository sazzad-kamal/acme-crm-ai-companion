"""Answer evaluation modules.

This package contains two separate evaluation modules:
- text: Answer text quality evaluation using RAGAS metrics
- action: Suggested action quality evaluation using LLM Judge
"""

# Re-export from submodules for convenience
from backend.eval.answer.action import (
    ActionCaseResult,
    ActionEvalResults,
    ActionJudgeResult,
    judge_suggested_action,
    run_action_eval,
)
from backend.eval.answer.action import print_summary as print_action_summary
from backend.eval.answer.shared import Question, generate_answer, load_questions
from backend.eval.answer.text import TextCaseResult, TextEvalResults, run_text_eval
from backend.eval.answer.text import print_summary as print_text_summary

__all__ = [
    # Shared
    "Question",
    "load_questions",
    "generate_answer",
    # Text eval
    "TextCaseResult",
    "TextEvalResults",
    "run_text_eval",
    "print_text_summary",
    # Action eval
    "ActionCaseResult",
    "ActionEvalResults",
    "ActionJudgeResult",
    "judge_suggested_action",
    "run_action_eval",
    "print_action_summary",
]
