"""Tests for backend.eval.answer.shared.models module."""

from __future__ import annotations

from backend.eval.answer.shared.models import Question


class TestQuestion:
    """Tests for Question dataclass."""

    def test_question_basic(self):
        """Test basic Question creation."""
        q = Question(text="What is the status?", difficulty=1, expected_sql="SELECT status FROM companies")
        assert q.text == "What is the status?"
        assert q.difficulty == 1
        assert q.expected_sql == "SELECT status FROM companies"

    def test_question_with_expected_answer(self):
        """Test Question with expected_answer field."""
        q = Question(
            text="What is the plan?",
            difficulty=1,
            expected_sql="SELECT plan FROM companies",
            expected_answer="The plan is Enterprise.",
        )
        assert q.expected_answer == "The plan is Enterprise."

    def test_question_expected_answer_optional(self):
        """Test expected_answer is optional."""
        q = Question(text="Test", difficulty=1, expected_sql="SELECT 1")
        assert q.expected_answer == ""

    def test_question_difficulty_default(self):
        """Test difficulty defaults to 1."""
        q = Question(text="Test", expected_sql="SELECT 1")
        assert q.difficulty == 1

    def test_question_expected_action_default(self):
        """Test expected_action defaults to False."""
        q = Question(text="Test", expected_sql="SELECT 1")
        assert q.expected_action is False

    def test_question_expected_action_true(self):
        """Test Question with expected_action=True."""
        q = Question(text="Notes about Anna", expected_sql="SELECT notes", expected_action=True)
        assert q.expected_action is True
