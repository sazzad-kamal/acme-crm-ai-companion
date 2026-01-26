"""Tests for backend.eval.answer.shared.loader module."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestLoadQuestions:
    """Tests for load_questions function."""

    def test_load_questions_all(self, monkeypatch, tmp_path):
        """Test loading all questions."""
        yaml_content = """
questions:
  - text: "Question 1"
    difficulty: 1
    expected_sql: "SELECT 1"
    expected_answer: "Answer 1"
  - text: "Question 2"
    difficulty: 2
    expected_sql: "SELECT 2"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.answer.shared.loader as loader_module

        monkeypatch.setattr(loader_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.answer.shared.loader import load_questions

        questions = load_questions()
        assert len(questions) == 2
        assert questions[0].text == "Question 1"
        assert questions[0].expected_sql == "SELECT 1"
        assert questions[0].expected_answer == "Answer 1"
        assert questions[1].expected_answer == ""  # Default


class TestGenerateAnswer:
    """Tests for generate_answer function."""

    def test_generate_answer_success(self, monkeypatch):
        """Test successful answer generation."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_answer
        from backend.eval.answer.shared.models import Question

        # Mock execute_sql
        monkeypatch.setattr(
            loader_module,
            "execute_sql",
            MagicMock(return_value=([{"value": 1}], None)),
        )

        # Mock call_answer_chain
        monkeypatch.setattr(
            loader_module,
            "call_answer_chain",
            MagicMock(return_value="The answer is 1."),
        )

        q = Question(text="Test", difficulty=1, expected_sql="SELECT 1")
        mock_conn = MagicMock()

        answer, results, error = generate_answer(q, mock_conn)

        assert answer == "The answer is 1."
        assert results == [{"value": 1}]
        assert error is None

    def test_generate_answer_sql_error(self, monkeypatch):
        """Test SQL error handling."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_answer
        from backend.eval.answer.shared.models import Question

        # Mock execute_sql to return error
        monkeypatch.setattr(
            loader_module,
            "execute_sql",
            MagicMock(return_value=([], "SQL syntax error")),
        )

        q = Question(text="Test", difficulty=1, expected_sql="INVALID SQL")
        mock_conn = MagicMock()

        answer, results, error = generate_answer(q, mock_conn)

        assert answer == ""
        assert results == []
        assert "SQL error" in error

    def test_generate_answer_exception(self, monkeypatch):
        """Test exception handling."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_answer
        from backend.eval.answer.shared.models import Question

        # Mock execute_sql to raise exception
        monkeypatch.setattr(
            loader_module,
            "execute_sql",
            MagicMock(side_effect=Exception("Connection failed")),
        )

        q = Question(text="Test", difficulty=1, expected_sql="SELECT 1")
        mock_conn = MagicMock()

        answer, results, error = generate_answer(q, mock_conn)

        assert answer == ""
        assert results == []
        assert "Error:" in error


class TestGenerateAction:
    """Tests for generate_action function."""

    def test_generate_action_success(self, monkeypatch):
        """Test successful action generation."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_action

        monkeypatch.setattr(
            loader_module,
            "call_action_chain",
            MagicMock(return_value="Schedule a call with Sarah Chen"),
        )

        action, error = generate_action("What deals?", "Acme has 3 deals.")

        assert action == "Schedule a call with Sarah Chen"
        assert error is None

    def test_generate_action_none(self, monkeypatch):
        """Test when no action suggested."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_action

        monkeypatch.setattr(
            loader_module,
            "call_action_chain",
            MagicMock(return_value=None),
        )

        action, error = generate_action("How many deals?", "There are 5 deals.")

        assert action is None
        assert error is None

    def test_generate_action_exception(self, monkeypatch):
        """Test exception handling."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_action

        monkeypatch.setattr(
            loader_module,
            "call_action_chain",
            MagicMock(side_effect=Exception("Chain failed")),
        )

        action, error = generate_action("Test?", "Answer.")

        assert action is None
        assert "Action error:" in error
