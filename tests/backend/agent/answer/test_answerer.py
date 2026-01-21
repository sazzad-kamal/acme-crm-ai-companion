"""Tests for answer/answerer.py extraction function."""

from backend.agent.answer.answerer import extract_suggested_action


class TestExtractSuggestedAction:
    """Tests for extract_suggested_action function."""

    def test_extracts_action_from_answer(self):
        """Extracts action and returns clean answer."""
        answer = (
            "Beta Tech has 3 open opportunities totaling $245,000.\n\n"
            "Suggested action: Schedule a call with Sarah Chen."
        )

        clean_answer, action = extract_suggested_action(answer)

        assert action == "Schedule a call with Sarah Chen."
        assert "Suggested action" not in clean_answer
        assert "Beta Tech has 3 open opportunities" in clean_answer

    def test_returns_none_when_no_action(self):
        """Returns None for action when no suggested action present."""
        answer = "Beta Tech has 3 open opportunities totaling $245,000."

        clean_answer, action = extract_suggested_action(answer)

        assert action is None
        assert clean_answer == answer

    def test_handles_case_insensitive(self):
        """Handles different casing of 'Suggested action'."""
        answer = "Some answer.\n\nSUGGESTED ACTION: Do something."

        clean_answer, action = extract_suggested_action(answer)

        assert action == "Do something."
        assert "SUGGESTED ACTION" not in clean_answer

    def test_strips_trailing_whitespace(self):
        """Strips trailing whitespace from clean answer."""
        answer = "Some answer.   \n\n\nSuggested action: Take action."

        clean_answer, action = extract_suggested_action(answer)

        assert action == "Take action."
        assert clean_answer == "Some answer."
        assert not clean_answer.endswith(" ")

    def test_handles_action_at_end_of_line(self):
        """Handles action at end of answer with no trailing newline."""
        answer = "Answer text.\n\nSuggested action: Call John."

        clean_answer, action = extract_suggested_action(answer)

        assert action == "Call John."
        assert clean_answer == "Answer text."

    def test_handles_empty_answer(self):
        """Handles empty answer gracefully."""
        clean_answer, action = extract_suggested_action("")

        assert clean_answer == ""
        assert action is None
