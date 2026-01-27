"""Tests for backend.agent.action.suggester module."""

from unittest.mock import patch

from backend.agent.action.suggester import ActionSuggestion, call_action_chain


class TestCallActionChain:
    """Tests for call_action_chain function."""

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_action_when_contextual(self, mock_get_chain):
        """Returns action string when question_type is contextual."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ActionSuggestion(
            question_type="contextual", action="Schedule a call with Sarah Chen"
        )

        result = call_action_chain(question="What deals does Acme have?", answer="Acme has 3 deals.")

        assert result == "Schedule a call with Sarah Chen"

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_lookup(self, mock_get_chain):
        """Returns None when question_type is lookup (blocked)."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ActionSuggestion(
            question_type="lookup", action="Some action"
        )

        result = call_action_chain(question="How many deals?", answer="There are 5 deals.")

        assert result is None

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_aggregation(self, mock_get_chain):
        """Returns None when question_type is aggregation (blocked)."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ActionSuggestion(
            question_type="aggregation", action="Review pipeline"
        )

        result = call_action_chain(question="What is the total?", answer="Total is $50k.")

        assert result is None

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_action_empty(self, mock_get_chain):
        """Returns None when contextual but action is empty."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ActionSuggestion(
            question_type="contextual", action=""
        )

        result = call_action_chain(question="Test?", answer="Answer.")

        assert result is None
