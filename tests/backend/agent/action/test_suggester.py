"""Tests for backend.agent.action.suggester module."""

from unittest.mock import patch

from backend.agent.action.suggester import ActionSuggestion, call_action_chain


class TestCallActionChain:
    """Tests for call_action_chain function."""

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_action_when_should_suggest(self, mock_get_chain):
        """Returns action string when should_suggest is True."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ActionSuggestion(
            should_suggest=True, action="Schedule a call with Sarah Chen"
        )

        result = call_action_chain(question="What deals does Acme have?", answer="Acme has 3 deals.")

        assert result == "Schedule a call with Sarah Chen"

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_should_not_suggest(self, mock_get_chain):
        """Returns None when should_suggest is False."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ActionSuggestion(
            should_suggest=False, action=""
        )

        result = call_action_chain(question="How many deals?", answer="There are 5 deals.")

        assert result is None

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_action_empty(self, mock_get_chain):
        """Returns None when should_suggest is True but action is empty."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ActionSuggestion(
            should_suggest=True, action=""
        )

        result = call_action_chain(question="Test?", answer="Answer.")

        assert result is None
