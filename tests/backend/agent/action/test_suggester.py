"""Tests for backend.agent.action.suggester module."""

from unittest.mock import patch

from backend.agent.action.suggester import call_action_chain


class TestCallActionChain:
    """Tests for call_action_chain function."""

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_action_when_present(self, mock_get_chain):
        """Returns action string when LLM suggests one."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = "Schedule a call with Sarah Chen"

        result = call_action_chain(question="What deals does Acme have?", answer="Acme has 3 deals.")

        assert result == "Schedule a call with Sarah Chen"

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_none_marker(self, mock_get_chain):
        """Returns None when LLM responds with NONE."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = "NONE"

        result = call_action_chain(question="How many deals?", answer="There are 5 deals.")

        assert result is None

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_none_marker_lowercase(self, mock_get_chain):
        """Returns None when LLM responds with none (case-insensitive)."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = "none"

        result = call_action_chain(question="What is the total?", answer="Total is $50k.")

        assert result is None

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_returns_none_when_empty(self, mock_get_chain):
        """Returns None when LLM returns empty string."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = ""

        result = call_action_chain(question="Test?", answer="Answer.")

        assert result is None

    @patch("backend.agent.action.suggester._get_action_chain")
    def test_strips_whitespace(self, mock_get_chain):
        """Strips whitespace from action."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = "  Review the pipeline  \n"

        result = call_action_chain(question="Pipeline?", answer="3 deals.")

        assert result == "Review the pipeline"
