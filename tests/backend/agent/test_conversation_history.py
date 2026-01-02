"""
Integration tests for conversation history in the agent pipeline.

Tests that conversation history flows correctly through the agent.

Run with: pytest backend/agent/tests/test_conversation_history.py -v
"""

import pytest

from backend.agent.state import AgentState, Message
from backend.agent.memory import (
    add_message,
    get_conversation_history,
    clear_session,
    _memory_store,
)
from backend.agent.formatters import format_conversation_history_section


@pytest.fixture(autouse=True)
def clean_memory():
    """Clear memory before and after each test."""
    _memory_store.clear()
    yield
    _memory_store.clear()


class TestMessageType:
    """Tests for the Message TypedDict."""

    def test_message_structure(self):
        """Test that Message has the expected structure."""
        msg: Message = {
            "role": "user",
            "content": "Hello",
            "company_id": "ACME-MFG",
        }
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        assert msg["company_id"] == "ACME-MFG"

    def test_message_optional_company(self):
        """Test that company_id can be None."""
        msg: Message = {
            "role": "assistant",
            "content": "Hi there",
            "company_id": None,
        }
        assert msg["company_id"] is None


class TestAgentStateWithMessages:
    """Tests for AgentState with messages field."""

    def test_state_with_empty_messages(self):
        """Test creating state with empty messages."""
        state: AgentState = {
            "question": "Hello",
            "mode": "auto",
            "messages": [],
        }
        assert state["messages"] == []

    def test_state_with_messages(self):
        """Test creating state with conversation history."""
        messages: list[Message] = [
            {"role": "user", "content": "Tell me about Acme", "company_id": None},
            {"role": "assistant", "content": "Acme is...", "company_id": "ACME-MFG"},
        ]
        state: AgentState = {
            "question": "What about their contacts?",
            "mode": "auto",
            "session_id": "test_session",
            "messages": messages,
        }
        assert len(state["messages"]) == 2
        assert state["messages"][1]["company_id"] == "ACME-MFG"


class TestFormatConversationHistorySection:
    """Tests for the conversation history formatter."""

    def test_format_empty(self):
        """Test formatting empty history."""
        result = format_conversation_history_section(None)
        assert result == ""

        result = format_conversation_history_section([])
        assert result == ""

    def test_format_single_turn(self):
        """Test formatting a single turn."""
        messages = [
            {"role": "user", "content": "What is Acme's status?", "company_id": None},
        ]
        result = format_conversation_history_section(messages)

        assert "=== RECENT CONVERSATION ===" in result
        assert "User: What is Acme's status?" in result

    def test_format_multi_turn(self):
        """Test formatting multiple turns."""
        messages = [
            {"role": "user", "content": "Tell me about Acme", "company_id": None},
            {"role": "assistant", "content": "Acme Manufacturing is a mid-market account", "company_id": "ACME-MFG"},
            {"role": "user", "content": "What about their contacts?", "company_id": None},
        ]
        result = format_conversation_history_section(messages)

        assert "User: Tell me about Acme" in result
        assert "Assistant: Acme Manufacturing is a mid-market account" in result
        assert "User: What about their contacts?" in result

    def test_format_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "A" * 200
        messages = [
            {"role": "assistant", "content": long_content, "company_id": None},
        ]
        result = format_conversation_history_section(messages)

        assert "..." in result
        # Should be truncated to ~150 chars + "..."
        assert "A" * 150 in result

    def test_format_respects_max_messages(self):
        """Test that only recent messages are included."""
        messages = [
            {"role": "user", "content": f"Question {i}", "company_id": None}
            for i in range(10)
        ]
        result = format_conversation_history_section(messages, max_messages=3)

        # Should only have the last 3
        assert "Question 7" in result
        assert "Question 8" in result
        assert "Question 9" in result
        assert "Question 0" not in result
        assert "Question 6" not in result


class TestConversationHistoryFlow:
    """Integration tests for conversation history in the pipeline."""

    def test_session_persistence(self):
        """Test that messages persist across calls for same session."""
        session_id = "test_persistence"

        # Simulate first turn
        add_message(session_id, "user", "Tell me about Acme", None)
        add_message(session_id, "assistant", "Acme is a manufacturing company", "ACME-MFG")

        # Verify history is available for second turn
        history = get_conversation_history(session_id)
        assert len(history) == 2
        assert history[0]["content"] == "Tell me about Acme"
        assert history[1]["company_id"] == "ACME-MFG"

        # Simulate second turn
        add_message(session_id, "user", "What about their contacts?", None)
        add_message(session_id, "assistant", "Here are the contacts...", "ACME-MFG")

        # Full history should be available
        history = get_conversation_history(session_id)
        assert len(history) == 4

    def test_session_isolation(self):
        """Test that different sessions are isolated."""
        add_message("session_a", "user", "About Acme", None)
        add_message("session_a", "assistant", "Acme info", "ACME-MFG")

        add_message("session_b", "user", "About Beta", None)
        add_message("session_b", "assistant", "Beta info", "BETA-TECH")

        history_a = get_conversation_history("session_a")
        history_b = get_conversation_history("session_b")

        assert len(history_a) == 2
        assert len(history_b) == 2
        assert history_a[1]["company_id"] == "ACME-MFG"
        assert history_b[1]["company_id"] == "BETA-TECH"

    def test_clear_session_between_eval_runs(self):
        """Test that sessions can be cleared for fresh eval runs."""
        session_id = "eval_session_1"

        # Add some history
        add_message(session_id, "user", "Old question", None)
        add_message(session_id, "assistant", "Old answer", None)
        assert len(get_conversation_history(session_id)) == 2

        # Clear for fresh run
        clear_session(session_id)
        assert len(get_conversation_history(session_id)) == 0

        # New history should start fresh
        add_message(session_id, "user", "New question", None)
        history = get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0]["content"] == "New question"


class TestPronounResolutionContext:
    """Tests for pronoun resolution via conversation history."""

    def test_company_context_for_pronoun_resolution(self):
        """Test that company context is available for pronoun resolution."""
        session_id = "pronoun_test"

        # First turn establishes context
        add_message(session_id, "user", "Tell me about Acme Manufacturing", None)
        add_message(session_id, "assistant", "Acme Manufacturing is...", "ACME-MFG")

        # Get history for pronoun resolution
        history = get_conversation_history(session_id)
        formatted = format_conversation_history_section(history)

        # The formatted history should contain the company context
        # which the router can use to resolve "their" in "What about their contacts?"
        assert "Acme Manufacturing" in formatted
        assert "User:" in formatted
        assert "Assistant:" in formatted

