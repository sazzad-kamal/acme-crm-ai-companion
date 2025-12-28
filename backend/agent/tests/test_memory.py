"""
Unit tests for conversation memory module.

Tests the in-memory conversation history storage.

Run with: pytest backend/agent/tests/test_memory.py -v
"""

import pytest

from backend.agent.memory import (
    get_conversation_history,
    add_message,
    clear_session,
    get_last_company_context,
    format_history_for_prompt,
    MAX_MESSAGES_PER_SESSION,
    _memory_store,
)


@pytest.fixture(autouse=True)
def clean_memory():
    """Clear memory before and after each test."""
    _memory_store.clear()
    yield
    _memory_store.clear()


class TestAddMessage:
    """Tests for add_message function."""

    def test_add_user_message(self):
        """Test adding a user message."""
        add_message("session1", "user", "Hello", None)

        messages = get_conversation_history("session1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[0]["company_id"] is None

    def test_add_assistant_message_with_company(self):
        """Test adding an assistant message with company context."""
        add_message("session1", "assistant", "Here's info about Acme", "ACME-MFG")

        messages = get_conversation_history("session1")
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["company_id"] == "ACME-MFG"

    def test_add_multiple_messages(self):
        """Test adding multiple messages to same session."""
        add_message("session1", "user", "Tell me about Acme", None)
        add_message("session1", "assistant", "Acme is...", "ACME-MFG")
        add_message("session1", "user", "What about their contacts?", None)

        messages = get_conversation_history("session1")
        assert len(messages) == 3

    def test_add_message_none_session(self):
        """Test that None session is a no-op."""
        add_message(None, "user", "Hello", None)

        # Should have no sessions
        assert len(_memory_store) == 0

    def test_max_messages_trimming(self):
        """Test that messages are trimmed to MAX_MESSAGES_PER_SESSION."""
        session = "session_trim"

        # Add more than max messages
        for i in range(MAX_MESSAGES_PER_SESSION + 5):
            add_message(session, "user", f"Message {i}", None)

        messages = get_conversation_history(session)
        assert len(messages) == MAX_MESSAGES_PER_SESSION

        # Should have the most recent messages
        assert messages[-1]["content"] == f"Message {MAX_MESSAGES_PER_SESSION + 4}"


class TestGetConversationHistory:
    """Tests for get_conversation_history function."""

    def test_get_empty_history(self):
        """Test getting history for non-existent session."""
        messages = get_conversation_history("nonexistent")
        assert messages == []

    def test_get_history_none_session(self):
        """Test that None session returns empty list."""
        messages = get_conversation_history(None)
        assert messages == []

    def test_get_history_returns_copy(self):
        """Test that returned list is a copy, not the original."""
        add_message("session1", "user", "Hello", None)

        messages1 = get_conversation_history("session1")
        messages2 = get_conversation_history("session1")

        # Should be equal but not same object
        assert messages1 == messages2
        assert messages1 is not messages2


class TestClearSession:
    """Tests for clear_session function."""

    def test_clear_existing_session(self):
        """Test clearing an existing session."""
        add_message("session1", "user", "Hello", None)
        assert len(get_conversation_history("session1")) == 1

        clear_session("session1")
        assert len(get_conversation_history("session1")) == 0

    def test_clear_nonexistent_session(self):
        """Test clearing a non-existent session (should not error)."""
        clear_session("nonexistent")  # Should not raise

    def test_clear_none_session(self):
        """Test clearing None session (should be no-op)."""
        clear_session(None)  # Should not raise


class TestGetLastCompanyContext:
    """Tests for get_last_company_context function."""

    def test_get_last_company(self):
        """Test getting the last company context."""
        add_message("session1", "user", "Tell me about Acme", None)
        add_message("session1", "assistant", "Acme info...", "ACME-MFG")
        add_message("session1", "user", "What about their contacts?", None)

        company = get_last_company_context("session1")
        assert company == "ACME-MFG"

    def test_get_last_company_multiple_companies(self):
        """Test that most recent company is returned."""
        add_message("session1", "assistant", "Acme info...", "ACME-MFG")
        add_message("session1", "assistant", "Beta info...", "BETA-TECH")

        company = get_last_company_context("session1")
        assert company == "BETA-TECH"

    def test_get_last_company_no_company(self):
        """Test getting company when none mentioned."""
        add_message("session1", "user", "Hello", None)
        add_message("session1", "assistant", "Hi there", None)

        company = get_last_company_context("session1")
        assert company is None

    def test_get_last_company_empty_session(self):
        """Test getting company from empty session."""
        company = get_last_company_context("nonexistent")
        assert company is None

    def test_get_last_company_none_session(self):
        """Test getting company from None session."""
        company = get_last_company_context(None)
        assert company is None


class TestFormatHistoryForPrompt:
    """Tests for format_history_for_prompt function."""

    def test_format_empty_history(self):
        """Test formatting empty history."""
        result = format_history_for_prompt([])
        assert result == ""

    def test_format_none_history(self):
        """Test formatting None history."""
        result = format_history_for_prompt(None)
        assert result == ""

    def test_format_single_message(self):
        """Test formatting a single message."""
        messages = [{"role": "user", "content": "Hello", "company_id": None}]
        result = format_history_for_prompt(messages)
        assert "User: Hello" in result

    def test_format_conversation(self):
        """Test formatting a full conversation."""
        messages = [
            {"role": "user", "content": "Tell me about Acme", "company_id": None},
            {"role": "assistant", "content": "Acme is a manufacturing company", "company_id": "ACME-MFG"},
            {"role": "user", "content": "What's their status?", "company_id": None},
        ]
        result = format_history_for_prompt(messages)

        assert "User: Tell me about Acme" in result
        assert "Assistant: Acme is a manufacturing company" in result
        assert "User: What's their status?" in result

    def test_format_truncates_long_messages(self):
        """Test that long messages are truncated."""
        long_content = "A" * 300
        messages = [{"role": "user", "content": long_content, "company_id": None}]
        result = format_history_for_prompt(messages)

        assert "..." in result
        assert len(result) < len(long_content)

    def test_format_respects_max_messages(self):
        """Test that max_messages limit is respected."""
        messages = [
            {"role": "user", "content": f"Message {i}", "company_id": None}
            for i in range(10)
        ]
        result = format_history_for_prompt(messages, max_messages=3)

        # Should only include last 3 messages
        assert "Message 7" in result
        assert "Message 8" in result
        assert "Message 9" in result
        assert "Message 0" not in result


class TestMultipleSessionsIsolation:
    """Tests for session isolation."""

    def test_sessions_are_isolated(self):
        """Test that different sessions are isolated."""
        add_message("session1", "user", "Hello from 1", None)
        add_message("session2", "user", "Hello from 2", None)

        messages1 = get_conversation_history("session1")
        messages2 = get_conversation_history("session2")

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0]["content"] == "Hello from 1"
        assert messages2[0]["content"] == "Hello from 2"

    def test_clear_one_session_preserves_others(self):
        """Test that clearing one session doesn't affect others."""
        add_message("session1", "user", "Hello", None)
        add_message("session2", "user", "World", None)

        clear_session("session1")

        assert len(get_conversation_history("session1")) == 0
        assert len(get_conversation_history("session2")) == 1
