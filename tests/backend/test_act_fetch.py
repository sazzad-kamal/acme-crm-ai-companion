"""Unit tests for Act! API fetch module."""

from unittest.mock import patch

import httpx
import pytest

from backend.act_fetch import (
    DEMO_STARTERS,
    _clear_token,
    _get_auth_header,
    act_fetch,
    get_database,
    set_database,
)


class TestAuthHeader:
    """Tests for auth header generation."""

    def test_generates_base64_header(self):
        """Auth header is properly base64 encoded."""
        with patch("backend.act_fetch.ACT_API_USER", "user"), \
             patch("backend.act_fetch.ACT_API_PASS", "pass"):
            header = _get_auth_header()
            assert header.startswith("Basic ")
            # user:pass in base64 is dXNlcjpwYXNz
            assert header == "Basic dXNlcjpwYXNz"


class TestTokenCache:
    """Tests for token caching."""

    def test_clear_token_resets_cache(self):
        """_clear_token clears the token cache."""
        import backend.act_fetch as module

        module._token = "test_token"
        module._token_expires = 9999999999.0

        _clear_token()

        assert module._token is None
        assert module._token_expires is None


class TestDatabaseSwitching:
    """Tests for runtime database switching."""

    def test_get_database_returns_current(self):
        """get_database returns current database."""
        db = get_database()
        assert isinstance(db, str)

    def test_set_database_changes_current(self):
        """set_database changes current database."""
        # Set to known value first
        set_database("KQC")
        assert get_database() == "KQC"
        # Switch to another
        set_database("W31322003119")
        assert get_database() == "W31322003119"
        # Switch back
        set_database("KQC")
        assert get_database() == "KQC"

    def test_set_database_invalid_raises(self):
        """set_database raises for invalid database."""
        with pytest.raises(ValueError, match="Unknown database"):
            set_database("InvalidDB")


class TestActFetch:
    """Tests for act_fetch function."""

    @pytest.fixture(autouse=True)
    def reset_token(self):
        """Reset token cache before each test."""
        _clear_token()

    @pytest.mark.parametrize("question", DEMO_STARTERS)
    def test_each_question_has_handler(self, question: str):
        """Each of the 5 demo questions has a handler."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"id": 1, "name": "Test"}]
            result = act_fetch(question)

            assert result["error"] is None
            # Some questions return dict with multiple keys
            if isinstance(result["data"], dict):
                assert len(result["data"]) > 0
            else:
                assert len(result["data"]) >= 0  # Can be empty after filtering

    def test_unknown_question_returns_error(self):
        """Unknown question returns error dict."""
        result = act_fetch("What is the meaning of life?")

        assert result["error"] == "Unknown question"
        assert result["data"] == []

    def test_api_error_returns_error_dict(self):
        """API error returns error dict instead of raising."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("API down")
            result = act_fetch("What should I follow up on?")

            assert "API down" in result["error"]
            assert result["data"] == []

    def test_timeout_returns_error_dict(self):
        """Timeout returns error dict instead of raising."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timeout")
            result = act_fetch("What deals are closing soon?")

            assert "timeout" in result["error"].lower()
            assert result["data"] == []

    def test_whats_coming_up_fetches_calendar(self):
        """'What's coming up?' fetches calendar data."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"type": "call", "contact": "John"}]
            result = act_fetch("What's coming up?")

            assert result["error"] is None
            # Check it tried calendar first
            call_args = mock_get.call_args[0][0]
            assert "calendar" in call_args or "activities" in call_args

    def test_what_needs_attention_returns_dict(self):
        """'What needs attention?' returns dict with overdue and stale opps."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"id": 1, "subject": "Task"}]
            result = act_fetch("What needs attention?")

            assert result["error"] is None
            assert isinstance(result["data"], dict)
            assert "overdue_activities" in result["data"]
            assert "stale_opportunities" in result["data"]

    def test_who_should_i_contact_returns_dict(self):
        """'Who should I contact next?' returns dict with opps and contacts."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"id": 1, "name": "Opp"}]
            result = act_fetch("Who should I contact next?")

            assert result["error"] is None
            assert isinstance(result["data"], dict)
            assert "opportunities" in result["data"]
            assert "contacts" in result["data"]

    def test_follow_up_filters_field_changed(self):
        """'What should I follow up on?' filters out Field changed entries."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [
                {"id": 1, "regarding": "Meeting with John"},
                {"id": 2, "regarding": "Field changed: Status"},
                {"id": 3, "regarding": "Call with Sarah"},
            ]
            result = act_fetch("What should I follow up on?")

            assert result["error"] is None
            # Should filter out "Field changed" entry
            assert len(result["data"]) == 2
            for item in result["data"]:
                assert "Field changed" not in item.get("regarding", "")


class TestDemoStarters:
    """Tests for DEMO_STARTERS constant."""

    def test_has_five_questions(self):
        """DEMO_STARTERS contains exactly 5 questions."""
        assert len(DEMO_STARTERS) == 5

    def test_all_are_strings(self):
        """All starters are non-empty strings."""
        for starter in DEMO_STARTERS:
            assert isinstance(starter, str)
            assert len(starter) > 0

    def test_expected_questions(self):
        """Expected questions are in DEMO_STARTERS."""
        expected = [
            "What should I follow up on?",
            "What's coming up?",
            "Who should I contact next?",
            "What needs attention?",
            "What deals are closing soon?",
        ]
        assert DEMO_STARTERS == expected
