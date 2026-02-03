"""Unit tests for Act! API fetch module."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.act_fetch import (
    DEMO_STARTERS,
    _clear_token,
    _get_auth_header,
    act_fetch,
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
            assert len(result["data"]) > 0
            mock_get.assert_called_once()

    def test_unknown_question_returns_error(self):
        """Unknown question returns error dict."""
        result = act_fetch("What is the meaning of life?")

        assert result["error"] == "Unknown question"
        assert result["data"] == []

    def test_api_error_returns_error_dict(self):
        """API error returns error dict instead of raising."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("API down")
            result = act_fetch("Brief me on my next call")

            assert "API down" in result["error"]
            assert result["data"] == []

    def test_timeout_returns_error_dict(self):
        """Timeout returns error dict instead of raising."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timeout")
            result = act_fetch("What should I focus on today?")

            assert "timeout" in result["error"].lower()
            assert result["data"] == []

    def test_brief_me_fetches_calendar(self):
        """'Brief me on my next call' fetches calendar data."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"type": "call", "contact": "John"}]
            result = act_fetch("Brief me on my next call")

            assert result["error"] is None
            assert len(result["data"]) == 1
            # Check it tried calendar first
            call_args = mock_get.call_args[0][0]
            assert "calendar" in call_args or "activities" in call_args

    def test_whats_urgent_fetches_high_priority(self):
        """'What's urgent?' fetches high priority items."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"priority": "High", "task": "Review contract"}]
            result = act_fetch("What's urgent?")

            assert result["error"] is None
            # Check filter includes priority
            call_kwargs = mock_get.call_args[1]
            if call_kwargs and "$filter" in call_kwargs.get("params", {}):
                assert "priority" in call_kwargs["params"]["$filter"].lower() or \
                       "overdue" in call_kwargs["params"]["$filter"].lower()

    def test_empty_data_returns_empty_list(self):
        """Empty API response returns empty data list."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = []
            result = act_fetch("Catch me up")

            assert result["error"] is None
            assert result["data"] == []


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
            "Brief me on my next call",
            "What should I focus on today?",
            "Who should I contact next?",
            "What's urgent?",
            "Catch me up",
        ]
        assert DEMO_STARTERS == expected
