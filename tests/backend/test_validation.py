"""
Tests for the validation module.
"""

import pytest

from backend.common.validation import (
    MAX_QUESTION_LENGTH,
    MAX_COMPANY_ID_LENGTH,
    VALID_MODES,
    validate_string_length,
    validate_non_empty,
    sanitize_string,
    truncate_string,
    check_for_injection,
    check_for_xss,
    is_safe_input,
    validate_question,
    validate_company_id,
    validate_mode,
    validate_session_id,
    validate_user_id,
    validate_chat_request,
)


# =============================================================================
# String Validation Tests
# =============================================================================

class TestValidateStringLength:
    """Tests for validate_string_length function."""

    def test_valid_length_returns_true(self):
        is_valid, error = validate_string_length("hello", "field", 10)
        assert is_valid is True
        assert error == ""

    def test_exceeds_max_returns_false(self):
        is_valid, error = validate_string_length("hello world", "field", 5)
        assert is_valid is False
        assert "exceeds maximum length" in error

    def test_below_min_returns_false(self):
        is_valid, error = validate_string_length("hi", "field", 10, min_length=5)
        assert is_valid is False
        assert "at least 5 characters" in error

    def test_non_string_returns_false(self):
        is_valid, error = validate_string_length(123, "field", 10)  # type: ignore
        assert is_valid is False
        assert "must be a string" in error

    def test_empty_string_with_zero_min(self):
        is_valid, error = validate_string_length("", "field", 10, min_length=0)
        assert is_valid is True

    def test_exact_max_length_is_valid(self):
        is_valid, error = validate_string_length("hello", "field", 5)
        assert is_valid is True


class TestValidateNonEmpty:
    """Tests for validate_non_empty function."""

    def test_non_empty_string_is_valid(self):
        is_valid, error = validate_non_empty("hello", "field")
        assert is_valid is True

    def test_empty_string_is_invalid(self):
        is_valid, error = validate_non_empty("", "field")
        assert is_valid is False
        assert "cannot be empty" in error

    def test_whitespace_only_is_invalid(self):
        is_valid, error = validate_non_empty("   ", "field")
        assert is_valid is False

    def test_none_is_invalid(self):
        is_valid, error = validate_non_empty(None, "field")
        assert is_valid is False


class TestSanitizeString:
    """Tests for sanitize_string function."""

    def test_strips_whitespace(self):
        result = sanitize_string("  hello  ")
        assert result == "hello"

    def test_escapes_html(self):
        result = sanitize_string("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_no_html_escape_when_disabled(self):
        result = sanitize_string("<b>bold</b>", strip_html=False)
        assert "<b>bold</b>" in result

    def test_empty_string_unchanged(self):
        result = sanitize_string("")
        assert result == ""

    def test_none_unchanged(self):
        result = sanitize_string(None)  # type: ignore
        assert result is None


class TestTruncateString:
    """Tests for truncate_string function."""

    def test_short_string_unchanged(self):
        result = truncate_string("hello", 10)
        assert result == "hello"

    def test_long_string_truncated(self):
        result = truncate_string("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_custom_suffix(self):
        result = truncate_string("hello world", 8, suffix="…")
        assert result == "hello w…"

    def test_empty_string_unchanged(self):
        result = truncate_string("", 10)
        assert result == ""


# =============================================================================
# Security Check Tests
# =============================================================================

class TestCheckForInjection:
    """Tests for check_for_injection function."""

    def test_normal_text_is_safe(self):
        is_safe, error = check_for_injection("What is the status of Acme Corp?")
        assert is_safe is True

    def test_select_statement_is_unsafe(self):
        is_safe, error = check_for_injection("SELECT * FROM users")
        assert is_safe is False
        assert "unsafe patterns" in error

    def test_drop_statement_is_unsafe(self):
        is_safe, error = check_for_injection("DROP TABLE users")
        assert is_safe is False

    def test_union_injection_is_unsafe(self):
        is_safe, error = check_for_injection("' UNION SELECT * FROM users --")
        assert is_safe is False

    def test_or_1_equals_1_is_unsafe(self):
        is_safe, error = check_for_injection("' OR 1=1 --")
        assert is_safe is False

    def test_empty_string_is_safe(self):
        is_safe, error = check_for_injection("")
        assert is_safe is True


class TestCheckForXss:
    """Tests for check_for_xss function."""

    def test_normal_text_is_safe(self):
        is_safe, error = check_for_xss("What is the status of Acme Corp?")
        assert is_safe is True

    def test_script_tag_is_unsafe(self):
        is_safe, error = check_for_xss("<script>alert('xss')</script>")
        assert is_safe is False

    def test_javascript_url_is_unsafe(self):
        is_safe, error = check_for_xss("javascript:alert('xss')")
        assert is_safe is False

    def test_event_handler_is_unsafe(self):
        is_safe, error = check_for_xss('<img src="x" onerror="alert(1)">')
        assert is_safe is False

    def test_empty_string_is_safe(self):
        is_safe, error = check_for_xss("")
        assert is_safe is True


class TestIsSafeInput:
    """Tests for is_safe_input function."""

    def test_normal_text_is_safe(self):
        assert is_safe_input("What is the status of Acme Corp?") is True

    def test_sql_injection_is_unsafe(self):
        assert is_safe_input("SELECT * FROM users") is False

    def test_xss_is_unsafe(self):
        assert is_safe_input("<script>alert('xss')</script>") is False


# =============================================================================
# Field Validation Tests
# =============================================================================

class TestValidateQuestion:
    """Tests for validate_question function."""

    def test_valid_question(self):
        is_valid, error = validate_question("What is the status of Acme Corp?")
        assert is_valid is True

    def test_empty_question_is_invalid(self):
        is_valid, error = validate_question("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_none_question_is_invalid(self):
        is_valid, error = validate_question(None)
        assert is_valid is False

    def test_too_long_question_is_invalid(self):
        long_question = "x" * (MAX_QUESTION_LENGTH + 1)
        is_valid, error = validate_question(long_question)
        assert is_valid is False
        assert "maximum length" in error


class TestValidateCompanyId:
    """Tests for validate_company_id function."""

    def test_none_is_valid(self):
        is_valid, error = validate_company_id(None)
        assert is_valid is True

    def test_valid_company_id(self):
        is_valid, error = validate_company_id("acme-corp-123")
        assert is_valid is True

    def test_too_long_is_invalid(self):
        long_id = "x" * (MAX_COMPANY_ID_LENGTH + 1)
        is_valid, error = validate_company_id(long_id)
        assert is_valid is False

    def test_injection_attempt_is_invalid(self):
        is_valid, error = validate_company_id("'; DROP TABLE companies; --")
        assert is_valid is False


class TestValidateMode:
    """Tests for validate_mode function."""

    def test_none_is_valid(self):
        is_valid, error = validate_mode(None)
        assert is_valid is True

    def test_valid_modes(self):
        for mode in VALID_MODES:
            is_valid, error = validate_mode(mode)
            assert is_valid is True, f"Mode {mode} should be valid"

    def test_invalid_mode(self):
        is_valid, error = validate_mode("invalid")
        assert is_valid is False
        assert "Invalid mode" in error


class TestValidateSessionId:
    """Tests for validate_session_id function."""

    def test_none_is_valid(self):
        is_valid, error = validate_session_id(None)
        assert is_valid is True

    def test_valid_session_id(self):
        is_valid, error = validate_session_id("session-123-abc")
        assert is_valid is True

    def test_too_long_is_invalid(self):
        long_id = "x" * 101
        is_valid, error = validate_session_id(long_id)
        assert is_valid is False


class TestValidateUserId:
    """Tests for validate_user_id function."""

    def test_none_is_valid(self):
        is_valid, error = validate_user_id(None)
        assert is_valid is True

    def test_valid_user_id(self):
        is_valid, error = validate_user_id("user-123")
        assert is_valid is True


# =============================================================================
# Request Validation Tests
# =============================================================================

class TestValidateChatRequest:
    """Tests for validate_chat_request function."""

    def test_valid_minimal_request(self):
        is_valid, errors = validate_chat_request(question="What is Acme?")
        assert is_valid is True
        assert len(errors) == 0

    def test_valid_full_request(self):
        is_valid, errors = validate_chat_request(
            question="What is Acme?",
            mode="data",
            company_id="acme-123",
            session_id="sess-123",
            user_id="user-123",
        )
        assert is_valid is True

    def test_empty_question_is_invalid(self):
        is_valid, errors = validate_chat_request(question="")
        assert is_valid is False
        assert len(errors) > 0

    def test_invalid_mode_returns_error(self):
        is_valid, errors = validate_chat_request(
            question="What is Acme?",
            mode="invalid",
        )
        assert is_valid is False
        assert any("mode" in e.lower() for e in errors)

    def test_multiple_errors_returned(self):
        is_valid, errors = validate_chat_request(
            question="",
            mode="invalid",
        )
        assert is_valid is False
        assert len(errors) >= 2
