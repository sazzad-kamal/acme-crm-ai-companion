"""
Input validation and sanitization utilities.

Provides centralized validation functions for:
- String length and content validation
- SQL injection pattern detection
- XSS prevention via HTML escaping
- Common field type validation (emails, IDs, etc.)
- Request payload validation helpers

Usage:
    from backend.common.validation import (
        validate_string_length,
        sanitize_string,
        validate_company_id,
        check_for_injection,
    )
"""

import html
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Maximum lengths for common field types
MAX_QUESTION_LENGTH = 2000
MAX_COMPANY_ID_LENGTH = 100
MAX_SESSION_ID_LENGTH = 100
MAX_USER_ID_LENGTH = 100
MAX_MODE_LENGTH = 20

# Patterns that might indicate SQL injection attempts
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
    r"(--|;|/\*|\*/)",
    r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
    r"(\b(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
    r"(EXEC\s*\(|EXECUTE\s*\()",
]

# Patterns that might indicate XSS attempts
XSS_PATTERNS = [
    r"<script\b[^>]*>",
    r"javascript:",
    r"on\w+\s*=",
    r"data:\s*text/html",
]

# Valid mode values
VALID_MODES = {"auto", "data", "docs", "data+docs"}


# =============================================================================
# String Validation
# =============================================================================

def validate_string_length(
    value: str,
    field_name: str,
    max_length: int,
    min_length: int = 0,
) -> tuple[bool, str]:
    """
    Validate string length is within bounds.

    Args:
        value: The string to validate
        field_name: Name of the field (for error messages)
        max_length: Maximum allowed length
        min_length: Minimum required length (default 0)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, str):
        return False, f"{field_name} must be a string"

    if len(value) < min_length:
        return False, f"{field_name} must be at least {min_length} characters"

    if len(value) > max_length:
        return False, f"{field_name} exceeds maximum length of {max_length} characters"

    return True, ""


def validate_non_empty(value: str | None, field_name: str) -> tuple[bool, str]:
    """
    Validate that a string is not empty or whitespace-only.

    Args:
        value: The string to validate
        field_name: Name of the field (for error messages)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if value is None or not value.strip():
        return False, f"{field_name} cannot be empty"
    return True, ""


def sanitize_string(value: str, strip_html: bool = True) -> str:
    """
    Sanitize a string for safe use.

    Args:
        value: The string to sanitize
        strip_html: Whether to HTML-escape the string

    Returns:
        Sanitized string
    """
    if not value:
        return value

    # Strip leading/trailing whitespace
    result = value.strip()

    # HTML escape if requested
    if strip_html:
        result = html.escape(result)

    return result


def truncate_string(value: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Args:
        value: The string to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to append when truncated

    Returns:
        Truncated string
    """
    if not value or len(value) <= max_length:
        return value

    return value[:max_length - len(suffix)] + suffix


# =============================================================================
# Security Checks
# =============================================================================

def check_for_injection(value: str) -> tuple[bool, str]:
    """
    Check if a string contains potential SQL injection patterns.

    Args:
        value: The string to check

    Returns:
        Tuple of (is_safe, warning_message)
    """
    if not value:
        return True, ""

    value_upper = value.upper()

    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value_upper, re.IGNORECASE):
            logger.warning(
                f"Potential SQL injection pattern detected",
                extra={"pattern": pattern, "value_preview": value[:50]},
            )
            return False, "Input contains potentially unsafe patterns"

    return True, ""


def check_for_xss(value: str) -> tuple[bool, str]:
    """
    Check if a string contains potential XSS patterns.

    Args:
        value: The string to check

    Returns:
        Tuple of (is_safe, warning_message)
    """
    if not value:
        return True, ""

    for pattern in XSS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            logger.warning(
                f"Potential XSS pattern detected",
                extra={"pattern": pattern, "value_preview": value[:50]},
            )
            return False, "Input contains potentially unsafe HTML/script patterns"

    return True, ""


def is_safe_input(value: str) -> bool:
    """
    Quick check if input appears safe (no injection/XSS patterns).

    Args:
        value: The string to check

    Returns:
        True if the input appears safe
    """
    is_sql_safe, _ = check_for_injection(value)
    is_xss_safe, _ = check_for_xss(value)
    return is_sql_safe and is_xss_safe


# =============================================================================
# Field-Specific Validation
# =============================================================================

def validate_question(question: str | None) -> tuple[bool, str]:
    """
    Validate a chat question.

    Args:
        question: The question to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check non-empty
    is_valid, error = validate_non_empty(question, "Question")
    if not is_valid:
        return False, error

    # Check length
    is_valid, error = validate_string_length(
        question, "Question", MAX_QUESTION_LENGTH, min_length=1
    )
    if not is_valid:
        return False, error

    return True, ""


def validate_company_id(company_id: str | None) -> tuple[bool, str]:
    """
    Validate a company ID.

    Args:
        company_id: The company ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if company_id is None:
        return True, ""  # Optional field

    # Check length
    is_valid, error = validate_string_length(
        company_id, "Company ID", MAX_COMPANY_ID_LENGTH
    )
    if not is_valid:
        return False, error

    # Check for injection
    is_safe, error = check_for_injection(company_id)
    if not is_safe:
        return False, error

    return True, ""


def validate_mode(mode: str | None) -> tuple[bool, str]:
    """
    Validate a mode value.

    Args:
        mode: The mode to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if mode is None:
        return True, ""  # Optional field (defaults to 'auto')

    if mode not in VALID_MODES:
        return False, f"Invalid mode: {mode}. Must be one of: {', '.join(VALID_MODES)}"

    return True, ""


def validate_session_id(session_id: str | None) -> tuple[bool, str]:
    """
    Validate a session ID.

    Args:
        session_id: The session ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if session_id is None:
        return True, ""  # Optional field

    return validate_string_length(session_id, "Session ID", MAX_SESSION_ID_LENGTH)


def validate_user_id(user_id: str | None) -> tuple[bool, str]:
    """
    Validate a user ID.

    Args:
        user_id: The user ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if user_id is None:
        return True, ""  # Optional field

    return validate_string_length(user_id, "User ID", MAX_USER_ID_LENGTH)


# =============================================================================
# Request Validation
# =============================================================================

def validate_chat_request(
    question: str | None,
    mode: str | None = None,
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate all fields in a chat request.

    Args:
        question: The question to ask
        mode: Optional mode (auto, data, docs, data+docs)
        company_id: Optional company ID
        session_id: Optional session ID
        user_id: Optional user ID

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Validate question
    is_valid, error = validate_question(question)
    if not is_valid:
        errors.append(error)

    # Validate mode
    is_valid, error = validate_mode(mode)
    if not is_valid:
        errors.append(error)

    # Validate company_id
    is_valid, error = validate_company_id(company_id)
    if not is_valid:
        errors.append(error)

    # Validate session_id
    is_valid, error = validate_session_id(session_id)
    if not is_valid:
        errors.append(error)

    # Validate user_id
    is_valid, error = validate_user_id(user_id)
    if not is_valid:
        errors.append(error)

    return len(errors) == 0, errors


__all__ = [
    # Constants
    "MAX_QUESTION_LENGTH",
    "MAX_COMPANY_ID_LENGTH",
    "VALID_MODES",
    # String validation
    "validate_string_length",
    "validate_non_empty",
    "sanitize_string",
    "truncate_string",
    # Security checks
    "check_for_injection",
    "check_for_xss",
    "is_safe_input",
    # Field validation
    "validate_question",
    "validate_company_id",
    "validate_mode",
    "validate_session_id",
    "validate_user_id",
    # Request validation
    "validate_chat_request",
]
