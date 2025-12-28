"""
Tests for backend/common/query_ops.py - shared query operations.
"""

import pytest
from unittest.mock import patch

from backend.common.query_ops import (
    rewrite_query,
    generate_hyde,
    QUERY_REWRITE_SYSTEM,
    HYDE_SYSTEM,
)


# =============================================================================
# rewrite_query Tests
# =============================================================================

class TestRewriteQuery:
    """Tests for rewrite_query function."""

    @patch('backend.common.query_ops.call_llm_safe')
    def test_returns_rewritten_query(self, mock_llm):
        """Returns LLM-rewritten query."""
        mock_llm.return_value = "What is the process for importing contacts into Acme CRM?"

        result = rewrite_query("how do i import contacts")

        assert result == "What is the process for importing contacts into Acme CRM?"

    @patch('backend.common.query_ops.call_llm_safe')
    def test_returns_original_on_failure(self, mock_llm):
        """Returns original query when LLM returns default."""
        original = "my question"
        mock_llm.return_value = original

        result = rewrite_query(original)

        assert result == original

    @patch('backend.common.query_ops.call_llm_safe')
    def test_includes_company_context(self, mock_llm):
        """Includes company name in prompt when provided."""
        mock_llm.return_value = "rewritten"

        rewrite_query("what's happening", company_name="Acme Corp")

        call_args = mock_llm.call_args
        prompt = call_args[1]["prompt"]
        assert "Acme Corp" in prompt

    @patch('backend.common.query_ops.call_llm_safe')
    def test_uses_query_rewrite_system_prompt(self, mock_llm):
        """Uses QUERY_REWRITE_SYSTEM as system prompt."""
        mock_llm.return_value = "rewritten"

        rewrite_query("test query")

        call_args = mock_llm.call_args
        assert call_args[1]["system_prompt"] == QUERY_REWRITE_SYSTEM

    @patch('backend.common.query_ops.call_llm_safe')
    def test_uses_max_tokens_150(self, mock_llm):
        """Uses max_tokens=150."""
        mock_llm.return_value = "rewritten"

        rewrite_query("test")

        call_args = mock_llm.call_args
        assert call_args[1]["max_tokens"] == 150

    @patch('backend.common.query_ops.call_llm_safe')
    def test_without_company_formats_prompt(self, mock_llm):
        """Formats prompt for CRM question without company."""
        mock_llm.return_value = "rewritten"

        rewrite_query("how to export data")

        prompt = mock_llm.call_args[1]["prompt"]
        assert "Rewrite this CRM question" in prompt


# =============================================================================
# generate_hyde Tests
# =============================================================================

class TestGenerateHyde:
    """Tests for generate_hyde function."""

    @patch('backend.common.query_ops.call_llm_safe')
    def test_returns_hypothetical_answer(self, mock_llm):
        """Returns LLM-generated hypothetical answer."""
        mock_llm.return_value = "Acme CRM allows you to import contacts via CSV upload."

        result = generate_hyde("how to import contacts")

        assert "import contacts via CSV" in result

    @patch('backend.common.query_ops.call_llm_safe')
    def test_returns_empty_on_failure(self, mock_llm):
        """Returns empty string when LLM fails."""
        mock_llm.return_value = ""

        result = generate_hyde("question")

        assert result == ""

    @patch('backend.common.query_ops.call_llm_safe')
    def test_includes_company_context(self, mock_llm):
        """Includes company name in prompt when provided."""
        mock_llm.return_value = "hypothetical answer"

        generate_hyde("what's happening", company_name="TechCorp")

        prompt = mock_llm.call_args[1]["prompt"]
        assert "TechCorp" in prompt

    @patch('backend.common.query_ops.call_llm_safe')
    def test_uses_hyde_system_prompt(self, mock_llm):
        """Uses HYDE_SYSTEM as system prompt."""
        mock_llm.return_value = "hypothetical"

        generate_hyde("test query")

        assert mock_llm.call_args[1]["system_prompt"] == HYDE_SYSTEM

    @patch('backend.common.query_ops.call_llm_safe')
    def test_uses_max_tokens_200(self, mock_llm):
        """Uses max_tokens=200."""
        mock_llm.return_value = "answer"

        generate_hyde("test")

        assert mock_llm.call_args[1]["max_tokens"] == 200

    @patch('backend.common.query_ops.call_llm_safe')
    def test_without_company_formats_prompt(self, mock_llm):
        """Formats prompt as Question: when no company."""
        mock_llm.return_value = "answer"

        generate_hyde("how to export data")

        prompt = mock_llm.call_args[1]["prompt"]
        assert "Question:" in prompt


# =============================================================================
# System Prompts Tests
# =============================================================================

class TestSystemPrompts:
    """Tests for system prompt constants."""

    def test_query_rewrite_system_has_key_instructions(self):
        """QUERY_REWRITE_SYSTEM has required instructions."""
        assert "rewriting" in QUERY_REWRITE_SYSTEM.lower()
        assert "CRM" in QUERY_REWRITE_SYSTEM
        assert "natural language" in QUERY_REWRITE_SYSTEM

    def test_hyde_system_has_key_instructions(self):
        """HYDE_SYSTEM has required instructions."""
        assert "hypothetical" in HYDE_SYSTEM.lower()
        assert "Acme CRM" in HYDE_SYSTEM
        assert "semantic search" in HYDE_SYSTEM
