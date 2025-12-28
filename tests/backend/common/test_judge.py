"""
Tests for backend/common/judge.py - shared LLM-as-judge utilities.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.common.judge import (
    JudgeScores,
    parse_judge_response,
    format_judge_system,
    judge_answer,
    JUDGE_SYSTEM_BASE,
    JUDGE_SYSTEM_CRM,
    JUDGE_SYSTEM_RAG,
    JUDGE_PROMPT,
)


# =============================================================================
# JudgeScores Tests
# =============================================================================

class TestJudgeScores:
    """Tests for JudgeScores dataclass."""

    def test_creates_with_required_fields(self):
        """Creates with answer_relevance and answer_grounded."""
        scores = JudgeScores(answer_relevance=1, answer_grounded=0)
        assert scores.answer_relevance == 1
        assert scores.answer_grounded == 0
        assert scores.explanation == ""

    def test_creates_with_explanation(self):
        """Creates with optional explanation."""
        scores = JudgeScores(
            answer_relevance=1,
            answer_grounded=1,
            explanation="Both criteria met."
        )
        assert scores.explanation == "Both criteria met."


# =============================================================================
# parse_judge_response Tests
# =============================================================================

class TestParseJudgeResponse:
    """Tests for parse_judge_response function."""

    def test_parses_plain_json(self):
        """Parses plain JSON response."""
        response = '{"answer_relevance": 1, "answer_grounded": 0, "explanation": "test"}'
        result = parse_judge_response(response)
        assert result["answer_relevance"] == 1
        assert result["answer_grounded"] == 0
        assert result["explanation"] == "test"

    def test_parses_json_in_markdown_code_block(self):
        """Parses JSON wrapped in ```json code block."""
        response = '''Here is my evaluation:
```json
{"answer_relevance": 1, "answer_grounded": 1, "explanation": "Good answer"}
```
'''
        result = parse_judge_response(response)
        assert result["answer_relevance"] == 1
        assert result["answer_grounded"] == 1

    def test_parses_json_in_plain_code_block(self):
        """Parses JSON wrapped in ``` code block."""
        response = '''```
{"answer_relevance": 0, "answer_grounded": 1}
```'''
        result = parse_judge_response(response)
        assert result["answer_relevance"] == 0
        assert result["answer_grounded"] == 1

    def test_extracts_json_from_text(self):
        """Extracts JSON object from surrounding text."""
        response = 'The evaluation is: {"answer_relevance": 1, "answer_grounded": 1} as shown.'
        result = parse_judge_response(response)
        assert result["answer_relevance"] == 1

    def test_raises_on_empty_response(self):
        """Raises ValueError on empty response."""
        with pytest.raises(ValueError, match="Empty response"):
            parse_judge_response("")

        with pytest.raises(ValueError, match="Empty response"):
            parse_judge_response("   ")

    def test_raises_on_none_response(self):
        """Raises ValueError on None response."""
        with pytest.raises((ValueError, TypeError)):
            parse_judge_response(None)

    def test_raises_on_invalid_json(self):
        """Raises on invalid JSON."""
        with pytest.raises(Exception):
            parse_judge_response("not valid json at all")


# =============================================================================
# format_judge_system Tests
# =============================================================================

class TestFormatJudgeSystem:
    """Tests for format_judge_system function."""

    def test_formats_with_domain_and_data_type(self):
        """Formats system prompt with domain and data_type."""
        result = format_judge_system(
            domain="test domain",
            data_type="test data"
        )
        assert "test domain" in result
        assert "test data" in result
        assert "ANSWER_RELEVANCE" in result
        assert "ANSWER_GROUNDED" in result

    def test_uses_default_data_type(self):
        """Uses 'facts' as default data_type."""
        result = format_judge_system(domain="CRM system")
        assert "facts" in result
        assert "CRM system" in result

    def test_crm_preset_contains_expected_values(self):
        """JUDGE_SYSTEM_CRM has correct values."""
        assert "CRM assistant" in JUDGE_SYSTEM_CRM
        assert "companies, dates, values" in JUDGE_SYSTEM_CRM

    def test_rag_preset_contains_expected_values(self):
        """JUDGE_SYSTEM_RAG has correct values."""
        assert "RAG system" in JUDGE_SYSTEM_RAG
        assert "document references" in JUDGE_SYSTEM_RAG


# =============================================================================
# judge_answer Tests
# =============================================================================

class TestJudgeAnswer:
    """Tests for judge_answer function."""

    @patch('backend.common.judge.call_llm')
    def test_returns_judge_scores_on_success(self, mock_call_llm):
        """Returns JudgeScores with parsed values."""
        mock_call_llm.return_value = '{"answer_relevance": 1, "answer_grounded": 1, "explanation": "Good"}'

        result = judge_answer(
            question="What is the renewal date?",
            answer="The renewal date is March 15, 2025.",
            sources=["company_001"],
        )

        assert isinstance(result, JudgeScores)
        assert result.answer_relevance == 1
        assert result.answer_grounded == 1
        assert result.explanation == "Good"

    @patch('backend.common.judge.call_llm')
    def test_uses_correct_prompt_format(self, mock_call_llm):
        """Formats prompt with question, answer, and sources."""
        mock_call_llm.return_value = '{"answer_relevance": 1, "answer_grounded": 1}'

        judge_answer(
            question="Test question?",
            answer="Test answer.",
            sources=["src1", "src2"],
        )

        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]
        assert "Test question?" in prompt
        assert "Test answer." in prompt
        assert "src1, src2" in prompt

    @patch('backend.common.judge.call_llm')
    def test_handles_empty_sources(self, mock_call_llm):
        """Handles empty sources list."""
        mock_call_llm.return_value = '{"answer_relevance": 1, "answer_grounded": 0}'

        judge_answer(
            question="Question?",
            answer="Answer.",
            sources=[],
        )

        prompt = mock_call_llm.call_args[0][0]
        assert "None" in prompt

    @patch('backend.common.judge.call_llm')
    def test_uses_custom_system_prompt(self, mock_call_llm):
        """Uses custom system prompt when provided."""
        mock_call_llm.return_value = '{"answer_relevance": 1, "answer_grounded": 1}'
        custom_prompt = "Custom judge system prompt"

        judge_answer(
            question="Q",
            answer="A",
            sources=[],
            system_prompt=custom_prompt,
        )

        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["system_prompt"] == custom_prompt

    @patch('backend.common.judge.call_llm')
    def test_uses_custom_model(self, mock_call_llm):
        """Uses custom model when provided."""
        mock_call_llm.return_value = '{"answer_relevance": 1, "answer_grounded": 1}'

        judge_answer(
            question="Q",
            answer="A",
            sources=[],
            model="gpt-4",
        )

        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["model"] == "gpt-4"

    @patch('backend.common.judge.call_llm')
    def test_returns_zero_scores_on_error(self, mock_call_llm):
        """Returns zero scores when LLM call fails."""
        mock_call_llm.side_effect = Exception("LLM error")

        result = judge_answer(
            question="Q",
            answer="A",
            sources=[],
        )

        assert result.answer_relevance == 0
        assert result.answer_grounded == 0
        assert "Judge error" in result.explanation

    @patch('backend.common.judge.call_llm')
    def test_handles_missing_fields_in_response(self, mock_call_llm):
        """Handles missing fields with defaults."""
        mock_call_llm.return_value = '{"answer_relevance": 1}'  # Missing answer_grounded

        result = judge_answer(
            question="Q",
            answer="A",
            sources=[],
        )

        assert result.answer_relevance == 1
        assert result.answer_grounded == 0  # Default
        assert result.explanation == ""  # Default


# =============================================================================
# JUDGE_PROMPT Tests
# =============================================================================

class TestJudgePrompt:
    """Tests for JUDGE_PROMPT template."""

    def test_prompt_has_placeholders(self):
        """Prompt contains required placeholders."""
        assert "{question}" in JUDGE_PROMPT
        assert "{answer}" in JUDGE_PROMPT
        assert "{sources}" in JUDGE_PROMPT

    def test_prompt_formats_correctly(self):
        """Prompt formats with values."""
        formatted = JUDGE_PROMPT.format(
            question="What is X?",
            answer="X is Y.",
            sources="doc1, doc2",
        )
        assert "What is X?" in formatted
        assert "X is Y." in formatted
        assert "doc1, doc2" in formatted
