"""
Tests for backend/rag/eval/judge.py - LLM-as-judge evaluation.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.rag.eval.judge import (
    judge_response,
    judge_account_response,
    check_privacy_leakage,
    compute_doc_recall,
    _call_judge_llm,
)
from backend.rag.eval.models import JudgeResult


# =============================================================================
# check_privacy_leakage Tests
# =============================================================================

class TestCheckPrivacyLeakage:
    """Tests for check_privacy_leakage function."""

    def test_no_leakage_when_same_company(self):
        """Returns 0 when all hits are from target company."""
        raw_hits = [
            {"company_id": "ACME-001", "text": "Data 1"},
            {"company_id": "ACME-001", "text": "Data 2"},
            {"company_id": "ACME-001", "text": "Data 3"},
        ]
        leakage, leaked_ids = check_privacy_leakage("ACME-001", raw_hits)

        assert leakage == 0
        assert leaked_ids == []

    def test_detects_single_leakage(self):
        """Detects leakage when one hit is from different company."""
        raw_hits = [
            {"company_id": "ACME-001", "text": "Data 1"},
            {"company_id": "OTHER-002", "text": "Data 2"},  # Leakage
            {"company_id": "ACME-001", "text": "Data 3"},
        ]
        leakage, leaked_ids = check_privacy_leakage("ACME-001", raw_hits)

        assert leakage == 1
        assert "OTHER-002" in leaked_ids

    def test_detects_multiple_leakages(self):
        """Detects multiple leaked company IDs."""
        raw_hits = [
            {"company_id": "ACME-001", "text": "Data 1"},
            {"company_id": "OTHER-002", "text": "Data 2"},
            {"company_id": "THIRD-003", "text": "Data 3"},
        ]
        leakage, leaked_ids = check_privacy_leakage("ACME-001", raw_hits)

        assert leakage == 1
        assert "OTHER-002" in leaked_ids
        assert "THIRD-003" in leaked_ids

    def test_deduplicates_leaked_ids(self):
        """Returns unique leaked IDs."""
        raw_hits = [
            {"company_id": "OTHER-002", "text": "Data 1"},
            {"company_id": "OTHER-002", "text": "Data 2"},  # Same company
        ]
        leakage, leaked_ids = check_privacy_leakage("ACME-001", raw_hits)

        assert leakage == 1
        assert leaked_ids == ["OTHER-002"]

    def test_handles_empty_hits(self):
        """Handles empty hits list."""
        leakage, leaked_ids = check_privacy_leakage("ACME-001", [])

        assert leakage == 0
        assert leaked_ids == []

    def test_handles_missing_company_id(self):
        """Handles hits without company_id field."""
        raw_hits = [
            {"text": "Data without company_id"},
            {"company_id": "", "text": "Empty company_id"},
        ]
        leakage, leaked_ids = check_privacy_leakage("ACME-001", raw_hits)

        assert leakage == 0
        assert leaked_ids == []


# =============================================================================
# compute_doc_recall Tests
# =============================================================================

class TestComputeDocRecall:
    """Tests for compute_doc_recall function."""

    def test_perfect_recall(self):
        """Returns 1.0 when all target docs are retrieved."""
        target = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4"]

        recall = compute_doc_recall(target, retrieved)

        assert recall == 1.0

    def test_zero_recall(self):
        """Returns 0.0 when no target docs are retrieved."""
        target = ["doc1", "doc2", "doc3"]
        retrieved = ["doc4", "doc5", "doc6"]

        recall = compute_doc_recall(target, retrieved)

        assert recall == 0.0

    def test_partial_recall(self):
        """Returns correct fraction for partial recall."""
        target = ["doc1", "doc2", "doc3", "doc4"]
        retrieved = ["doc1", "doc2"]

        recall = compute_doc_recall(target, retrieved)

        assert recall == 0.5

    def test_empty_target_returns_one(self):
        """Returns 1.0 when target is empty (no expectations)."""
        recall = compute_doc_recall([], ["doc1", "doc2"])

        assert recall == 1.0

    def test_empty_retrieved_returns_zero(self):
        """Returns 0.0 when nothing retrieved but targets exist."""
        recall = compute_doc_recall(["doc1"], [])

        assert recall == 0.0

    def test_handles_duplicates_in_target(self):
        """Handles duplicate doc_ids in target."""
        target = ["doc1", "doc1", "doc2"]  # doc1 appears twice
        retrieved = ["doc1", "doc2"]

        recall = compute_doc_recall(target, retrieved)

        assert recall == 1.0  # Both unique targets found

    def test_case_sensitive(self):
        """Doc IDs are case-sensitive."""
        target = ["Doc1"]
        retrieved = ["doc1"]

        recall = compute_doc_recall(target, retrieved)

        assert recall == 0.0  # Different case


# =============================================================================
# _call_judge_llm Tests
# =============================================================================

class TestCallJudgeLLM:
    """Tests for _call_judge_llm helper."""

    @patch('backend.rag.eval.judge.call_llm')
    def test_parses_valid_json_response(self, mock_llm):
        """Parses valid JSON response from LLM."""
        mock_llm.return_value = '''{"context_relevance": 1, "answer_relevance": 1,
            "groundedness": 1, "needs_human_review": 0,
            "confidence": 0.9, "explanation": "Good response"}'''

        result = _call_judge_llm("prompt", "system")

        assert isinstance(result, JudgeResult)
        assert result.context_relevance == 1
        assert result.answer_relevance == 1
        assert result.groundedness == 1
        assert result.needs_human_review == 0
        assert result.confidence == 0.9
        assert result.explanation == "Good response"

    @patch('backend.rag.eval.judge.call_llm')
    def test_extracts_json_from_text(self, mock_llm):
        """Extracts JSON from text with surrounding content."""
        mock_llm.return_value = '''Here is my evaluation:
            {"context_relevance": 1, "answer_relevance": 0,
            "groundedness": 1, "needs_human_review": 1}
            End of evaluation.'''

        result = _call_judge_llm("prompt", "system")

        assert result.context_relevance == 1
        assert result.answer_relevance == 0

    @patch('backend.rag.eval.judge.call_llm')
    def test_handles_empty_response(self, mock_llm):
        """Handles empty response from LLM."""
        mock_llm.return_value = ""

        result = _call_judge_llm("prompt", "system")

        assert result.context_relevance == 0
        assert result.needs_human_review == 1
        assert "Judge error" in result.explanation

    @patch('backend.rag.eval.judge.call_llm')
    def test_handles_invalid_json(self, mock_llm):
        """Handles invalid JSON response."""
        mock_llm.return_value = "This is not JSON at all"

        result = _call_judge_llm("prompt", "system")

        assert result.context_relevance == 0
        assert "Judge error" in result.explanation

    @patch('backend.rag.eval.judge.call_llm')
    def test_handles_missing_fields(self, mock_llm):
        """Uses defaults for missing fields."""
        mock_llm.return_value = '{"context_relevance": 1}'

        result = _call_judge_llm("prompt", "system")

        assert result.context_relevance == 1
        assert result.answer_relevance == 0  # Default
        assert result.groundedness == 0  # Default
        assert result.needs_human_review == 1  # Default
        assert result.confidence == 0.5  # Default

    @patch('backend.rag.eval.judge.call_llm')
    def test_handles_llm_exception(self, mock_llm):
        """Handles exception from LLM call."""
        mock_llm.side_effect = Exception("API Error")

        result = _call_judge_llm("prompt", "system")

        assert result.context_relevance == 0
        assert result.confidence == 0.0
        assert "Judge error" in result.explanation


# =============================================================================
# judge_response Tests
# =============================================================================

class TestJudgeResponse:
    """Tests for judge_response function."""

    @patch('backend.rag.eval.judge._call_judge_llm')
    def test_formats_prompt_correctly(self, mock_judge):
        """Formats prompt with all parameters."""
        mock_judge.return_value = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
            confidence=0.9,
            explanation="Good",
        )

        judge_response(
            question="What is X?",
            context="X is Y. This is the context.",
            answer="X is Y.",
            doc_ids=["doc1", "doc2"],
        )

        call_args = mock_judge.call_args[0]
        prompt = call_args[0]

        assert "What is X?" in prompt
        assert "X is Y. This is the context." in prompt
        assert "X is Y." in prompt
        assert "doc1, doc2" in prompt

    @patch('backend.rag.eval.judge._call_judge_llm')
    def test_returns_judge_result(self, mock_judge):
        """Returns JudgeResult from LLM."""
        expected = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=0,
            needs_human_review=1,
            confidence=0.7,
            explanation="Some hallucination detected",
        )
        mock_judge.return_value = expected

        result = judge_response(
            question="Q",
            context="C",
            answer="A",
            doc_ids=[],
        )

        assert result == expected


# =============================================================================
# judge_account_response Tests
# =============================================================================

class TestJudgeAccountResponse:
    """Tests for judge_account_response function."""

    @patch('backend.rag.eval.judge._call_judge_llm')
    def test_formats_prompt_with_company_info(self, mock_judge):
        """Includes company info in prompt."""
        mock_judge.return_value = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
            confidence=0.9,
            explanation="Good",
        )

        judge_account_response(
            company_id="ACME-001",
            company_name="Acme Corp",
            question="What is the status?",
            context="Status is active.",
            answer="The account is active.",
            sources=["src1"],
        )

        call_args = mock_judge.call_args[0]
        prompt = call_args[0]

        assert "ACME-001" in prompt
        assert "Acme Corp" in prompt
        assert "What is the status?" in prompt

    @patch('backend.rag.eval.judge._call_judge_llm')
    def test_returns_judge_result(self, mock_judge):
        """Returns JudgeResult for account response."""
        expected = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
            confidence=0.95,
            explanation="Accurate account info",
        )
        mock_judge.return_value = expected

        result = judge_account_response(
            company_id="ACME-001",
            company_name="Acme Corp",
            question="Q",
            context="C",
            answer="A",
            sources=[],
        )

        assert result == expected
