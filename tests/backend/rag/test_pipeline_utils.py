"""
Tests for backend/rag/pipeline/utils.py and base.py.
"""

import pytest
import time
from unittest.mock import MagicMock

from backend.rag.pipeline.utils import preprocess_query, extract_citations
from backend.rag.pipeline.base import PipelineProgress


# =============================================================================
# preprocess_query Tests
# =============================================================================

class TestPreprocessQuery:
    """Tests for preprocess_query function."""

    def test_strips_whitespace(self):
        """Strips leading and trailing whitespace."""
        assert preprocess_query("  hello world  ") == "hello world"
        assert preprocess_query("\n\thello\t\n") == "hello"

    def test_collapses_multiple_spaces(self):
        """Collapses multiple spaces to single space."""
        assert preprocess_query("hello    world") == "hello world"
        assert preprocess_query("a   b   c") == "a b c"

    def test_handles_empty_string(self):
        """Handles empty string."""
        assert preprocess_query("") == ""
        assert preprocess_query("   ") == ""

    def test_handles_newlines_and_tabs(self):
        """Converts newlines and tabs to spaces."""
        assert preprocess_query("hello\nworld") == "hello world"
        assert preprocess_query("hello\tworld") == "hello world"
        assert preprocess_query("hello\n\n\tworld") == "hello world"

    def test_preserves_single_spaces(self):
        """Preserves single spaces."""
        assert preprocess_query("hello world test") == "hello world test"


# =============================================================================
# extract_citations Tests
# =============================================================================

class TestExtractCitations:
    """Tests for extract_citations function."""

    def test_extracts_simple_citations(self):
        """Extracts simple [doc_id] citations."""
        text = "According to [doc1], this is true. See also [doc2]."
        result = extract_citations(text)
        assert "doc1" in result
        assert "doc2" in result

    def test_extracts_citations_with_numbers(self):
        """Extracts citations with numbers."""
        text = "Reference [doc123] and [section_456]."
        result = extract_citations(text)
        assert "doc123" in result
        assert "section_456" in result

    def test_extracts_citations_with_colons(self):
        """Extracts citations with double colons."""
        text = "See [module::section] for details."
        result = extract_citations(text)
        assert "module::section" in result

    def test_extracts_citations_with_hyphens(self):
        """Extracts citations with hyphens."""
        text = "Reference [my-doc-id]."
        result = extract_citations(text)
        assert "my-doc-id" in result

    def test_removes_duplicates(self):
        """Removes duplicate citations."""
        text = "See [doc1] and [doc1] again, also [DOC1]."
        result = extract_citations(text)
        # Should be deduplicated (case-insensitive)
        assert len([c for c in result if c.lower() == "doc1"]) == 1

    def test_returns_empty_for_no_citations(self):
        """Returns empty list when no citations."""
        assert extract_citations("No citations here.") == []
        assert extract_citations("") == []

    def test_preserves_order(self):
        """Preserves first-seen order."""
        text = "[first] then [second] then [third]"
        result = extract_citations(text)
        assert result[0] == "first"
        assert result[1] == "second"
        assert result[2] == "third"


# =============================================================================
# PipelineProgress Tests
# =============================================================================

class TestPipelineProgress:
    """Tests for PipelineProgress class."""

    def test_initializes_empty(self):
        """Initializes with empty steps."""
        progress = PipelineProgress()
        assert progress.steps == []
        assert progress.callback is None

    def test_initializes_with_callback(self):
        """Initializes with optional callback."""
        callback = MagicMock()
        progress = PipelineProgress(callback=callback)
        assert progress.callback is callback

    def test_start_and_complete_step(self):
        """Tracks step start and completion."""
        progress = PipelineProgress()

        progress.start_step("test_step", "Testing")
        progress.complete_step("test_step", "Test done")

        assert len(progress.steps) == 1
        step = progress.steps[0]
        assert step["id"] == "test_step"
        assert step["label"] == "Test done"
        assert step["status"] == "done"
        assert "elapsed_ms" in step

    def test_complete_step_records_elapsed_time(self):
        """Records elapsed time between start and complete."""
        progress = PipelineProgress()

        progress.start_step("step1", "Starting")
        time.sleep(0.05)  # 50ms
        progress.complete_step("step1", "Done")

        elapsed = progress.steps[0]["elapsed_ms"]
        assert elapsed >= 40  # Allow some variance
        assert elapsed < 200  # But not too much

    def test_complete_step_custom_status(self):
        """Allows custom status on complete."""
        progress = PipelineProgress()

        progress.start_step("step1", "Starting")
        progress.complete_step("step1", "Failed", status="error")

        assert progress.steps[0]["status"] == "error"

    def test_multiple_steps(self):
        """Tracks multiple sequential steps."""
        progress = PipelineProgress()

        progress.start_step("step1", "Step 1")
        progress.complete_step("step1", "Step 1 done")

        progress.start_step("step2", "Step 2")
        progress.complete_step("step2", "Step 2 done")

        progress.start_step("step3", "Step 3")
        progress.complete_step("step3", "Step 3 done")

        assert len(progress.steps) == 3
        assert progress.steps[0]["id"] == "step1"
        assert progress.steps[1]["id"] == "step2"
        assert progress.steps[2]["id"] == "step3"

    def test_get_steps_returns_copy(self):
        """get_steps returns the steps list."""
        progress = PipelineProgress()

        progress.start_step("step1", "Test")
        progress.complete_step("step1", "Done")

        steps = progress.get_steps()
        assert len(steps) == 1
        assert steps[0]["id"] == "step1"

    def test_total_elapsed_ms(self):
        """Tracks total elapsed time."""
        progress = PipelineProgress()
        time.sleep(0.05)  # 50ms

        elapsed = progress.total_elapsed_ms()
        assert elapsed >= 40
        assert elapsed < 200

    def test_callback_on_start(self):
        """Calls callback on step start."""
        callback = MagicMock()
        progress = PipelineProgress(callback=callback)

        progress.start_step("test", "Starting test")

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "test"
        assert "Starting" in args[1]

    def test_callback_on_complete(self):
        """Calls callback on step complete."""
        callback = MagicMock()
        progress = PipelineProgress(callback=callback)

        progress.start_step("test", "Starting")
        callback.reset_mock()

        progress.complete_step("test", "Done testing")

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "test"
        assert args[1] == "Done testing"
        assert isinstance(args[2], float)  # elapsed_ms

    def test_complete_without_start(self):
        """Handles complete without start (edge case)."""
        progress = PipelineProgress()

        # Should not crash, elapsed will be 0
        progress.complete_step("orphan", "Orphan step")

        assert len(progress.steps) == 1
        assert progress.steps[0]["elapsed_ms"] == 0
