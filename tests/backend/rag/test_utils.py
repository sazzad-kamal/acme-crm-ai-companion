"""
Tests for RAG utility functions.

Tests token estimation and CSV directory finding.

Run with:
    pytest tests/backend/rag/test_utils.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from backend.rag.utils import (
    estimate_tokens,
    tokens_to_chars,
    find_csv_dir,
    CHARS_PER_TOKEN,
    CSV_DIR_CANDIDATES,
    REQUIRED_CSV_FILES,
)


# =============================================================================
# Test: Token Estimation
# =============================================================================

class TestTokenEstimation:
    """Tests for token estimation functions."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        text = "This is a test"  # 14 chars
        tokens = estimate_tokens(text)

        # Should be approximately 14/4 = 3-4 tokens
        assert tokens >= 3
        assert tokens <= 4

    def test_estimate_tokens_empty_string(self):
        """Test estimation with empty string."""
        tokens = estimate_tokens("")

        assert tokens == 0

    def test_estimate_tokens_single_char(self):
        """Test estimation with single character."""
        tokens = estimate_tokens("a")

        assert tokens == 0  # 1 char / 4 = 0

    def test_estimate_tokens_long_text(self):
        """Test estimation with long text."""
        text = "word " * 100  # 500 chars
        tokens = estimate_tokens(text)

        # Should be approximately 500/4 = 125 tokens
        assert tokens >= 120
        assert tokens <= 130

    def test_tokens_to_chars_basic(self):
        """Test converting tokens to characters."""
        chars = tokens_to_chars(100)

        assert chars == 400  # 100 * 4

    def test_tokens_to_chars_zero(self):
        """Test converting zero tokens."""
        chars = tokens_to_chars(0)

        assert chars == 0

    def test_tokens_to_chars_large(self):
        """Test converting large token count."""
        chars = tokens_to_chars(10000)

        assert chars == 40000

    def test_round_trip_approximately_correct(self):
        """Test that token <-> char conversion is approximately reversible."""
        original_chars = 1000
        tokens = original_chars // CHARS_PER_TOKEN
        back_to_chars = tokens_to_chars(tokens)

        # Should be close to original (within tolerance)
        assert abs(back_to_chars - original_chars) <= CHARS_PER_TOKEN

    def test_chars_per_token_constant(self):
        """Test that CHARS_PER_TOKEN has expected value."""
        assert CHARS_PER_TOKEN == 4


# =============================================================================
# Test: CSV Directory Finding
# =============================================================================

class TestFindCsvDir:
    """Tests for CSV directory finding."""

    def test_find_csv_dir_returns_path(self):
        """Test that find_csv_dir returns a Path object."""
        csv_dir = find_csv_dir()

        assert isinstance(csv_dir, Path)
        assert csv_dir.exists()
        assert csv_dir.is_dir()

    def test_csv_dir_contains_required_files(self):
        """Test that CSV directory contains required files."""
        csv_dir = find_csv_dir()

        for required_file in REQUIRED_CSV_FILES:
            file_path = csv_dir / required_file
            assert file_path.exists(), f"Missing required file: {required_file}"

    def test_csv_dir_candidates_is_list(self):
        """Test that CSV_DIR_CANDIDATES is a list of paths."""
        assert isinstance(CSV_DIR_CANDIDATES, list)
        assert len(CSV_DIR_CANDIDATES) > 0
        assert all(isinstance(p, Path) for p in CSV_DIR_CANDIDATES)

    def test_required_files_is_list(self):
        """Test that REQUIRED_CSV_FILES is a list of strings."""
        assert isinstance(REQUIRED_CSV_FILES, list)
        assert len(REQUIRED_CSV_FILES) > 0
        assert all(isinstance(f, str) for f in REQUIRED_CSV_FILES)

    def test_required_files_includes_companies(self):
        """Test that required files includes companies.csv."""
        assert "companies.csv" in REQUIRED_CSV_FILES

    def test_required_files_includes_history(self):
        """Test that required files includes history.csv."""
        assert "history.csv" in REQUIRED_CSV_FILES

    def test_find_csv_dir_error_when_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when no dir found."""
        # Patch to use nonexistent directories
        with patch('backend.rag.utils.CSV_DIR_CANDIDATES', [tmp_path / "nonexistent"]):
            with pytest.raises(FileNotFoundError, match="Could not find CSV data directory"):
                find_csv_dir()

    def test_error_message_includes_checked_paths(self, tmp_path):
        """Test that error message includes paths checked."""
        fake_candidates = [tmp_path / "fake1", tmp_path / "fake2"]

        with patch('backend.rag.utils.CSV_DIR_CANDIDATES', fake_candidates):
            with pytest.raises(FileNotFoundError) as exc_info:
                find_csv_dir()

            error_msg = str(exc_info.value)
            assert "Checked:" in error_msg
            assert "fake1" in error_msg or "fake2" in error_msg

    def test_finds_first_valid_directory(self, tmp_path):
        """Test that first valid directory is returned."""
        # Create two valid directories
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"

        dir1.mkdir()
        dir2.mkdir()

        for req_file in REQUIRED_CSV_FILES:
            (dir1 / req_file).touch()
            (dir2 / req_file).touch()

        with patch('backend.rag.utils.CSV_DIR_CANDIDATES', [dir1, dir2]):
            found_dir = find_csv_dir()

            # Should find dir1 (first one)
            assert found_dir == dir1

    def test_skips_invalid_directories(self, tmp_path):
        """Test that invalid directories are skipped."""
        invalid_dir = tmp_path / "invalid"
        valid_dir = tmp_path / "valid"

        invalid_dir.mkdir()
        valid_dir.mkdir()

        # Only create required files in valid dir
        for req_file in REQUIRED_CSV_FILES:
            (valid_dir / req_file).touch()

        with patch('backend.rag.utils.CSV_DIR_CANDIDATES', [invalid_dir, valid_dir]):
            found_dir = find_csv_dir()

            # Should find valid dir (second one)
            assert found_dir == valid_dir


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestUtilsEdgeCases:
    """Tests for edge cases in utility functions."""

    def test_estimate_tokens_unicode(self):
        """Test token estimation with unicode characters."""
        text = "Hello 你好 世界 🎉"
        tokens = estimate_tokens(text)

        # Should handle unicode gracefully
        assert tokens >= 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_whitespace(self):
        """Test token estimation with only whitespace."""
        text = "   \n\t  "
        tokens = estimate_tokens(text)

        # Should count characters including whitespace
        assert tokens == len(text) // CHARS_PER_TOKEN

    def test_estimate_tokens_very_long(self):
        """Test token estimation with very long text."""
        text = "a" * 1000000  # 1 million characters
        tokens = estimate_tokens(text)

        # Should handle large text
        assert tokens == 250000  # 1M / 4

    def test_tokens_to_chars_negative(self):
        """Test tokens_to_chars with negative input."""
        chars = tokens_to_chars(-10)

        assert chars == -40  # Still does calculation

    def test_find_csv_dir_with_extra_files(self, tmp_path):
        """Test that extra files in directory don't break finding."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()

        # Add required files
        for req_file in REQUIRED_CSV_FILES:
            (csv_dir / req_file).touch()

        # Add extra files
        (csv_dir / "extra1.csv").touch()
        (csv_dir / "extra2.txt").touch()

        with patch('backend.rag.utils.CSV_DIR_CANDIDATES', [csv_dir]):
            found_dir = find_csv_dir()

            assert found_dir == csv_dir

    def test_csv_dir_is_file_not_directory(self, tmp_path):
        """Test handling when candidate is a file not directory."""
        csv_file = tmp_path / "csv"
        csv_file.touch()  # Create file instead of directory

        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        for req_file in REQUIRED_CSV_FILES:
            (valid_dir / req_file).touch()

        with patch('backend.rag.utils.CSV_DIR_CANDIDATES', [csv_file, valid_dir]):
            found_dir = find_csv_dir()

            # Should skip file and find valid dir
            assert found_dir == valid_dir


# =============================================================================
# Test: Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_chars_per_token_is_positive(self):
        """Test that CHARS_PER_TOKEN is positive."""
        assert CHARS_PER_TOKEN > 0

    def test_csv_dir_candidates_not_empty(self):
        """Test that CSV_DIR_CANDIDATES is not empty."""
        assert len(CSV_DIR_CANDIDATES) > 0

    def test_required_csv_files_not_empty(self):
        """Test that REQUIRED_CSV_FILES is not empty."""
        assert len(REQUIRED_CSV_FILES) > 0

    def test_csv_candidates_are_absolute_or_resolvable(self):
        """Test that CSV candidates can be resolved."""
        for candidate in CSV_DIR_CANDIDATES:
            # Should be a Path object
            assert isinstance(candidate, Path)
            # Path should be resolvable (may or may not exist)
            try:
                _ = candidate.resolve()
            except Exception as e:
                pytest.fail(f"Could not resolve path {candidate}: {e}")
