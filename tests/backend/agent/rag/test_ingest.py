"""
Tests for RAG ingest module.

Covers ingest_private_texts function with mocked dependencies.
These tests focus on the early-exit paths that don't require llama_index.
More comprehensive integration tests are in e2e tests.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open


class TestIngestPrivateTexts:
    """Tests for ingest_private_texts function."""

    def test_ingest_private_texts_file_not_exists(self):
        """Test private texts ingestion when JSONL file doesn't exist."""
        from backend.agent.rag.ingest import ingest_private_texts

        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = ingest_private_texts()

        assert result == 0

    def test_ingest_returns_zero_when_no_file(self):
        """Test that ingest returns 0 when JSONL file doesn't exist."""
        from backend.agent.rag.ingest import ingest_private_texts

        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = ingest_private_texts()
            assert result == 0

    def test_ingest_private_texts_jsonl_path_config(self):
        """Test that JSONL_PATH is correctly configured."""
        from backend.agent.rag.config import JSONL_PATH

        assert JSONL_PATH is not None
        assert str(JSONL_PATH).endswith(".jsonl")

    def test_ingest_private_texts_qdrant_path_config(self):
        """Test that QDRANT_PATH is correctly configured."""
        from backend.agent.rag.config import QDRANT_PATH

        assert QDRANT_PATH is not None
        assert "qdrant" in str(QDRANT_PATH)

    def test_ingest_private_texts_collection_config(self):
        """Test that PRIVATE_COLLECTION is correctly configured."""
        from backend.agent.rag.config import PRIVATE_COLLECTION

        assert PRIVATE_COLLECTION is not None
        assert isinstance(PRIVATE_COLLECTION, str)
        assert len(PRIVATE_COLLECTION) > 0

    def test_ingest_private_texts_embedding_model_config(self):
        """Test that EMBEDDING_MODEL is correctly configured."""
        from backend.agent.rag.config import EMBEDDING_MODEL

        assert EMBEDDING_MODEL is not None
        assert isinstance(EMBEDDING_MODEL, str)
        assert len(EMBEDDING_MODEL) > 0
