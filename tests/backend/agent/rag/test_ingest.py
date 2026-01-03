"""
Tests for RAG ingest module.

Covers ingest_docs and ingest_private_texts functions with mocked dependencies.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path


class TestIngestDocs:
    """Tests for ingest_docs function."""

    def test_ingest_docs_success(self):
        """Test successful docs ingestion."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"file_path": "/docs/contacts.md"}

        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 10
        mock_client.get_collection.return_value = mock_collection_info

        with patch("backend.agent.rag.ingest.DOCS_DIR") as mock_docs_dir, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("llama_index.core.SimpleDirectoryReader", return_value=mock_reader), \
             patch("llama_index.core.VectorStoreIndex") as mock_index, \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_docs_dir.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()

            from backend.agent.rag.ingest import ingest_docs

            result = ingest_docs(recreate=False)

        assert result == 10
        mock_client.close.assert_called_once()

    def test_ingest_docs_with_recreate(self):
        """Test docs ingestion with collection recreation."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"file_path": "/docs/guide.md"}

        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 5
        mock_client.get_collection.return_value = mock_collection_info

        with patch("backend.agent.rag.ingest.DOCS_DIR") as mock_docs_dir, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("llama_index.core.SimpleDirectoryReader", return_value=mock_reader), \
             patch("llama_index.core.VectorStoreIndex"), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_docs_dir.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()

            from backend.agent.rag.ingest import ingest_docs

            result = ingest_docs(recreate=True)

        assert result == 5
        mock_client.delete_collection.assert_called_once()

    def test_ingest_docs_dir_not_exists(self):
        """Test docs ingestion when directory doesn't exist."""
        with patch("backend.agent.rag.ingest.DOCS_DIR") as mock_docs_dir, \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"):

            mock_docs_dir.exists.return_value = False

            from backend.agent.rag.ingest import ingest_docs

            result = ingest_docs()

        assert result == 0

    def test_ingest_docs_no_documents(self):
        """Test docs ingestion when no markdown files found."""
        mock_reader = MagicMock()
        mock_reader.load_data.return_value = []

        with patch("backend.agent.rag.ingest.DOCS_DIR") as mock_docs_dir, \
             patch("llama_index.core.SimpleDirectoryReader", return_value=mock_reader), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"):

            mock_docs_dir.exists.return_value = True

            from backend.agent.rag.ingest import ingest_docs

            result = ingest_docs()

        assert result == 0

    def test_ingest_docs_metadata_fallback(self):
        """Test docs ingestion with missing file_path metadata."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}  # No file_path

        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1
        mock_client.get_collection.return_value = mock_collection_info

        with patch("backend.agent.rag.ingest.DOCS_DIR") as mock_docs_dir, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("llama_index.core.SimpleDirectoryReader", return_value=mock_reader), \
             patch("llama_index.core.VectorStoreIndex"), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_docs_dir.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()

            from backend.agent.rag.ingest import ingest_docs

            result = ingest_docs(recreate=False)

        assert result == 1
        # doc_id should be "unknown" when file_path is empty
        assert mock_doc.metadata["doc_id"] == "unknown"


class TestIngestPrivateTexts:
    """Tests for ingest_private_texts function."""

    def test_ingest_private_texts_success(self):
        """Test successful private texts ingestion."""
        jsonl_content = '{"id": "note_001", "text": "Meeting notes", "company_id": "COMP001", "type": "note", "title": "Sales Call"}\n'

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1
        mock_client.get_collection.return_value = mock_collection_info

        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_jsonl_path, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("builtins.open", mock_open(read_data=jsonl_content)), \
             patch("llama_index.core.Document") as mock_doc_cls, \
             patch("llama_index.core.VectorStoreIndex"), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_jsonl_path.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()
            mock_doc_cls.return_value = MagicMock()

            from backend.agent.rag.ingest import ingest_private_texts

            result = ingest_private_texts(recreate=False)

        assert result == 1
        mock_client.close.assert_called_once()

    def test_ingest_private_texts_with_recreate(self):
        """Test private texts ingestion with collection recreation."""
        jsonl_content = '{"id": "note_001", "text": "Content"}\n'

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1
        mock_client.get_collection.return_value = mock_collection_info

        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_jsonl_path, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("builtins.open", mock_open(read_data=jsonl_content)), \
             patch("llama_index.core.Document") as mock_doc_cls, \
             patch("llama_index.core.VectorStoreIndex"), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_jsonl_path.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()
            mock_doc_cls.return_value = MagicMock()

            from backend.agent.rag.ingest import ingest_private_texts

            result = ingest_private_texts(recreate=True)

        assert result == 1
        mock_client.delete_collection.assert_called_once()

    def test_ingest_private_texts_file_not_exists(self):
        """Test private texts ingestion when JSONL file doesn't exist."""
        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_jsonl_path, \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"):

            mock_jsonl_path.exists.return_value = False

            from backend.agent.rag.ingest import ingest_private_texts

            result = ingest_private_texts()

        assert result == 0

    def test_ingest_private_texts_empty_file(self):
        """Test private texts ingestion with empty JSONL file."""
        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_jsonl_path, \
             patch("builtins.open", mock_open(read_data="")), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"):

            mock_jsonl_path.exists.return_value = True

            from backend.agent.rag.ingest import ingest_private_texts

            result = ingest_private_texts()

        assert result == 0

    def test_ingest_private_texts_invalid_json_line(self):
        """Test private texts ingestion skips invalid JSON lines."""
        jsonl_content = '{"id": "note_001", "text": "Valid"}\ninvalid json line\n{"id": "note_002", "text": "Also valid"}\n'

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 2
        mock_client.get_collection.return_value = mock_collection_info

        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_jsonl_path, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("builtins.open", mock_open(read_data=jsonl_content)), \
             patch("llama_index.core.Document") as mock_doc_cls, \
             patch("llama_index.core.VectorStoreIndex"), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_jsonl_path.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()
            mock_doc_cls.return_value = MagicMock()

            from backend.agent.rag.ingest import ingest_private_texts

            result = ingest_private_texts(recreate=False)

        # Should still succeed with 2 valid docs
        assert result == 2

    def test_ingest_private_texts_blank_lines(self):
        """Test private texts ingestion skips blank lines."""
        jsonl_content = '{"id": "note_001", "text": "Content"}\n\n   \n{"id": "note_002", "text": "More"}\n'

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 2
        mock_client.get_collection.return_value = mock_collection_info

        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_jsonl_path, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("builtins.open", mock_open(read_data=jsonl_content)), \
             patch("llama_index.core.Document") as mock_doc_cls, \
             patch("llama_index.core.VectorStoreIndex"), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_jsonl_path.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()
            mock_doc_cls.return_value = MagicMock()

            from backend.agent.rag.ingest import ingest_private_texts

            result = ingest_private_texts(recreate=False)

        assert result == 2

    def test_ingest_private_texts_metadata_fields(self):
        """Test private texts ingestion creates documents with correct metadata."""
        jsonl_content = '{"id": "note_001", "text": "Meeting notes", "company_id": "COMP001", "type": "note", "title": "Call", "contact_id": "CONT001", "opportunity_id": "OPP001"}\n'

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1
        mock_client.get_collection.return_value = mock_collection_info

        created_docs = []

        def capture_doc(**kwargs):
            doc = MagicMock()
            doc.text = kwargs.get("text")
            doc.metadata = kwargs.get("metadata")
            created_docs.append(doc)
            return doc

        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_jsonl_path, \
             patch("backend.agent.rag.ingest.QDRANT_PATH") as mock_qdrant_path, \
             patch("backend.agent.rag.ingest.QdrantClient", return_value=mock_client), \
             patch("builtins.open", mock_open(read_data=jsonl_content)), \
             patch("llama_index.core.Document", side_effect=capture_doc), \
             patch("llama_index.core.VectorStoreIndex"), \
             patch("llama_index.core.Settings"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore"):

            mock_jsonl_path.exists.return_value = True
            mock_qdrant_path.mkdir = MagicMock()

            from backend.agent.rag.ingest import ingest_private_texts

            result = ingest_private_texts(recreate=False)

        assert result == 1
        assert len(created_docs) == 1
        doc = created_docs[0]
        assert doc.text == "Meeting notes"
        assert doc.metadata["doc_id"] == "note_001"
        assert doc.metadata["company_id"] == "COMP001"
        assert doc.metadata["type"] == "note"
        assert doc.metadata["contact_id"] == "CONT001"
        assert doc.metadata["opportunity_id"] == "OPP001"
