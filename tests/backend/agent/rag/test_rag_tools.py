"""
Tests for RAG tools module.

Covers tool_docs_rag and tool_account_rag functions with mocked dependencies.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestGetEmbedModel:
    """Tests for _get_embed_model function."""

    def test_get_embed_model_returns_model(self):
        """Test _get_embed_model returns HuggingFaceEmbedding."""
        with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding") as mock_embed:
            mock_model = MagicMock()
            mock_embed.return_value = mock_model

            from backend.agent.rag.tools import _get_embed_model

            result = _get_embed_model()

            assert result is mock_model
            mock_embed.assert_called_once()


class TestToolDocsRag:
    """Tests for tool_docs_rag function."""

    def test_tool_docs_rag_success(self):
        """Test successful docs RAG retrieval."""
        from backend.agent.rag.tools import tool_docs_rag

        mock_node1 = MagicMock()
        mock_node1.text = "This is documentation about contacts."
        mock_node1.metadata = {"doc_id": "contacts_guide", "file_name": "contacts.md"}

        mock_node2 = MagicMock()
        mock_node2.text = "More info about contacts."
        mock_node2.metadata = {"doc_id": "contacts_guide"}  # Same doc

        mock_node3 = MagicMock()
        mock_node3.text = "Pipeline documentation here."
        mock_node3.metadata = {"doc_id": "pipeline_overview"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node1, mock_node2, mock_node3]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_docs_index", return_value=mock_index):
            context, sources = tool_docs_rag("How do I create contacts?", top_k=5)

        assert "documentation about contacts" in context
        assert "Pipeline documentation" in context
        assert len(sources) == 2  # Two unique docs
        assert sources[0].type == "doc"
        assert sources[0].id == "contacts_guide"

    def test_tool_docs_rag_empty_results(self):
        """Test docs RAG with no results."""
        from backend.agent.rag.tools import tool_docs_rag

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_docs_index", return_value=mock_index):
            context, sources = tool_docs_rag("Unknown topic")

        assert context == ""
        assert sources == []

    def test_tool_docs_rag_exception(self):
        """Test docs RAG handles exceptions gracefully."""
        from backend.agent.rag.tools import tool_docs_rag

        with patch("backend.agent.rag.tools.get_docs_index", side_effect=Exception("Index error")):
            context, sources = tool_docs_rag("Any question")

        assert context == ""
        assert sources == []

    def test_tool_docs_rag_metadata_fallback(self):
        """Test docs RAG uses file_name when doc_id missing."""
        from backend.agent.rag.tools import tool_docs_rag

        mock_node = MagicMock()
        mock_node.text = "Some content"
        mock_node.metadata = {"file_name": "getting_started.md"}  # No doc_id

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_docs_index", return_value=mock_index):
            context, sources = tool_docs_rag("Getting started")

        assert len(sources) == 1
        assert sources[0].id == "getting_started.md"

    def test_tool_docs_rag_unknown_doc_id(self):
        """Test docs RAG with missing metadata uses 'unknown'."""
        from backend.agent.rag.tools import tool_docs_rag

        mock_node = MagicMock()
        mock_node.text = "Content without metadata"
        mock_node.metadata = {}  # Empty metadata

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_docs_index", return_value=mock_index):
            context, sources = tool_docs_rag("Question")

        assert len(sources) == 1
        assert sources[0].id == "unknown"


class TestToolAccountRag:
    """Tests for tool_account_rag function."""

    def test_tool_account_rag_success(self):
        """Test successful account RAG retrieval."""
        from backend.agent.rag.tools import tool_account_rag

        mock_node1 = MagicMock()
        mock_node1.text = "Meeting notes from last week."
        mock_node1.metadata = {"type": "note", "source_id": "note_001", "title": "Sales Call Notes"}

        mock_node2 = MagicMock()
        mock_node2.text = "Proposal document content."
        mock_node2.metadata = {"type": "attachment", "source_id": "att_001", "title": "Q4 Proposal"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node1, mock_node2]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_qdrant_client") as mock_client, \
             patch("backend.agent.rag.tools._get_embed_model") as mock_embed, \
             patch("llama_index.core.VectorStoreIndex") as mock_index_cls, \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore") as mock_store_cls, \
             patch("llama_index.core.Settings"):

            mock_client.return_value = MagicMock()
            mock_embed.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mock_index_cls.from_vector_store.return_value = mock_index

            context, sources = tool_account_rag("What were the meeting notes?", "COMP001")

        assert "Meeting notes" in context
        assert "Proposal document" in context
        assert len(sources) == 2
        assert sources[0].type == "note"
        assert sources[0].id == "note_001"
        assert sources[1].type == "attachment"

    def test_tool_account_rag_empty_results(self):
        """Test account RAG with no results."""
        from backend.agent.rag.tools import tool_account_rag

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_qdrant_client") as mock_client, \
             patch("backend.agent.rag.tools._get_embed_model") as mock_embed, \
             patch("llama_index.core.VectorStoreIndex") as mock_index_cls, \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore") as mock_store_cls, \
             patch("llama_index.core.Settings"):

            mock_client.return_value = MagicMock()
            mock_embed.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mock_index_cls.from_vector_store.return_value = mock_index

            context, sources = tool_account_rag("Unknown query", "COMP001")

        assert context == ""
        assert sources == []

    def test_tool_account_rag_exception(self):
        """Test account RAG handles exceptions gracefully."""
        from backend.agent.rag.tools import tool_account_rag

        with patch("backend.agent.rag.tools.get_qdrant_client", side_effect=Exception("Client error")):
            context, sources = tool_account_rag("Any question", "COMP001")

        assert context == ""
        assert sources == []

    def test_tool_account_rag_metadata_defaults(self):
        """Test account RAG uses defaults for missing metadata."""
        from backend.agent.rag.tools import tool_account_rag

        mock_node = MagicMock()
        mock_node.text = "Content with minimal metadata"
        mock_node.metadata = {}  # Empty metadata

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_qdrant_client") as mock_client, \
             patch("backend.agent.rag.tools._get_embed_model") as mock_embed, \
             patch("llama_index.core.VectorStoreIndex") as mock_index_cls, \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore") as mock_store_cls, \
             patch("llama_index.core.Settings"):

            mock_client.return_value = MagicMock()
            mock_embed.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mock_index_cls.from_vector_store.return_value = mock_index

            context, sources = tool_account_rag("Question", "COMP001")

        assert len(sources) == 1
        assert sources[0].type == "note"  # Default type
        assert sources[0].id == "unknown"  # Default id

    def test_tool_account_rag_doc_id_fallback(self):
        """Test account RAG uses doc_id when source_id missing."""
        from backend.agent.rag.tools import tool_account_rag

        mock_node = MagicMock()
        mock_node.text = "Some content"
        mock_node.metadata = {"type": "attachment", "doc_id": "doc_123"}  # No source_id

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.tools.get_qdrant_client") as mock_client, \
             patch("backend.agent.rag.tools._get_embed_model") as mock_embed, \
             patch("llama_index.core.VectorStoreIndex") as mock_index_cls, \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore") as mock_store_cls, \
             patch("llama_index.core.Settings"):

            mock_client.return_value = MagicMock()
            mock_embed.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mock_index_cls.from_vector_store.return_value = mock_index

            context, sources = tool_account_rag("Question", "COMP001")

        assert len(sources) == 1
        assert sources[0].id == "doc_123"
