"""
Tests for RAG tools module.

Covers tool_account_rag function with mocked dependencies.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch


# Mock llama_index modules before any imports
mock_huggingface_module = MagicMock()
mock_core_module = MagicMock()
mock_qdrant_module = MagicMock()


class TestGetEmbedModel:
    """Tests for _get_embed_model function."""

    def test_get_embed_model_returns_model(self):
        """Test _get_embed_model returns HuggingFaceEmbedding."""
        mock_embed_class = MagicMock()
        mock_model = MagicMock()
        mock_embed_class.return_value = mock_model

        # Mock the llama_index module
        mock_hf = MagicMock()
        mock_hf.HuggingFaceEmbedding = mock_embed_class

        with patch.dict(sys.modules, {"llama_index.embeddings.huggingface": mock_hf}):
            # Clear any cached imports
            if "backend.agent.rag.tools" in sys.modules:
                del sys.modules["backend.agent.rag.tools"]

            from backend.agent.rag.tools import _get_embed_model

            result = _get_embed_model()

            assert result is mock_model
            mock_embed_class.assert_called_once()


class TestToolAccountRag:
    """Tests for tool_account_rag function."""

    def _setup_llama_mocks(self, mock_retriever):
        """Setup common llama_index mocks."""
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        mock_index_cls = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index

        mock_settings = MagicMock()
        mock_vector_store_cls = MagicMock()
        mock_embed_cls = MagicMock()

        mock_core = MagicMock()
        mock_core.Settings = mock_settings
        mock_core.VectorStoreIndex = mock_index_cls

        mock_hf = MagicMock()
        mock_hf.HuggingFaceEmbedding = mock_embed_cls

        mock_qdrant_vs = MagicMock()
        mock_qdrant_vs.QdrantVectorStore = mock_vector_store_cls

        return {
            "llama_index.core": mock_core,
            "llama_index.embeddings.huggingface": mock_hf,
            "llama_index.vector_stores.qdrant": mock_qdrant_vs,
        }

    def test_tool_account_rag_success(self):
        """Test successful account RAG retrieval."""
        mock_node1 = MagicMock()
        mock_node1.text = "Meeting notes from last week."
        mock_node1.metadata = {"type": "note", "source_id": "note_001", "title": "Sales Call Notes"}

        mock_node2 = MagicMock()
        mock_node2.text = "Proposal document content."
        mock_node2.metadata = {"type": "attachment", "source_id": "att_001", "title": "Q4 Proposal"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node1, mock_node2]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            # Clear cached import
            if "backend.agent.rag.tools" in sys.modules:
                del sys.modules["backend.agent.rag.tools"]

            from backend.agent.rag import tools

            # Patch where it's used, not where it's defined
            with patch.object(tools, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = tools.tool_account_rag("What were the meeting notes?", "COMP001")

        assert "Meeting notes" in context
        assert "Proposal document" in context
        assert len(sources) == 2
        assert sources[0].type == "note"
        assert sources[0].id == "note_001"
        assert sources[1].type == "attachment"

    def test_tool_account_rag_empty_results(self):
        """Test account RAG with no results."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.rag.tools" in sys.modules:
                del sys.modules["backend.agent.rag.tools"]

            from backend.agent.rag import tools

            with patch.object(tools, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = tools.tool_account_rag("Unknown query", "COMP001")

        assert context == ""
        assert sources == []

    def test_tool_account_rag_exception(self):
        """Test account RAG handles exceptions gracefully."""
        with patch("backend.agent.rag.tools.get_qdrant_client", side_effect=Exception("Client error")):
            from backend.agent.rag.tools import tool_account_rag

            context, sources = tool_account_rag("Any question", "COMP001")

        assert context == ""
        assert sources == []

    def test_tool_account_rag_metadata_defaults(self):
        """Test account RAG uses defaults for missing metadata."""
        mock_node = MagicMock()
        mock_node.text = "Content with minimal metadata"
        mock_node.metadata = {}  # Empty metadata

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        # Create fresh mock index that returns our retriever
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        mock_index_cls = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index

        mock_core = MagicMock()
        mock_core.Settings = MagicMock()
        mock_core.VectorStoreIndex = mock_index_cls

        mock_hf = MagicMock()
        mock_qdrant_vs = MagicMock()

        llama_mocks = {
            "llama_index.core": mock_core,
            "llama_index.embeddings.huggingface": mock_hf,
            "llama_index.vector_stores.qdrant": mock_qdrant_vs,
        }

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.rag.tools" in sys.modules:
                del sys.modules["backend.agent.rag.tools"]

            # Import after setting up llama_index mocks
            from backend.agent.rag import tools

            # Patch where it's used, not where it's defined
            with patch.object(tools, "get_qdrant_client") as mock_client_fn:
                mock_client_fn.return_value = MagicMock()
                context, sources = tools.tool_account_rag("Question", "COMP001")

        assert len(sources) == 1
        assert sources[0].type == "note"  # Default type
        assert sources[0].id == "unknown"  # Default id

    def test_tool_account_rag_doc_id_fallback(self):
        """Test account RAG uses doc_id when source_id missing."""
        mock_node = MagicMock()
        mock_node.text = "Some content"
        mock_node.metadata = {"type": "attachment", "doc_id": "doc_123"}  # No source_id

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        # Create fresh mock index that returns our retriever
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        mock_index_cls = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index

        mock_core = MagicMock()
        mock_core.Settings = MagicMock()
        mock_core.VectorStoreIndex = mock_index_cls

        mock_hf = MagicMock()
        mock_qdrant_vs = MagicMock()

        llama_mocks = {
            "llama_index.core": mock_core,
            "llama_index.embeddings.huggingface": mock_hf,
            "llama_index.vector_stores.qdrant": mock_qdrant_vs,
        }

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.rag.tools" in sys.modules:
                del sys.modules["backend.agent.rag.tools"]

            from backend.agent.rag import tools

            with patch.object(tools, "get_qdrant_client") as mock_client_fn:
                mock_client_fn.return_value = MagicMock()
                context, sources = tools.tool_account_rag("Question", "COMP001")

        assert len(sources) == 1
        assert sources[0].id == "doc_123"
