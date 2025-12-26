"""
Tests for context building utilities.

Tests ContextBuilder class and helper functions.

Run with:
    pytest tests/backend/rag/test_context_builder.py -v
"""

import pytest

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.pipeline.context_builder import (
    ContextBuilder,
    build_context,
    build_context_with_sources,
    build_private_context,
    build_docs_context,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_chunks():
    """Create sample DocumentChunks for testing."""
    return [
        DocumentChunk(
            chunk_id="doc1::0",
            doc_id="doc1",
            title="Document One",
            text="This is the first chunk of text.",
            metadata={"section_heading": "Introduction"},
        ),
        DocumentChunk(
            chunk_id="doc1::1",
            doc_id="doc1",
            title="Document One",
            text="This is the second chunk with more content.",
            metadata={"section_heading": "Details"},
        ),
        DocumentChunk(
            chunk_id="doc2::0",
            doc_id="doc2",
            title="Document Two",
            text="Content from a different document.",
            metadata={"section_heading": "Overview"},
        ),
    ]


@pytest.fixture
def scored_chunks(sample_chunks):
    """Create sample ScoredChunks for testing."""
    return [
        ScoredChunk(chunk=chunk, rerank_score=0.9 - i * 0.1)
        for i, chunk in enumerate(sample_chunks)
    ]


# =============================================================================
# Test: ContextBuilder - Basic Functionality
# =============================================================================

class TestContextBuilderBasic:
    """Tests for basic ContextBuilder functionality."""

    def test_initialization(self):
        """Test that ContextBuilder initializes correctly."""
        builder = ContextBuilder(max_tokens=1000)

        assert builder.max_tokens == 1000
        assert builder.total_tokens == 0
        assert builder.remaining_tokens == 1000

    def test_add_single_chunk(self, sample_chunks):
        """Test adding a single chunk."""
        builder = ContextBuilder(max_tokens=1000)
        success = builder.add_chunk(sample_chunks[0])

        assert success is True
        assert builder.total_tokens > 0
        assert len(builder._parts) > 0

    def test_add_multiple_chunks(self, sample_chunks):
        """Test adding multiple chunks."""
        builder = ContextBuilder(max_tokens=1000)
        added = builder.add_chunks(sample_chunks)

        assert added == len(sample_chunks)
        assert builder.total_tokens > 0

    def test_build_returns_string(self, sample_chunks):
        """Test that build() returns a string."""
        builder = ContextBuilder(max_tokens=1000)
        builder.add_chunks(sample_chunks)
        context = builder.build()

        assert isinstance(context, str)
        assert len(context) > 0

    def test_respects_max_tokens(self, sample_chunks):
        """Test that builder respects max_tokens limit."""
        builder = ContextBuilder(max_tokens=50)  # Very small limit
        builder.add_chunks(sample_chunks)

        assert builder.total_tokens <= 50 * 1.2  # Allow small tolerance

    def test_stops_adding_when_full(self, sample_chunks):
        """Test that adding stops when token limit reached."""
        builder = ContextBuilder(max_tokens=20)  # Very small
        added = builder.add_chunks(sample_chunks)

        # Should not add all chunks
        assert added < len(sample_chunks)


# =============================================================================
# Test: ContextBuilder - Sources Tracking
# =============================================================================

class TestContextBuilderSources:
    """Tests for source tracking functionality."""

    def test_tracks_sources(self, sample_chunks):
        """Test that sources are tracked."""
        builder = ContextBuilder(max_tokens=1000)
        builder.add_chunks(sample_chunks)

        sources = builder.get_sources()

        assert len(sources) == len(sample_chunks)
        assert all(isinstance(s, dict) for s in sources)

    def test_get_unique_doc_ids(self, sample_chunks):
        """Test getting unique doc IDs."""
        builder = ContextBuilder(max_tokens=1000)
        builder.add_chunks(sample_chunks)

        doc_ids = builder.get_unique_doc_ids()

        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert len(doc_ids) == 2  # Only 2 unique docs

    def test_sources_have_required_fields(self, sample_chunks):
        """Test that sources have required fields."""
        builder = ContextBuilder(max_tokens=1000)
        builder.add_chunks(sample_chunks)

        sources = builder.get_sources()

        for source in sources:
            assert "type" in source
            assert "id" in source
            assert "doc_id" in source


# =============================================================================
# Test: ContextBuilder - Scored Chunks
# =============================================================================

class TestContextBuilderScoredChunks:
    """Tests for handling scored chunks."""

    def test_add_scored_chunk(self, scored_chunks):
        """Test adding a single scored chunk."""
        builder = ContextBuilder(max_tokens=1000)
        success = builder.add_scored_chunk(scored_chunks[0])

        assert success is True

    def test_add_scored_chunks(self, scored_chunks):
        """Test adding multiple scored chunks."""
        builder = ContextBuilder(max_tokens=1000)
        added = builder.add_scored_chunks(scored_chunks)

        assert added == len(scored_chunks)

    def test_scored_chunks_use_score(self, scored_chunks):
        """Test that scored chunks use their score."""
        builder = ContextBuilder(max_tokens=1000)

        # Custom format function that uses score
        def custom_format(chunk, score):
            return f"Score: {score}\n{chunk.text}"

        builder.add_scored_chunk(scored_chunks[0], format_fn=custom_format)
        context = builder.build()

        assert "Score:" in context


# =============================================================================
# Test: ContextBuilder - Custom Formatting
# =============================================================================

class TestContextBuilderFormatting:
    """Tests for custom formatting."""

    def test_custom_format_function(self, sample_chunks):
        """Test using a custom format function."""
        builder = ContextBuilder(max_tokens=1000)

        def custom_format(chunk, score):
            return f"CUSTOM: {chunk.doc_id}"

        builder.add_chunk(sample_chunks[0], format_fn=custom_format)
        context = builder.build()

        assert "CUSTOM:" in context

    def test_default_format_includes_doc_id(self, sample_chunks):
        """Test that default format includes doc_id."""
        builder = ContextBuilder(max_tokens=1000)
        builder.add_chunk(sample_chunks[0])
        context = builder.build()

        assert "doc1" in context

    def test_header_added(self):
        """Test that header is added when specified."""
        builder = ContextBuilder(max_tokens=1000, header="TEST HEADER")
        context = builder.build()

        assert "TEST HEADER" in context

    def test_separator_used(self, sample_chunks):
        """Test that separator is used between chunks."""
        builder = ContextBuilder(max_tokens=1000, separator="\n---SEP---\n")
        builder.add_chunks(sample_chunks[:2])
        context = builder.build()

        assert "---SEP---" in context


# =============================================================================
# Test: Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_build_context_simple(self, sample_chunks):
        """Test simple build_context helper."""
        context = build_context(sample_chunks, max_tokens=1000)

        assert isinstance(context, str)
        assert len(context) > 0

    def test_build_context_with_sources(self, sample_chunks):
        """Test build_context_with_sources helper."""
        context, sources = build_context_with_sources(sample_chunks, max_tokens=1000)

        assert isinstance(context, str)
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_build_private_context(self, scored_chunks):
        """Test build_private_context helper."""
        # Add company_id to metadata
        for sc in scored_chunks:
            sc.chunk.metadata["company_id"] = "ACME-123"
            sc.chunk.metadata["type"] = "history"
            sc.chunk.metadata["source_id"] = "hist-1"

        context, sources = build_private_context(
            scored_chunks, company_id="ACME-123", max_tokens=1000
        )

        assert isinstance(context, str)
        assert "ACME-123" in context  # Company ID in header
        assert isinstance(sources, list)

    def test_build_docs_context(self, scored_chunks):
        """Test build_docs_context helper."""
        context, sources = build_docs_context(scored_chunks, max_tokens=1000)

        assert isinstance(context, str)
        assert "PRODUCT DOCS" in context  # Header
        assert isinstance(sources, list)

    def test_build_docs_context_empty(self):
        """Test build_docs_context with empty chunks."""
        context, sources = build_docs_context([], max_tokens=1000)

        assert context == ""
        assert sources == []


# =============================================================================
# Test: Token Counting
# =============================================================================

class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_token_count_increases(self, sample_chunks):
        """Test that token count increases as chunks are added."""
        builder = ContextBuilder(max_tokens=1000)

        initial_tokens = builder.total_tokens
        builder.add_chunk(sample_chunks[0])

        assert builder.total_tokens > initial_tokens

    def test_remaining_tokens_decreases(self, sample_chunks):
        """Test that remaining tokens decreases."""
        builder = ContextBuilder(max_tokens=1000)

        initial_remaining = builder.remaining_tokens
        builder.add_chunk(sample_chunks[0])

        assert builder.remaining_tokens < initial_remaining

    def test_total_plus_remaining_equals_max(self, sample_chunks):
        """Test that total + remaining = max (approximately)."""
        builder = ContextBuilder(max_tokens=1000)
        builder.add_chunk(sample_chunks[0])

        # Allow small tolerance due to calculation differences
        assert abs((builder.total_tokens + builder.remaining_tokens) - builder.max_tokens) < 10


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestContextBuilderEdgeCases:
    """Tests for edge cases."""

    def test_zero_max_tokens(self, sample_chunks):
        """Test with max_tokens=0."""
        builder = ContextBuilder(max_tokens=0)
        added = builder.add_chunks(sample_chunks)

        assert added == 0  # Should not add anything

    def test_very_large_chunk(self):
        """Test with chunk larger than max_tokens."""
        large_chunk = DocumentChunk(
            chunk_id="large::0",
            doc_id="large",
            title="Large",
            text="word " * 1000,
            metadata={},
        )

        builder = ContextBuilder(max_tokens=100)
        success = builder.add_chunk(large_chunk)

        # Should attempt partial add
        assert isinstance(success, bool)

    def test_empty_chunk_list(self):
        """Test adding empty chunk list."""
        builder = ContextBuilder(max_tokens=1000)
        added = builder.add_chunks([])

        assert added == 0

    def test_chunk_with_no_text(self):
        """Test chunk with empty text."""
        empty_chunk = DocumentChunk(
            chunk_id="empty::0",
            doc_id="empty",
            title="Empty",
            text="",
            metadata={},
        )

        builder = ContextBuilder(max_tokens=1000)
        success = builder.add_chunk(empty_chunk)

        # Should handle gracefully
        assert isinstance(success, bool)

    def test_custom_token_estimator(self, sample_chunks):
        """Test using custom token estimator."""
        def custom_estimator(text):
            return len(text)  # Use character count

        builder = ContextBuilder(max_tokens=1000, token_estimator=custom_estimator)
        builder.add_chunk(sample_chunks[0])

        # Should use custom estimator
        assert builder.total_tokens > 0
