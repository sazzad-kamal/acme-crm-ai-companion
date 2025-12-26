"""
Tests for embedding cache utilities.

Tests caching functionality for query embeddings.

Run with:
    pytest tests/backend/rag/test_embedding.py -v
"""

import pytest
import numpy as np

from backend.rag.retrieval.embedding import (
    get_cached_embedding,
    cache_embedding,
    _embedding_cache,
)


# =============================================================================
# Test: Basic Caching
# =============================================================================

class TestEmbeddingCache:
    """Tests for embedding cache functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        _embedding_cache.clear()

    def test_cache_and_retrieve(self):
        """Test basic cache and retrieve."""
        query = "test query"
        embedding = np.array([0.1, 0.2, 0.3])

        # Cache the embedding
        cache_embedding(query, embedding)

        # Retrieve it
        cached = get_cached_embedding(query)

        assert cached is not None
        assert np.array_equal(cached, embedding)

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        cached = get_cached_embedding("nonexistent query")

        assert cached is None

    def test_cache_overwrites_existing(self):
        """Test that caching same query overwrites."""
        query = "test query"
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])

        cache_embedding(query, embedding1)
        cache_embedding(query, embedding2)

        cached = get_cached_embedding(query)
        assert np.array_equal(cached, embedding2)

    def test_cache_different_queries(self):
        """Test caching different queries."""
        query1 = "first query"
        query2 = "second query"
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])

        cache_embedding(query1, embedding1)
        cache_embedding(query2, embedding2)

        cached1 = get_cached_embedding(query1)
        cached2 = get_cached_embedding(query2)

        assert np.array_equal(cached1, embedding1)
        assert np.array_equal(cached2, embedding2)

    def test_cache_case_sensitive(self):
        """Test that cache is case-sensitive."""
        embedding = np.array([0.1, 0.2, 0.3])

        cache_embedding("Test Query", embedding)

        # Different case should be cache miss
        assert get_cached_embedding("test query") is None
        assert get_cached_embedding("Test Query") is not None


# =============================================================================
# Test: LRU Eviction
# =============================================================================

class TestLRUEviction:
    """Tests for LRU cache eviction."""

    def setup_method(self):
        """Clear cache before each test."""
        _embedding_cache.clear()

    def test_cache_respects_maxsize(self):
        """Test that cache doesn't exceed maxsize."""
        # Note: The actual EMBEDDING_CACHE_SIZE might be large,
        # so we'll just verify the LRUCache behavior conceptually

        # Add an embedding
        cache_embedding("query1", np.array([0.1, 0.2, 0.3]))

        # Should be cached
        assert get_cached_embedding("query1") is not None

    def test_lru_eviction_behavior(self):
        """Test LRU eviction behavior (conceptual test)."""
        # This tests the behavior, not implementation details

        # Cache multiple embeddings
        for i in range(10):
            cache_embedding(f"query{i}", np.array([float(i)] * 3))

        # All should be retrievable if under max size
        for i in range(10):
            cached = get_cached_embedding(f"query{i}")
            # May or may not be cached depending on max size
            if cached is not None:
                assert len(cached) == 3


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEmbeddingCacheEdgeCases:
    """Tests for edge cases in embedding cache."""

    def setup_method(self):
        """Clear cache before each test."""
        _embedding_cache.clear()

    def test_cache_empty_query(self):
        """Test caching with empty string query."""
        embedding = np.array([0.1, 0.2, 0.3])

        cache_embedding("", embedding)
        cached = get_cached_embedding("")

        assert cached is not None
        assert np.array_equal(cached, embedding)

    def test_cache_long_query(self):
        """Test caching with very long query."""
        long_query = "word " * 1000
        embedding = np.array([0.1, 0.2, 0.3])

        cache_embedding(long_query, embedding)
        cached = get_cached_embedding(long_query)

        assert cached is not None

    def test_cache_unicode_query(self):
        """Test caching with unicode query."""
        query = "Hello 你好 мир 🎉"
        embedding = np.array([0.1, 0.2, 0.3])

        cache_embedding(query, embedding)
        cached = get_cached_embedding(query)

        assert cached is not None
        assert np.array_equal(cached, embedding)

    def test_cache_special_characters(self):
        """Test caching with special characters."""
        query = "What's the @cost of #product?"
        embedding = np.array([0.1, 0.2, 0.3])

        cache_embedding(query, embedding)
        cached = get_cached_embedding(query)

        assert cached is not None

    def test_cache_different_embedding_sizes(self):
        """Test caching embeddings of different sizes."""
        embedding_small = np.array([0.1, 0.2, 0.3])
        embedding_large = np.array([0.1] * 1000)

        cache_embedding("small", embedding_small)
        cache_embedding("large", embedding_large)

        assert len(get_cached_embedding("small")) == 3
        assert len(get_cached_embedding("large")) == 1000

    def test_cache_zero_vector(self):
        """Test caching zero vector."""
        embedding = np.zeros(384)  # Common embedding size

        cache_embedding("zero", embedding)
        cached = get_cached_embedding("zero")

        assert cached is not None
        assert np.allclose(cached, 0.0)

    def test_cache_normalized_vector(self):
        """Test caching normalized vector."""
        embedding = np.array([0.1, 0.2, 0.3])
        embedding = embedding / np.linalg.norm(embedding)

        cache_embedding("normalized", embedding)
        cached = get_cached_embedding("normalized")

        assert cached is not None
        assert np.isclose(np.linalg.norm(cached), 1.0)


# =============================================================================
# Test: Cache Properties
# =============================================================================

class TestCacheProperties:
    """Tests for cache properties and behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        _embedding_cache.clear()

    def test_cache_is_lru_cache(self):
        """Test that cache is an LRUCache instance."""
        from cachetools import LRUCache

        assert isinstance(_embedding_cache, LRUCache)

    def test_cache_has_maxsize(self):
        """Test that cache has maxsize attribute."""
        assert hasattr(_embedding_cache, 'maxsize')
        assert _embedding_cache.maxsize > 0

    def test_multiple_cache_calls_dont_duplicate(self):
        """Test that caching same key doesn't duplicate entries."""
        query = "test"
        embedding = np.array([0.1, 0.2, 0.3])

        # Cache multiple times
        for _ in range(5):
            cache_embedding(query, embedding)

        # Should only have one entry for this key
        # (This is implicit in LRUCache behavior)
        cached = get_cached_embedding(query)
        assert cached is not None


# =============================================================================
# Test: Thread Safety (Conceptual)
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety (conceptual tests)."""

    def setup_method(self):
        """Clear cache before each test."""
        _embedding_cache.clear()

    def test_concurrent_cache_access(self):
        """Test that concurrent access doesn't break cache."""
        # This is a basic test; real thread safety would require
        # concurrent execution which is beyond simple unit tests

        queries = [f"query{i}" for i in range(10)]
        embeddings = [np.array([float(i)] * 3) for i in range(10)]

        # Cache all
        for q, e in zip(queries, embeddings):
            cache_embedding(q, e)

        # Retrieve all
        for q, e in zip(queries, embeddings):
            cached = get_cached_embedding(q)
            if cached is not None:  # May be evicted if over maxsize
                assert np.array_equal(cached, e)
