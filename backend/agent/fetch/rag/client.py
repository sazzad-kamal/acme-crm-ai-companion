"""
Qdrant client singleton.

Provides thread-safe access to Qdrant client.
"""

import logging
import threading

from qdrant_client import QdrantClient

from backend.agent.fetch.rag.config import QDRANT_PATH

logger = logging.getLogger(__name__)


# =============================================================================
# Singleton Qdrant Client
# =============================================================================

_qdrant_client: QdrantClient | None = None
_qdrant_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Get shared Qdrant client (singleton)."""
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        return _qdrant_client


def close_qdrant_client() -> None:
    """Close the shared Qdrant client (for cleanup)."""
    global _qdrant_client
    with _qdrant_lock:
        if _qdrant_client is not None:
            _qdrant_client.close()
            _qdrant_client = None


__all__ = [
    "get_qdrant_client",
    "close_qdrant_client",
]
