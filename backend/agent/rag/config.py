"""
RAG configuration constants.

Centralized configuration for Qdrant paths, collection names, and models.
"""

from pathlib import Path

# Directory paths
_BACKEND_ROOT = Path(__file__).parent.parent.parent
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
JSONL_PATH = _BACKEND_ROOT / "data" / "csv" / "private_texts.jsonl"

# Collection names
PRIVATE_COLLECTION = "acme_crm_private"

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


__all__ = [
    "QDRANT_PATH",
    "JSONL_PATH",
    "PRIVATE_COLLECTION",
    "EMBEDDING_MODEL",
]
