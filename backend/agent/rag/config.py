"""
RAG configuration constants.

Centralized configuration for Qdrant paths, collection names, and models.
"""

from pathlib import Path

# Directory paths
_BACKEND_ROOT = Path(__file__).parent.parent.parent
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
DOCS_DIR = _BACKEND_ROOT / "data" / "docs"
JSONL_PATH = _BACKEND_ROOT / "data" / "csv" / "private_texts.jsonl"

# Collection names
DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_crm_private"

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


__all__ = [
    "QDRANT_PATH",
    "DOCS_DIR",
    "JSONL_PATH",
    "DOCS_COLLECTION",
    "PRIVATE_COLLECTION",
    "EMBEDDING_MODEL",
]
