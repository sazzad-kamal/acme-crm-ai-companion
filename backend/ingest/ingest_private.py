"""
Ingest private CRM text (history, notes, attachments) into Qdrant using LlamaIndex.

Reads from private_texts.jsonl which contains pre-processed CRM text data.

Usage:
    python -m backend.ingest.ingest_private
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths
_BACKEND_ROOT = Path(__file__).parent.parent
JSONL_PATH = _BACKEND_ROOT / "data" / "csv" / "private_texts.jsonl"
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
COLLECTION_NAME = "acme_crm_private"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def ingest_private_texts(recreate: bool = True) -> int:
    """
    Ingest private texts from JSONL into Qdrant.

    Args:
        recreate: If True, delete and recreate the collection

    Returns:
        Number of chunks ingested
    """
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    logger.info(f"Ingesting private texts from {JSONL_PATH}")

    # Configure LlamaIndex
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Load JSONL documents
    if not JSONL_PATH.exists():
        logger.warning(f"Private texts file not found: {JSONL_PATH}")
        return 0

    documents = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                doc = Document(
                    text=record.get("text", ""),
                    metadata={
                        "doc_id": record.get("id", ""),
                        "source_id": record.get("id", ""),
                        "company_id": record.get("company_id", ""),
                        "type": record.get("type", ""),
                        "title": record.get("title", ""),
                        "contact_id": record.get("contact_id"),
                        "opportunity_id": record.get("opportunity_id"),
                    },
                )
                documents.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    if not documents:
        logger.warning("No documents loaded from JSONL")
        return 0

    logger.info(f"Loaded {len(documents)} documents from JSONL")

    # Initialize Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Delete existing collection if recreate=True
    if recreate and client.collection_exists(COLLECTION_NAME):
        logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    # Build index (this ingests the documents)
    logger.info("Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True,
    )

    # Get final count
    info = client.get_collection(COLLECTION_NAME)
    chunk_count = info.points_count

    logger.info(f"Ingested {chunk_count} chunks into '{COLLECTION_NAME}'")

    # Close the client to release the lock
    client.close()

    return chunk_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_private_texts()
