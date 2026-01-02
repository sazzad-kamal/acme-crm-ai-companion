"""
Document ingestion for Qdrant vector store using LlamaIndex.

Provides functions to ingest:
- Documentation (markdown files) into acme_crm_docs collection
- Private CRM text (history, notes) into acme_crm_private collection

Usage:
    python -m backend.ingest
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

_BACKEND_ROOT = Path(__file__).parent
DOCS_DIR = _BACKEND_ROOT / "data" / "docs"
JSONL_PATH = _BACKEND_ROOT / "data" / "csv" / "private_texts.jsonl"
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"

DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_crm_private"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# =============================================================================
# Document Ingestion
# =============================================================================

def ingest_docs(recreate: bool = True) -> int:
    """
    Ingest all markdown docs into Qdrant.

    Args:
        recreate: If True, delete and recreate the collection

    Returns:
        Number of chunks ingested
    """
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    logger.info(f"Ingesting docs from {DOCS_DIR}")

    # Configure LlamaIndex
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Load markdown documents
    if not DOCS_DIR.exists():
        logger.warning(f"Docs directory not found: {DOCS_DIR}")
        return 0

    reader = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        required_exts=[".md"],
        recursive=False,
    )
    documents = reader.load_data()

    if not documents:
        logger.warning("No markdown documents found")
        return 0

    # Add doc_id metadata from filename
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        doc_id = Path(file_path).stem if file_path else "unknown"
        doc.metadata["doc_id"] = doc_id

    logger.info(f"Loaded {len(documents)} documents")

    # Initialize Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Delete existing collection if recreate=True
    if recreate and client.collection_exists(DOCS_COLLECTION):
        logger.info(f"Deleting existing collection: {DOCS_COLLECTION}")
        client.delete_collection(DOCS_COLLECTION)

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=DOCS_COLLECTION,
    )

    # Build index (this ingests the documents)
    logger.info("Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True,
    )

    # Get final count
    info = client.get_collection(DOCS_COLLECTION)
    chunk_count = info.points_count

    logger.info(f"Ingested {chunk_count} chunks into '{DOCS_COLLECTION}'")

    # Close the client to release the lock
    client.close()

    return chunk_count


# =============================================================================
# Private Text Ingestion
# =============================================================================

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
    if recreate and client.collection_exists(PRIVATE_COLLECTION):
        logger.info(f"Deleting existing collection: {PRIVATE_COLLECTION}")
        client.delete_collection(PRIVATE_COLLECTION)

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=PRIVATE_COLLECTION,
    )

    # Build index (this ingests the documents)
    logger.info("Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True,
    )

    # Get final count
    info = client.get_collection(PRIVATE_COLLECTION)
    chunk_count = info.points_count

    logger.info(f"Ingested {chunk_count} chunks into '{PRIVATE_COLLECTION}'")

    # Close the client to release the lock
    client.close()

    return chunk_count


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Ingesting documentation...")
    docs_count = ingest_docs()
    print(f"  {docs_count} doc chunks")

    print("Ingesting private texts...")
    private_count = ingest_private_texts()
    print(f"  {private_count} private chunks")

    print("Done!")
