"""
Ingest markdown documentation into Qdrant using LlamaIndex.

Usage:
    python -m backend.ingest.ingest_docs
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths
_BACKEND_ROOT = Path(__file__).parent.parent
DOCS_DIR = _BACKEND_ROOT / "data" / "docs"
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
COLLECTION_NAME = "acme_crm_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


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
    ingest_docs()
