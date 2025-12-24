# backend.rag.ingest - Document Ingestion
"""
Ingestion scripts for RAG documents.

Modules:
- docs: Markdown document ingestion
- private_text: Private CRM text ingestion  
- text_builder: Private text JSONL builder

These are CLI scripts - import them directly to run:
    python -m backend.rag.ingest.docs
    python -m backend.rag.ingest.private_text
"""

# Export nothing by default - these are CLI modules
__all__: list[str] = []
