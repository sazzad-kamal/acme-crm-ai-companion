# project1_rag - Docs-only RAG experiment for Acme CRM
"""
A self-contained RAG experiment that:
- Ingests and chunks Markdown docs
- Builds hybrid retrieval (Qdrant + BM25)
- Implements query rewrite + HyDE
- Provides evaluation harness with RAG triad metrics
"""
