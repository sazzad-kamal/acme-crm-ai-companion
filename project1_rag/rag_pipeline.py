"""
RAG Pipeline for Acme CRM docs.

Implements:
- Query preprocessing and normalization
- Query rewriting (clarification)
- HyDE (Hypothetical Document Embeddings)
- Retrieval with gating and per-doc caps
- Context building
- Answer generation with citations

Usage:
    from project1_rag.rag_pipeline import answer_question
    from project1_rag.retrieval_backend import create_backend
    
    backend = create_backend()
    result = answer_question("What is an Opportunity?", backend)
    print(result["answer"])
"""

import re
from typing import Optional
from collections import defaultdict

from project1_rag.doc_models import DocumentChunk, ScoredChunk
from project1_rag.retrieval_backend import RetrievalBackend
from shared.llm_client import call_llm, call_llm_with_metrics


# =============================================================================
# Configuration
# =============================================================================

# Context building
MAX_CONTEXT_TOKENS = 3000
MAX_CHUNKS_PER_DOC = 3
MIN_BM25_SCORE_RATIO = 0.1  # Drop chunks with BM25 < 10% of top score

# Approximate tokens per character
CHARS_PER_TOKEN = 4


# =============================================================================
# Prompts
# =============================================================================

QUERY_REWRITE_SYSTEM = """You are a query rewriting assistant for a CRM documentation search system.
Your job is to take a user's question about Acme CRM Suite and rewrite it to be clearer and more specific.
Keep the rewritten query in natural language (not keywords).
If the query is already clear, return it mostly unchanged.
Only output the rewritten query, nothing else."""

HYDE_SYSTEM = """You are an expert on Acme CRM Suite documentation.
Given a question, write a short hypothetical answer (2-3 sentences) as if it came from the documentation.
This will be used for semantic search, so include relevant terminology and concepts.
Only output the hypothetical answer, nothing else."""

ANSWER_SYSTEM = """You are an AI assistant answering questions about Acme CRM Suite.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. If the answer is not in the context, say "I don't see this documented in the provided sources."
3. Cite your sources using [doc_id] format, e.g., [opportunities_pipeline_and_forecasts].
4. Be concise but complete.
5. If multiple docs cover different aspects, synthesize the information and cite all relevant sources.

Context from Acme CRM Suite documentation:
{context}

Question: {question}

Answer (with citations):"""


# =============================================================================
# Pipeline Components
# =============================================================================

def preprocess_query(query: str) -> str:
    """
    Light preprocessing of the query.
    
    - Strip whitespace
    - Collapse multiple spaces
    - Remove excessive punctuation
    """
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    return query


def rewrite_query(query: str) -> str:
    """
    Use LLM to rewrite vague queries into clearer ones.
    
    Args:
        query: Original user query
        
    Returns:
        Rewritten query (or original if rewriting fails)
    """
    try:
        rewritten = call_llm(
            prompt=f"Rewrite this CRM question to be clearer: {query}",
            system_prompt=QUERY_REWRITE_SYSTEM,
            max_tokens=150,
        )
        return rewritten.strip() or query
    except Exception as e:
        print(f"Warning: Query rewrite failed: {e}")
        return query


def generate_hyde_answer(query: str) -> str:
    """
    Generate a hypothetical answer for HyDE retrieval.
    
    Args:
        query: The user's question
        
    Returns:
        A hypothetical answer to use for embedding
    """
    try:
        hyde = call_llm(
            prompt=f"Question: {query}",
            system_prompt=HYDE_SYSTEM,
            max_tokens=200,
        )
        return hyde.strip()
    except Exception as e:
        print(f"Warning: HyDE generation failed: {e}")
        return ""


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def apply_lexical_gate(
    scored_chunks: list[ScoredChunk],
    min_ratio: float = MIN_BM25_SCORE_RATIO,
) -> list[ScoredChunk]:
    """
    Filter out chunks with very low BM25 scores (lexical gate).
    
    Args:
        scored_chunks: List of scored chunks from retrieval
        min_ratio: Minimum BM25 score as ratio of top score
        
    Returns:
        Filtered list of scored chunks
    """
    if not scored_chunks:
        return []
    
    # Find max BM25 score
    max_bm25 = max(sc.bm25_score for sc in scored_chunks)
    
    if max_bm25 <= 0:
        return scored_chunks  # Can't filter by BM25
    
    threshold = max_bm25 * min_ratio
    
    return [sc for sc in scored_chunks if sc.bm25_score >= threshold]


def apply_per_doc_cap(
    scored_chunks: list[ScoredChunk],
    max_per_doc: int = MAX_CHUNKS_PER_DOC,
) -> list[ScoredChunk]:
    """
    Limit the number of chunks per document.
    
    Args:
        scored_chunks: List of scored chunks (assumed sorted by relevance)
        max_per_doc: Maximum chunks to keep per doc_id
        
    Returns:
        Filtered list respecting per-doc cap
    """
    doc_counts = defaultdict(int)
    filtered = []
    
    for sc in scored_chunks:
        doc_id = sc.chunk.doc_id
        if doc_counts[doc_id] < max_per_doc:
            filtered.append(sc)
            doc_counts[doc_id] += 1
    
    return filtered


def build_context(
    chunks: list[DocumentChunk],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    """
    Build a context string from chunks for the LLM prompt.
    
    Args:
        chunks: List of document chunks to include
        max_tokens: Maximum tokens for the context
        
    Returns:
        Formatted context string with doc_id labels
    """
    context_parts = []
    total_tokens = 0
    
    for chunk in chunks:
        # Format: [doc_id] Section: text
        section = chunk.metadata.get("section_heading", "")
        header = f"[{chunk.doc_id}]"
        if section:
            header += f" {section}"
        
        chunk_text = f"{header}\n{chunk.text}\n"
        chunk_tokens = estimate_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            # Try to fit partial text
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:
                truncated = chunk.text[:remaining_tokens * CHARS_PER_TOKEN]
                context_parts.append(f"{header}\n{truncated}...")
            break
        
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    return "\n---\n".join(context_parts)


def generate_answer(
    question: str,
    context: str,
    chunks_used: list[DocumentChunk],
) -> dict:
    """
    Generate an answer using the LLM with the provided context.
    
    Args:
        question: The user's question
        context: Formatted context string
        chunks_used: List of chunks used (for metadata)
        
    Returns:
        Dict with answer and metadata
    """
    prompt = ANSWER_SYSTEM.format(context=context, question=question)
    
    result = call_llm_with_metrics(
        prompt=prompt,
        model="gpt-4.1-mini",
        max_tokens=800,
    )
    
    return {
        "answer": result["response"],
        "latency_ms": result["latency_ms"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def answer_question(
    question: str,
    backend: RetrievalBackend,
    *,
    k: int = 8,
    use_hyde: bool = True,
    use_rewrite: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Full RAG pipeline to answer a question.
    
    Args:
        question: User's question
        backend: Initialized RetrievalBackend
        k: Number of chunks to retrieve
        use_hyde: Whether to use HyDE for retrieval
        use_rewrite: Whether to rewrite the query
        verbose: Print debug information
        
    Returns:
        Dict containing:
        - answer: The generated answer
        - used_chunks: List of DocumentChunk objects used
        - rewritten_question: The rewritten query
        - hyde_answer: The hypothetical answer (if used)
        - doc_ids_used: List of unique doc_ids cited
        - metrics: Dict with latency, token counts, etc.
    """
    metrics = {
        "total_latency_ms": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }
    
    # 1. Preprocess
    question = preprocess_query(question)
    
    if verbose:
        print(f"Original question: {question}")
    
    # 2. Query rewrite
    rewritten_question = question
    if use_rewrite:
        rewritten_question = rewrite_query(question)
        metrics["llm_calls"] += 1
        if verbose:
            print(f"Rewritten question: {rewritten_question}")
    
    # 3. HyDE
    hyde_answer = ""
    retrieval_query = rewritten_question
    if use_hyde:
        hyde_answer = generate_hyde_answer(rewritten_question)
        metrics["llm_calls"] += 1
        if hyde_answer:
            # Use both rewritten query and HyDE for retrieval
            retrieval_query = f"{rewritten_question} {hyde_answer}"
        if verbose:
            print(f"HyDE answer: {hyde_answer[:100]}...")
    
    # 4. Retrieval
    scored_chunks = backend.retrieve_candidates(
        query=retrieval_query,
        k_dense=k * 3,
        k_bm25=k * 3,
        top_n=k * 2,
        use_reranker=True,
    )
    
    if verbose:
        print(f"Retrieved {len(scored_chunks)} candidates after reranking")
    
    # 5. Gating
    # Apply lexical gate
    gated_chunks = apply_lexical_gate(scored_chunks)
    if verbose:
        print(f"After lexical gate: {len(gated_chunks)} chunks")
    
    # Apply per-doc cap
    capped_chunks = apply_per_doc_cap(gated_chunks, max_per_doc=MAX_CHUNKS_PER_DOC)
    if verbose:
        print(f"After per-doc cap: {len(capped_chunks)} chunks")
    
    # Take top k
    final_chunks = [sc.chunk for sc in capped_chunks[:k]]
    
    if verbose:
        print(f"Final chunks: {len(final_chunks)}")
        for i, chunk in enumerate(final_chunks):
            print(f"  {i+1}. [{chunk.doc_id}] {chunk.text[:50]}...")
    
    # 6. Build context
    context = build_context(final_chunks, max_tokens=MAX_CONTEXT_TOKENS)
    
    if verbose:
        print(f"Context size: ~{estimate_tokens(context)} tokens")
    
    # 7. Generate answer
    answer_result = generate_answer(question, context, final_chunks)
    metrics["llm_calls"] += 1
    metrics["total_latency_ms"] += answer_result["latency_ms"]
    metrics["total_tokens"] += answer_result["total_tokens"]
    
    # Extract doc_ids from answer (look for [doc_id] citations)
    citation_pattern = r'\[([a-z_]+)\]'
    cited_docs = set(re.findall(citation_pattern, answer_result["answer"].lower()))
    
    # Get all doc_ids from used chunks
    used_doc_ids = list(set(c.doc_id for c in final_chunks))
    
    return {
        "answer": answer_result["answer"],
        "used_chunks": final_chunks,
        "rewritten_question": rewritten_question,
        "hyde_answer": hyde_answer,
        "doc_ids_used": used_doc_ids,
        "cited_docs": list(cited_docs),
        "num_chunks_used": len(final_chunks),
        "context_tokens": estimate_tokens(context),
        "metrics": {
            "answer_latency_ms": answer_result["latency_ms"],
            "prompt_tokens": answer_result["prompt_tokens"],
            "completion_tokens": answer_result["completion_tokens"],
            "total_tokens": answer_result["total_tokens"],
            "llm_calls": metrics["llm_calls"],
        },
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from project1_rag.retrieval_backend import create_backend
    
    print("Testing RAG Pipeline")
    print("=" * 60)
    
    backend = create_backend()
    
    question = "What is an Opportunity in Acme CRM Suite and what fields does it have?"
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    result = answer_question(question, backend, verbose=True)
    
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\n" + "-" * 60)
    print(f"Doc IDs used: {result['doc_ids_used']}")
    print(f"Cited docs: {result['cited_docs']}")
    print(f"Chunks used: {result['num_chunks_used']}")
    print(f"Metrics: {result['metrics']}")
