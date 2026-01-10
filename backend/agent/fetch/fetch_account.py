"""Account context RAG fetch node for LangGraph sequential execution."""

import logging
import time

from backend.agent.core.state import AgentState
from backend.agent.fetch.rag import call_account_rag

logger = logging.getLogger(__name__)


def fetch_account_node(state: AgentState) -> AgentState:
    """
    Fetch account context via RAG (conditional on needs_account_rag and company_id).

    The LLM query planner decides whether RAG is needed (needs_account_rag=True).
    company_id is resolved by fetch_crm from SQL query results.
    """
    start_time = time.time()

    # Check boolean flag set by query planner
    needs_rag = state.get("needs_account_rag", False)
    if not needs_rag:
        logger.info("[FetchAccount] Skipped (needs_account_rag=False)")
        return {"account_context_answer": "", "account_rag_invoked": False}

    # Get company_id resolved by fetch_crm from SQL results
    company_id = state.get("resolved_company_id")
    if not company_id:
        logger.info("[FetchAccount] Skipped (no company_id resolved)")
        return {"account_context_answer": "", "account_rag_invoked": False}

    question = state.get("question", "")

    logger.info(f"[FetchAccount] Searching account context for {company_id}...")

    try:
        account_answer, account_sources = call_account_rag(
            question=question,
            company_id=company_id,
        )

        # Split combined context back into individual chunks for RAGAS evaluation
        # The RAG tool joins chunks with "\n\n---\n\n"
        context_chunks = account_answer.split("\n\n---\n\n") if account_answer else []

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[FetchAccount] Complete in {latency_ms}ms, "
            f"sources={len(account_sources)}, chunks={len(context_chunks)}"
        )

        return {
            "account_context_answer": account_answer,
            "sources": account_sources,  # Uses reducer to merge with other sources
            "account_chunks": context_chunks,  # Individual chunks for RAGAS (account source)
            "account_rag_invoked": True,  # RAG was called (even if empty results)
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[FetchAccount] Failed after {latency_ms}ms: {e}")
        return {
            "account_context_answer": "",
            "account_rag_invoked": True,
            "error": f"Account fetch failed: {e}",
        }


__all__ = ["fetch_account_node"]
