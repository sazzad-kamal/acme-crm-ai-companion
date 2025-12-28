"""
LangGraph-based agent orchestration.

Implements a graph workflow for answering CRM questions:

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  Route  │  Determine mode & extract entities
    └────┬────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌─────────────┐
    │  Data   │      │  Docs   │      │ Data + Docs │
    │  Only   │      │  Only   │      │   (Both)    │
    └────┬────┘      └────┬────┘      └──────┬──────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Answer    │  Synthesize with LLM
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Follow-up  │  Generate suggestions
                    └──────┬──────┘
                           │
                           ▼
                       ┌───────┐
                       │  END  │
                       └───────┘

Usage:
    from backend.agent.graph import agent_graph, run_agent

    result = run_agent("What's going on with Acme Manufacturing?")
"""

import logging
import time
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.state import AgentState
from backend.agent.nodes import (
    route_node,
    data_node,
    docs_node,
    skip_data_node,
    skip_docs_node,
    data_and_docs_parallel_node,
    answer_node,
    followup_node,
    route_by_mode,
)
from backend.agent.audit import AgentAuditLogger
from backend.agent.memory import get_conversation_history, add_message


logger = logging.getLogger(__name__)


# =============================================================================
# LangGraph Checkpointing
# =============================================================================

# Global checkpointer for conversation persistence
_checkpointer = MemorySaver()


# =============================================================================
# Graph Construction
# =============================================================================

def build_agent_graph(checkpointer=None):
    """
    Build the LangGraph workflow.

    Uses RunnableParallel pattern for data+docs mode to fetch
    CRM data and documentation concurrently, reducing latency.

    Args:
        checkpointer: Optional LangGraph checkpointer for conversation persistence.
                     If None, uses the global MemorySaver.

    Returns compiled graph ready for execution.
    """
    # Create graph with state schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("data", data_node)
    graph.add_node("docs", docs_node)
    graph.add_node("skip_data", skip_data_node)
    graph.add_node("skip_docs", skip_docs_node)
    graph.add_node("data_and_docs", data_and_docs_parallel_node)  # Parallel node
    graph.add_node("answer", answer_node)
    graph.add_node("followup", followup_node)

    # Set entry point
    graph.set_entry_point("route")

    # Add conditional routing after route node
    # - data_only: fetch data only, skip docs
    # - docs_only: skip data, fetch docs only
    # - data_and_docs: use parallel node for concurrent fetching
    graph.add_conditional_edges(
        "route",
        route_by_mode,
        {
            "data_only": "data",
            "docs_only": "skip_data",
            "data_and_docs": "data_and_docs",  # Use parallel node
        }
    )

    # After skip_data (docs-only mode), go to docs
    graph.add_edge("skip_data", "docs")

    # After data (data-only mode), go to skip_docs
    graph.add_edge("data", "skip_docs")

    # Parallel node goes directly to answer (already fetched both)
    graph.add_edge("data_and_docs", "answer")

    # Docs leads to answer
    graph.add_edge("docs", "answer")

    # Skip docs leads to answer
    graph.add_edge("skip_docs", "answer")

    # Answer leads to followup
    graph.add_edge("answer", "followup")

    # Followup is the end
    graph.add_edge("followup", END)

    # Compile with checkpointer for conversation persistence
    return graph.compile(checkpointer=checkpointer or _checkpointer)


# Compile the graph once at module load (with checkpointing enabled)
agent_graph = build_agent_graph()


# =============================================================================
# Runner Function
# =============================================================================

def run_agent(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """
    Run the agent graph and return formatted response.

    Uses LangGraph checkpointing for conversation persistence when session_id
    is provided. The checkpointer automatically stores and retrieves conversation
    state based on the thread_id.

    Args:
        question: The user's question
        mode: Mode override ("auto", "docs", "data", "data+docs")
        company_id: Pre-specified company ID
        session_id: Optional session ID (used as thread_id for checkpointing)
        user_id: Optional user ID

    Returns:
        Dict matching ChatResponse schema
    """
    start_time = time.time()

    # Load conversation history for multi-turn support (legacy + fallback)
    messages = get_conversation_history(session_id)
    if messages:
        logger.debug(f"[Agent] Loaded {len(messages)} messages from session {session_id}")

    # Initialize state
    initial_state: AgentState = {
        "question": question,
        "mode": mode,
        "company_id": company_id,
        "session_id": session_id,
        "user_id": user_id,
        "messages": messages,
        "sources": [],
        "steps": [],
        "raw_data": {},
        "follow_up_suggestions": [],
    }

    # Build config with thread_id for LangGraph checkpointing
    # Always provide a thread_id (use session_id if provided, else generate one)
    thread_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    if session_id:
        logger.debug(f"[Agent] Using LangGraph checkpointing with thread_id={session_id}")

    # Run the graph
    logger.info(f"[Agent] Starting graph execution for: {question[:50]}...")

    try:
        # Invoke with config for checkpointing
        final_state = agent_graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"[Agent] Graph execution failed: {e}")
        return {
            "answer": f"I'm sorry, I encountered an error: {str(e)}",
            "sources": [],
            "steps": [{"id": "error", "label": f"Error: {str(e)[:50]}", "status": "error"}],
            "raw_data": {},
            "follow_up_suggestions": [],
            "meta": {
                "mode_used": "error",
                "latency_ms": int((time.time() - start_time) * 1000),
            }
        }

    latency_ms = int((time.time() - start_time) * 1000)

    # Audit logging
    audit = AgentAuditLogger()
    audit.log_query(
        question=question,
        mode_used=final_state.get("mode_used", "unknown"),
        company_id=final_state.get("resolved_company_id"),
        latency_ms=latency_ms,
        source_count=len(final_state.get("sources", [])),
        user_id=user_id,
        session_id=session_id,
    )

    logger.info(f"[Agent] Complete in {latency_ms}ms")

    # Save conversation to memory for multi-turn support (legacy compatibility)
    # Note: LangGraph checkpointing also persists state automatically
    resolved_company = final_state.get("resolved_company_id")
    add_message(session_id, "user", question, resolved_company)
    add_message(session_id, "assistant", final_state.get("answer", ""), resolved_company)

    # Build response
    return {
        "answer": final_state.get("answer", ""),
        "sources": [s.model_dump() if hasattr(s, 'model_dump') else s for s in final_state.get("sources", [])],
        "steps": final_state.get("steps", []),
        "raw_data": final_state.get("raw_data", {}),
        "follow_up_suggestions": final_state.get("follow_up_suggestions", []),
        "meta": {
            "mode_used": final_state.get("mode_used", "unknown"),
            "latency_ms": latency_ms,
            "company_id": final_state.get("resolved_company_id"),
            "days": final_state.get("days", 90),
        }
    }


# =============================================================================
# Backwards Compatibility
# =============================================================================

def answer_question(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """
    Backwards-compatible wrapper for run_agent.

    This maintains the same API as the previous orchestrator.
    """
    return run_agent(
        question=question,
        mode=mode,
        company_id=company_id,
        session_id=session_id,
        user_id=user_id,
    )


# =============================================================================
# Visualization
# =============================================================================

def get_graph_mermaid() -> str:
    """
    Get Mermaid diagram of the graph.

    Returns:
        Mermaid diagram string
    """
    return """
graph TD
    START((Start)) --> route[Route]
    route -->|data| data[Fetch CRM Data]
    route -->|docs| skip_data[Skip Data]
    route -->|data+docs| data_and_docs[Parallel Fetch<br/>Data + Docs]

    data --> skip_docs[Skip Docs]
    skip_data --> docs[Fetch Docs]

    docs --> answer[Synthesize Answer]
    skip_docs --> answer
    data_and_docs --> answer

    answer --> followup[Generate Follow-ups]
    followup --> END((End))

    style route fill:#e1f5fe
    style data fill:#fff3e0
    style docs fill:#f3e5f5
    style data_and_docs fill:#ffecb3
    style answer fill:#e8f5e9
    style followup fill:#fce4ec
"""


def print_graph_ascii() -> None:
    """Print ASCII representation of the graph."""
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │              LANGGRAPH AGENT (with Parallel Fetch)          │
    └─────────────────────────────────────────────────────────────┘

                           ┌─────────┐
                           │  START  │
                           └────┬────┘
                                │
                                ▼
                           ┌─────────┐
                           │  Route  │  (LLM structured output)
                           └────┬────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
       ┌─────────┐        ┌─────────┐        ┌───────────────┐
       │  Data   │        │  Docs   │        │  Data + Docs  │
       │  Only   │        │  Only   │        │  (Parallel)   │
       └────┬────┘        └────┬────┘        └───────┬───────┘
            │                  │                     │
            ▼                  ▼                     │
       ┌─────────┐        ┌─────────┐               │
       │Skip Docs│        │Fetch Doc│               │
       └────┬────┘        └────┬────┘               │
            │                  │                     │
            └──────────────────┴─────────────────────┘
                                │
                                ▼
                         ┌─────────────┐
                         │   Answer    │  (LCEL chain)
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │  Follow-up  │  (Structured output)
                         └──────┬──────┘
                                │
                                ▼
                            ┌───────┐
                            │  END  │
                            └───────┘
    """)


def get_checkpointer() -> MemorySaver:
    """Get the global checkpointer instance."""
    return _checkpointer


def get_session_state(session_id: str) -> dict | None:
    """
    Get the checkpointed state for a session.

    Args:
        session_id: The session/thread ID

    Returns:
        The stored state dict, or None if not found
    """
    try:
        config = {"configurable": {"thread_id": session_id}}
        checkpoint = _checkpointer.get(config)
        if checkpoint:
            return checkpoint.get("channel_values", {})
    except Exception as e:
        logger.warning(f"Failed to get session state: {e}")
    return None


__all__ = [
    "agent_graph",
    "run_agent",
    "answer_question",
    "build_agent_graph",
    "get_graph_mermaid",
    "print_graph_ascii",
    # Checkpointing
    "get_checkpointer",
    "get_session_state",
]


# =============================================================================
# CLI / Test
# =============================================================================

if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    print("\n" + "=" * 60)
    print("LangGraph Agent Workflow")
    print("=" * 60)

    print_graph_ascii()

    # Test if MOCK_LLM is set
    if os.environ.get("MOCK_LLM"):
        print("\n[MOCK MODE ENABLED]")

    print("\n" + "-" * 60)
    print("Testing Agent Graph")
    print("-" * 60)

    test_questions = [
        # Company-specific queries
        ("What's going on with Acme Manufacturing?", "auto"),

        # Documentation queries
        ("How do I create a new opportunity?", "auto"),

        # Renewals
        ("What renewals are coming up?", "auto"),

        # Pipeline summary (aggregate)
        ("What's the total pipeline value?", "auto"),

        # Contact search
        ("Who are the decision makers?", "auto"),

        # Company search
        ("Show me all enterprise accounts", "auto"),

        # Groups
        ("Who is in the at-risk accounts group?", "auto"),

        # Attachments
        ("Find all proposals", "auto"),
    ]

    for q, mode in test_questions:
        print(f"\nQ: {q}")
        print(f"Mode: {mode}")

        result = run_agent(q, mode=mode)

        print(f"  → Mode used: {result['meta']['mode_used']}")
        print(f"  → Company: {result['meta'].get('company_id', 'None')}")
        print(f"  → Sources: {len(result['sources'])}")
        print(f"  → Latency: {result['meta']['latency_ms']}ms")
        print(f"  → Steps: {[s['id'] for s in result['steps']]}")
        print(f"  → Answer: {result['answer'][:100]}...")
