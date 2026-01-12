"""LangGraph agent orchestration: Route → fetch_sql → fetch_rag → Answer → Followup."""

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.agent.answer.node import answer_node
from backend.agent.core.state import AgentState
from backend.agent.fetch.fetch_sql import fetch_sql_node
from backend.agent.fetch.fetch_rag import fetch_rag_node
from backend.agent.followup.node import followup_node
from backend.agent.route.node import route_node

# LangGraph event constants (not exported by langgraph package)
class LangGraphEvent:
    CHAIN_START = "on_chain_start"
    CHAIN_END = "on_chain_end"
    CHAT_MODEL_STREAM = "on_chat_model_stream"

GRAPH_NAME = "LangGraph"  # Name used for whole graph in events

# Our node names
ANSWER_NODE = "answer"

_checkpointer = MemorySaver()


def build_thread_config(session_id: str | None) -> dict:
    """Build LangGraph config with thread_id for checkpointing."""
    return {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}


def _build_graph():
    """Build the LangGraph workflow with sequential fetch nodes."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("fetch_sql", fetch_sql_node)
    graph.add_node("fetch_rag", fetch_rag_node)
    graph.add_node(ANSWER_NODE, answer_node)
    graph.add_node("followup", followup_node)

    # Entry point
    graph.set_entry_point("route")

    # Sequential flow: route → fetch_sql → fetch_rag → answer → followup
    graph.add_edge("route", "fetch_sql")
    graph.add_edge("fetch_sql", "fetch_rag")
    graph.add_edge("fetch_rag", ANSWER_NODE)
    graph.add_edge(ANSWER_NODE, "followup")
    graph.add_edge("followup", END)

    return graph.compile(checkpointer=_checkpointer)


agent_graph = _build_graph()

__all__ = ["agent_graph", "build_thread_config", "LangGraphEvent", "GRAPH_NAME", "ANSWER_NODE"]
