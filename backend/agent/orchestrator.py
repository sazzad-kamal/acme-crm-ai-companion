"""
Agent orchestration - redirects to graph.py.

The agent is implemented using LangGraph in backend/agent/graph.py.
This module exists only for import convenience.
"""

from backend.agent.graph import answer_question, run_agent, agent_graph
from backend.agent.progress import AgentProgress

__all__ = ["answer_question", "run_agent", "agent_graph", "AgentProgress"]
