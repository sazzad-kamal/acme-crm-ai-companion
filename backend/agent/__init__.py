# backend.agent - Agent Orchestration
"""
Agent orchestration for CRM AI Companion.

Modules:
- config: Agent configuration
- schemas: Pydantic models for requests/responses
- datastore: CRM data access via DuckDB
- router: Rule-based mode routing
- llm_router: LLM-based routing with query understanding
- tools: Tool functions for data retrieval
- orchestrator: Main agent orchestration
- audit: Agent audit logging
- prompts: LLM prompt templates
- formatters: Context formatting functions
- llm_helpers: LLM call utilities with retry
- progress: Pipeline progress tracking
"""

from backend.agent.orchestrator import answer_question
from backend.agent.progress import AgentProgress

__all__ = ["answer_question", "AgentProgress"]
