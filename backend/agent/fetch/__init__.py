"""
Fetch nodes - parallel data retrieval for LangGraph workflow.

Exports:
    fetch_crm_node: Fetch CRM data based on intent
    fetch_account_node: Fetch account context via RAG
"""

from backend.agent.fetch.fetch_account import fetch_account_node
from backend.agent.fetch.fetch_crm import fetch_crm_node

__all__ = ["fetch_crm_node", "fetch_account_node"]
