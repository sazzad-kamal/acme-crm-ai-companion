"""
Hardcoded question tree for demo reliability.

This module provides a deterministic tree of questions and follow-ups,
ensuring 100% reliability for demos - no LLM generation variability.

Structure:
- 3 starter questions covering the 3 top CRM entities:
  * Opportunities: "What deals are in the pipeline?"
  * Companies: "Which accounts are at risk?"
  * Contacts: "Which contacts haven't been contacted recently?"
- Each question has 3 follow-ups, with varying depths (4-6 levels)

Usage:
    from backend.agent.followup.tree import get_follow_ups, get_starters

    # Get starter questions
    starters = get_starters()

    # Get follow-ups for a question
    follow_ups = get_follow_ups("What deals are in the pipeline?")

Eval-specific functions (get_expected_*, validate_*, get_all_paths, etc.)
are in backend.eval.tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

__all__ = [
    "get_starters",
    "get_follow_ups",
    "get_graph",
]

# =============================================================================
# Load JSON and Build Graph
# =============================================================================

_DATA_PATH = Path(__file__).parent / "data.json"
with open(_DATA_PATH) as f:
    _raw_data: dict[str, list[str]] = json.load(f)

# Starter questions - the 3 top CRM questions (one per entity type)
_STARTERS: list[str] = [
    "What deals are in the pipeline?",
    "Which accounts are at risk?",
    "Which contacts haven't been contacted recently?",
]

# Build directed graph
_G = nx.DiGraph()

for question, follow_ups in _raw_data.items():
    _G.add_node(question)
    for follow_up in follow_ups:
        _G.add_edge(question, follow_up)


# =============================================================================
# Public API
# =============================================================================


def get_starters() -> list[str]:
    """Get the starter questions."""
    return _STARTERS.copy()


def get_follow_ups(question: str) -> list[str]:
    """
    Get follow-up questions for a given question.

    Returns hardcoded follow-ups from the tree, or empty list
    if the question isn't in the tree.
    """
    if question in _G:
        return list(_G.successors(question))
    return []


def get_graph() -> nx.DiGraph:
    """Get the question tree graph (read-only copy)."""
    return _G.copy()
