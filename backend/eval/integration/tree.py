"""Evaluation utilities for the question tree."""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import networkx as nx
import yaml

from backend.agent.followup.tree import get_graph, get_starters

logger = logging.getLogger(__name__)

_EVAL_FIXTURES_PATH = Path(__file__).parent / "fixtures"


def _load_yaml_fixture(filename: str) -> dict:
    """Load a YAML fixture file."""
    filepath = _EVAL_FIXTURES_PATH / filename
    if not filepath.exists():
        return {}
    try:
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception as e:
        logger.warning(f"Failed to load {filename}: {e}")
        return {}


@cache
def _load_expected() -> dict[str, dict]:
    """Load expected fixtures (cached)."""
    return _load_yaml_fixture("expected.yaml")


def _find_paths(graph: nx.DiGraph, starters: list[str]) -> list[list[str]]:
    """Find all paths from starters to leaf nodes in the graph."""
    leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    paths: list[list[str]] = []
    for starter in starters:
        for leaf in leaves:
            try:
                for path in nx.all_simple_paths(graph, starter, leaf):
                    paths.append(path)
            except nx.NodeNotFound:
                continue
    return paths


@cache
def _compute_paths_and_stats() -> tuple[list[list[str]], dict]:
    """Compute all paths and tree stats in a single pass (cached)."""
    graph = get_graph()
    starters = get_starters()

    # Build reachable subgraph
    reachable: set[str] = set()
    for starter in starters:
        if starter in graph:
            reachable.add(starter)
            reachable |= nx.descendants(graph, starter)
    subgraph: nx.DiGraph = graph.subgraph(reachable).copy()  # type: ignore[assignment]

    reachable_starters = [s for s in starters if s in subgraph]
    paths = _find_paths(subgraph, reachable_starters)

    stats = {
        "num_starters": len(starters),
        "num_questions": subgraph.number_of_nodes(),
        "num_edges": subgraph.number_of_edges(),
        "num_paths": len(paths),
        "path_lengths": {
            "min": min(len(p) for p in paths) if paths else 0,
            "max": max(len(p) for p in paths) if paths else 0,
        },
    }
    return paths, stats


def get_expected_answer(question: str) -> str | None:
    """Get the expected answer for a question (for RAGAS answer_correctness)."""
    entry = _load_expected().get(question)
    if entry is None:
        return None
    return entry.get("answer")


def get_expected_action(question: str) -> bool | None:
    """Get whether an action is expected for a question. None if not in fixture."""
    entry = _load_expected().get(question)
    if entry is None:
        return None
    return entry.get("action")


def get_all_paths() -> list[list[str]]:
    """Get all conversation paths from starters to leaf nodes."""
    paths, _ = _compute_paths_and_stats()
    return paths


def get_tree_stats() -> dict:
    """Get statistics about the question tree."""
    _, stats = _compute_paths_and_stats()
    return stats


__all__ = [
    "get_all_paths",
    "get_expected_action",
    "get_expected_answer",
    "get_tree_stats",
]
