"""
Tests for backend/agent/route/node.py.

Tests the routing node for query planning.
"""

import os

import pytest
from unittest.mock import patch

os.environ["MOCK_LLM"] = "1"


class TestRouteNode:
    """Tests for route_node function."""

    @patch('backend.agent.route.node.get_slot_plan')
    def test_route_node_returns_slot_plan(self, mock_slot_planner):
        """Returns slot plan in state."""
        from backend.agent.route.node import route_node
        from backend.agent.route.slot_query import SlotPlan, SlotQuery

        mock_slot_plan = SlotPlan(
            queries=[SlotQuery(table="companies", filters=[])],
            needs_rag=True
        )
        mock_slot_planner.return_value = mock_slot_plan

        state = {
            "question": "What's happening with Acme?",
            "messages": [],
        }

        result = route_node(state)

        assert result["slot_plan"] is not None
        assert len(result["slot_plan"].queries) == 1
        assert result["needs_rag"] is True

    @patch('backend.agent.route.node.get_slot_plan')
    def test_route_node_passes_conversation_history(self, mock_slot_planner):
        """Passes formatted conversation history to planner."""
        from backend.agent.route.node import route_node
        from backend.agent.route.slot_query import SlotPlan, SlotQuery

        mock_slot_plan = SlotPlan(
            queries=[SlotQuery(table="opportunities", filters=[])],
            needs_rag=False
        )
        mock_slot_planner.return_value = mock_slot_plan

        state = {
            "question": "What about their pipeline?",
            "messages": [
                {"role": "user", "content": "Tell me about Acme"},
                {"role": "assistant", "content": "Acme is a company..."},
            ],
        }

        route_node(state)

        call_kwargs = mock_slot_planner.call_args[1]
        assert "conversation_history" in call_kwargs
        assert len(call_kwargs["conversation_history"]) > 0

    @patch('backend.agent.route.node.get_slot_plan')
    def test_route_node_handles_exception_with_fallback(self, mock_slot_planner):
        """Handles exception and uses fallback slot plan."""
        from backend.agent.route.node import route_node

        mock_slot_planner.side_effect = Exception("Planner error")

        state = {
            "question": "What's happening with our company accounts?",
            "messages": [],
        }

        result = route_node(state)

        assert "slot_plan" in result
        assert result["needs_rag"] is False
        assert "error" in result

    @patch('backend.agent.route.node.get_slot_plan')
    def test_route_node_handles_empty_messages(self, mock_slot_planner):
        """Handles state with no messages."""
        from backend.agent.route.node import route_node
        from backend.agent.route.slot_query import SlotPlan, SlotQuery

        mock_slot_plan = SlotPlan(
            queries=[SlotQuery(table="companies", filters=[])],
            needs_rag=False
        )
        mock_slot_planner.return_value = mock_slot_plan

        state = {
            "question": "Test question",
            # No messages key
        }

        result = route_node(state)

        call_kwargs = mock_slot_planner.call_args[1]
        assert call_kwargs["conversation_history"] == ""
