"""
Tests for backend/agent/route/node.py.

Tests the routing node for query planning.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ["MOCK_LLM"] = "1"


# =============================================================================
# Route Node Tests
# =============================================================================


class TestRouteNode:
    """Tests for route_node function."""

    @patch('backend.agent.route.node.get_query_plan')
    def test_route_node_returns_query_plan(self, mock_planner):
        """Returns query plan in state."""
        from backend.agent.route.node import route_node
        from backend.agent.route.query_planner import QueryPlan, SQLQuery

        mock_plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM companies LIMIT 1", purpose="company_info")],
            needs_account_rag=True
        )
        mock_planner.return_value = mock_plan

        state = {
            "question": "What's happening with Acme?",
            "messages": [],
        }

        result = route_node(state)

        assert result["query_plan"] is mock_plan
        assert result["needs_account_rag"] is True
        assert result["days"] == 90  # From config.default_days

    @patch('backend.agent.route.node.get_query_plan')
    def test_route_node_passes_conversation_history(self, mock_planner):
        """Passes formatted conversation history to planner."""
        from backend.agent.route.node import route_node
        from backend.agent.route.query_planner import QueryPlan, SQLQuery

        mock_plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM opportunities", purpose="pipeline")],
            needs_account_rag=False
        )
        mock_planner.return_value = mock_plan

        state = {
            "question": "What about their pipeline?",
            "messages": [
                {"role": "user", "content": "Tell me about Acme"},
                {"role": "assistant", "content": "Acme is a company..."},
            ],
        }

        route_node(state)

        call_kwargs = mock_planner.call_args[1]
        assert "conversation_history" in call_kwargs
        assert len(call_kwargs["conversation_history"]) > 0

    @patch('backend.agent.route.node.get_query_plan')
    def test_route_node_records_latency(self, mock_planner):
        """Records routing latency."""
        from backend.agent.route.node import route_node
        from backend.agent.route.query_planner import QueryPlan, SQLQuery

        mock_plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM companies", purpose="companies")],
            needs_account_rag=False
        )
        mock_planner.return_value = mock_plan

        state = {
            "question": "Test question",
            "messages": [],
        }

        result = route_node(state)

        assert "router_latency_ms" in result
        assert result["router_latency_ms"] >= 0

    @patch('backend.agent.route.node.get_query_plan')
    def test_route_node_returns_steps(self, mock_planner):
        """Returns steps for progress tracking."""
        from backend.agent.route.node import route_node
        from backend.agent.route.query_planner import QueryPlan, SQLQuery

        mock_plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM companies", purpose="companies")],
            needs_account_rag=False
        )
        mock_planner.return_value = mock_plan

        state = {
            "question": "Test",
            "messages": [],
        }

        result = route_node(state)

        assert "steps" in result
        assert len(result["steps"]) > 0
        assert result["steps"][0]["id"] == "router"
        assert result["steps"][0]["status"] == "done"

    @patch('backend.agent.route.node.get_query_plan')
    def test_route_node_handles_exception_with_fallback(self, mock_planner):
        """Handles exception and uses fallback query plan."""
        from backend.agent.route.node import route_node

        mock_planner.side_effect = Exception("Planner error")

        state = {
            "question": "What's happening with our company accounts?",
            "messages": [],
        }

        result = route_node(state)

        # Should fallback with default query plan
        assert "query_plan" in result
        assert result["needs_account_rag"] is False
        assert "error" in result
        assert result["steps"][0]["status"] == "error"

    @patch('backend.agent.route.node.get_query_plan')
    def test_route_node_handles_empty_messages(self, mock_planner):
        """Handles state with no messages."""
        from backend.agent.route.node import route_node
        from backend.agent.route.query_planner import QueryPlan, SQLQuery

        mock_plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM companies", purpose="companies")],
            needs_account_rag=False
        )
        mock_planner.return_value = mock_plan

        state = {
            "question": "Test question",
            # No messages key
        }

        result = route_node(state)

        # Should work without messages
        call_kwargs = mock_planner.call_args[1]
        assert call_kwargs["conversation_history"] == ""
