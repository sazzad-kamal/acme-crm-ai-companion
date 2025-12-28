"""
Tests for the LLM client module (LangChain-based).

Run with:
    pytest tests/backend/test_llm_client.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Mock OpenAI before importing the module
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"


# =============================================================================
# Import after setting env vars
# =============================================================================

from backend.common.llm_client import (
    clear_llm_cache,
    call_llm,
    call_llm_with_metrics,
    call_llm_safe,
    CostTracker,
    get_cost_tracker,
    reset_cost_tracker,
    MODEL_COSTS,
    _requires_max_completion_tokens,
)


# =============================================================================
# CostTracker Tests
# =============================================================================

class TestCostTracker:
    """Tests for CostTracker class."""

    def test_initial_state(self):
        """Test that CostTracker starts with zero values."""
        tracker = CostTracker()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.calls == 0

    def test_add_usage_updates_totals(self):
        """Test that add_usage updates running totals."""
        tracker = CostTracker()
        tracker.add_usage("gpt-4o-mini", 100, 50)

        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.calls == 1

    def test_add_usage_accumulates(self):
        """Test that multiple add_usage calls accumulate."""
        tracker = CostTracker()
        tracker.add_usage("gpt-4o-mini", 100, 50)
        tracker.add_usage("gpt-4o-mini", 200, 100)

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150
        assert tracker.calls == 2

    def test_add_usage_tracks_by_model(self):
        """Test that usage is tracked per model."""
        tracker = CostTracker()
        tracker.add_usage("gpt-4o-mini", 100, 50)
        tracker.add_usage("gpt-4o", 200, 100)

        assert "gpt-4o-mini" in tracker.costs_by_model
        assert "gpt-4o" in tracker.costs_by_model
        assert tracker.costs_by_model["gpt-4o-mini"]["input"] == 100
        assert tracker.costs_by_model["gpt-4o"]["input"] == 200

    def test_add_usage_returns_cost(self):
        """Test that add_usage returns the cost for the call."""
        tracker = CostTracker()
        cost = tracker.add_usage("gpt-4o-mini", 1_000_000, 0)

        # gpt-4o-mini input: $0.15 per 1M tokens
        assert cost == pytest.approx(0.15, abs=0.01)

    def test_get_summary_returns_dict(self):
        """Test that get_summary returns a proper dict."""
        tracker = CostTracker()
        tracker.add_usage("gpt-4o-mini", 100, 50)

        summary = tracker.get_summary()

        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "total_cost_usd" in summary
        assert "total_calls" in summary
        assert "by_model" in summary

    def test_unknown_model_uses_default_cost(self):
        """Test that unknown models use default pricing."""
        tracker = CostTracker()
        cost = tracker.add_usage("unknown-model", 1_000_000, 0)

        # Default input: $2.50 per 1M tokens
        assert cost == pytest.approx(2.50, abs=0.01)


# =============================================================================
# Global Cost Tracker Tests
# =============================================================================

class TestGlobalCostTracker:
    """Tests for global cost tracker functions."""

    def test_get_cost_tracker_returns_tracker(self):
        """Test that get_cost_tracker returns a CostTracker."""
        tracker = get_cost_tracker()
        assert isinstance(tracker, CostTracker)

    def test_reset_cost_tracker_clears_state(self):
        """Test that reset_cost_tracker creates fresh tracker."""
        tracker1 = get_cost_tracker()
        tracker1.add_usage("gpt-4o-mini", 100, 50)

        reset_cost_tracker()
        tracker2 = get_cost_tracker()

        assert tracker2.calls == 0
        assert tracker2.total_input_tokens == 0


# =============================================================================
# Model Configuration Tests
# =============================================================================

class TestModelConfiguration:
    """Tests for model configuration helpers."""

    def test_requires_max_completion_tokens_for_o1(self):
        """Test that o1 models require max_completion_tokens."""
        assert _requires_max_completion_tokens("o1-preview") is True
        assert _requires_max_completion_tokens("o1-mini") is True

    def test_requires_max_completion_tokens_for_o3(self):
        """Test that o3 models require max_completion_tokens."""
        assert _requires_max_completion_tokens("o3-mini") is True

    def test_gpt4_does_not_require_max_completion_tokens(self):
        """Test that GPT-4 models use regular max_tokens."""
        assert _requires_max_completion_tokens("gpt-4o") is False
        assert _requires_max_completion_tokens("gpt-4o-mini") is False
        assert _requires_max_completion_tokens("gpt-4-turbo") is False

    def test_case_insensitive(self):
        """Test that check is case-insensitive."""
        assert _requires_max_completion_tokens("O1-preview") is True
        assert _requires_max_completion_tokens("O1-MINI") is True


# =============================================================================
# Cache Operations Tests
# =============================================================================

class TestCacheOperations:
    """Tests for cache get/set operations."""

    def test_clear_cache_returns_count(self):
        """Test that clear_llm_cache returns the number of cleared entries."""
        count = clear_llm_cache()
        assert isinstance(count, int)
        assert count >= 0


# =============================================================================
# LLM Call Tests (Mocked)
# =============================================================================

class TestCallLLM:
    """Tests for call_llm function with mocked LangChain."""

    @patch("backend.common.llm_client._get_chat_model")
    def test_call_llm_returns_string(self, mock_get_model):
        """Test that call_llm returns a string response."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm("Test prompt")

        assert isinstance(result, str)
        assert result == "Test response"

    @patch("backend.common.llm_client._get_chat_model")
    def test_call_llm_with_system_prompt(self, mock_get_model):
        """Test that system prompt is included in messages."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        call_llm("User prompt", system_prompt="System prompt")

        # Verify invoke was called with messages
        call_args = mock_chat.invoke.call_args[0][0]
        assert len(call_args) == 2  # System + User
        assert call_args[0].content == "System prompt"
        assert call_args[1].content == "User prompt"

    @patch("backend.common.llm_client._get_chat_model")
    def test_call_llm_handles_empty_response(self, mock_get_model):
        """Test handling of empty response content."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = None
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm("Prompt")

        assert result == ""


# =============================================================================
# LLM Call with Metrics Tests
# =============================================================================

class TestCallLLMWithMetrics:
    """Tests for call_llm_with_metrics function."""

    @patch("backend.common.llm_client._get_chat_model")
    def test_returns_dict_with_response(self, mock_get_model):
        """Test that call_llm_with_metrics returns dict with response."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm_with_metrics("Test prompt")

        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] == "Test response"

    @patch("backend.common.llm_client._get_chat_model")
    def test_returns_latency_metric(self, mock_get_model):
        """Test that call_llm_with_metrics includes latency."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm_with_metrics("Prompt")

        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], (int, float))
        assert result["latency_ms"] >= 0

    @patch("backend.common.llm_client._get_chat_model")
    def test_returns_token_counts(self, mock_get_model):
        """Test that call_llm_with_metrics includes token counts."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm_with_metrics("Prompt")

        assert "prompt_tokens" in result
        assert "completion_tokens" in result
        assert "total_tokens" in result
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20
        assert result["total_tokens"] == 30

    @patch("backend.common.llm_client._get_chat_model")
    def test_returns_cost(self, mock_get_model):
        """Test that call_llm_with_metrics includes cost."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}}
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm_with_metrics("Prompt", model="gpt-4o-mini")

        assert "cost_usd" in result
        assert result["cost_usd"] >= 0


# =============================================================================
# Safe LLM Call Tests
# =============================================================================

class TestCallLLMSafe:
    """Tests for call_llm_safe function."""

    @patch("backend.common.llm_client._get_chat_model")
    def test_returns_response_on_success(self, mock_get_model):
        """Test that call_llm_safe returns response on success."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Success response"
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm_safe("Prompt")

        assert result == "Success response"

    @patch("backend.common.llm_client._get_chat_model")
    def test_returns_default_on_failure(self, mock_get_model):
        """Test that call_llm_safe returns default on failure."""
        mock_get_model.side_effect = Exception("API Error")

        result = call_llm_safe("Prompt", default="fallback")

        assert result == "fallback"

    @patch("backend.common.llm_client._get_chat_model")
    def test_returns_default_on_empty_response(self, mock_get_model):
        """Test that call_llm_safe returns default on empty response."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "   "  # Whitespace only
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm_safe("Prompt", default="fallback")

        assert result == "fallback"


# =============================================================================
# Configuration Tests
# =============================================================================

class TestLLMConfiguration:
    """Tests for LLM client configuration."""

    def test_default_model_is_set(self):
        """Test that a default model is used."""
        import inspect
        from backend.common.llm_client import call_llm

        sig = inspect.signature(call_llm)
        model_default = sig.parameters["model"].default

        assert model_default is not None
        assert isinstance(model_default, str)
        assert model_default == "gpt-4o-mini"

    def test_default_temperature_is_deterministic(self):
        """Test that default temperature is 0.0 for deterministic output."""
        import inspect
        from backend.common.llm_client import call_llm

        sig = inspect.signature(call_llm)
        temp_default = sig.parameters["temperature"].default

        assert temp_default == 0.0

    def test_cache_is_enabled_by_default(self):
        """Test that cache is enabled by default in call_llm."""
        import inspect
        from backend.common.llm_client import call_llm

        sig = inspect.signature(call_llm)
        cache_default = sig.parameters["use_cache"].default

        assert cache_default is True


# =============================================================================
# MODEL_COSTS Tests
# =============================================================================

class TestModelCosts:
    """Tests for model cost configuration."""

    def test_model_costs_has_required_keys(self):
        """Test that MODEL_COSTS has expected models."""
        assert "gpt-4o" in MODEL_COSTS
        assert "gpt-4o-mini" in MODEL_COSTS
        assert "default" in MODEL_COSTS

    def test_model_costs_have_input_output(self):
        """Test that each model has input and output costs."""
        for model, costs in MODEL_COSTS.items():
            assert "input" in costs, f"{model} missing 'input' cost"
            assert "output" in costs, f"{model} missing 'output' cost"

    def test_costs_are_positive(self):
        """Test that all costs are positive numbers."""
        for model, costs in MODEL_COSTS.items():
            assert costs["input"] >= 0, f"{model} has negative input cost"
            assert costs["output"] >= 0, f"{model} has negative output cost"


# =============================================================================
# Integration Tests
# =============================================================================

class TestLLMClientIntegration:
    """Integration tests for LLM client."""

    @patch("backend.common.llm_client._get_chat_model")
    def test_full_call_flow(self, mock_get_model):
        """Test the full call flow from prompt to response."""
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The answer is 42."
        mock_response.response_metadata = {
            "token_usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20}
        }
        mock_chat.invoke.return_value = mock_response
        mock_get_model.return_value = mock_chat

        result = call_llm_with_metrics(
            prompt="What is the meaning of life?",
            system_prompt="You are a helpful assistant.",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100,
        )

        assert result["response"] == "The answer is 42."
        assert result["total_tokens"] == 20
        assert result["latency_ms"] >= 0
        mock_chat.invoke.assert_called_once()
