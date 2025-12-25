"""
Tests for backend/api modules (chat, health, data).

Tests the individual API route handlers after refactoring.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set mock mode before imports
os.environ["MOCK_LLM"] = "1"

from fastapi.testclient import TestClient
from backend.main import app

# client fixture is provided by conftest.py


# =============================================================================
# Health Module Tests
# =============================================================================

class TestHealthModule:
    """Tests for backend/api/health.py routes."""
    
    def test_health_returns_status_ok(self, client):
        """Health endpoint returns OK status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_health_includes_version(self, client):
        """Health endpoint includes version."""
        response = client.get("/api/health")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)
    
    def test_health_includes_services(self, client):
        """Health endpoint includes services status."""
        response = client.get("/api/health")
        data = response.json()
        assert "services" in data
        assert "data" in data["services"]
    
    def test_info_returns_app_info(self, client):
        """Info endpoint returns app information."""
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data
    
    def test_info_includes_cors_origins(self, client):
        """Info endpoint includes CORS origins."""
        response = client.get("/api/info")
        data = response.json()
        assert "cors_origins" in data
        assert isinstance(data["cors_origins"], list)


# =============================================================================
# Data Module Tests
# =============================================================================

class TestDataModule:
    """Tests for backend/api/data.py routes."""
    
    def test_companies_endpoint_exists(self, client):
        """Companies endpoint is accessible."""
        response = client.get("/api/data/companies")
        assert response.status_code == 200
    
    def test_companies_returns_data_structure(self, client):
        """Companies returns proper data structure."""
        response = client.get("/api/data/companies")
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert "columns" in data
    
    def test_contacts_endpoint_exists(self, client):
        """Contacts endpoint is accessible."""
        response = client.get("/api/data/contacts")
        assert response.status_code == 200
    
    def test_contacts_returns_data_structure(self, client):
        """Contacts returns proper data structure."""
        response = client.get("/api/data/contacts")
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert "columns" in data
    
    def test_opportunities_endpoint_exists(self, client):
        """Opportunities endpoint is accessible."""
        response = client.get("/api/data/opportunities")
        assert response.status_code == 200
    
    def test_activities_endpoint_exists(self, client):
        """Activities endpoint is accessible."""
        response = client.get("/api/data/activities")
        assert response.status_code == 200
    
    def test_groups_endpoint_exists(self, client):
        """Groups endpoint is accessible."""
        response = client.get("/api/data/groups")
        assert response.status_code == 200
    
    def test_history_endpoint_exists(self, client):
        """History endpoint is accessible."""
        response = client.get("/api/data/history")
        assert response.status_code == 200
    
    def test_all_endpoints_return_lists(self, client):
        """All data endpoints return lists."""
        endpoints = [
            "/api/data/companies",
            "/api/data/contacts",
            "/api/data/opportunities",
            "/api/data/activities",
            "/api/data/groups",
            "/api/data/history",
        ]
        for endpoint in endpoints:
            response = client.get(endpoint)
            data = response.json()
            assert isinstance(data["data"], list), f"{endpoint} should return list"


# =============================================================================
# Chat Module Tests
# =============================================================================

class TestChatModule:
    """Tests for backend/api/chat.py routes."""
    
    def test_chat_endpoint_accepts_post(self, client):
        """Chat endpoint accepts POST requests."""
        response = client.post("/api/chat", json={"question": "test"})
        assert response.status_code == 200
    
    def test_chat_returns_answer(self, client):
        """Chat endpoint returns an answer."""
        response = client.post("/api/chat", json={"question": "What is Acme?"})
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
    
    def test_chat_returns_sources(self, client):
        """Chat endpoint returns sources list."""
        response = client.post("/api/chat", json={"question": "What is Acme?"})
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)
    
    def test_chat_returns_steps(self, client):
        """Chat endpoint returns processing steps."""
        response = client.post("/api/chat", json={"question": "What is Acme?"})
        data = response.json()
        assert "steps" in data
        assert isinstance(data["steps"], list)
    
    def test_chat_returns_meta(self, client):
        """Chat endpoint returns metadata."""
        response = client.post("/api/chat", json={"question": "What is Acme?"})
        data = response.json()
        assert "meta" in data
        assert "latency_ms" in data["meta"]
    
    def test_chat_rejects_empty_question(self, client):
        """Chat rejects empty question."""
        response = client.post("/api/chat", json={"question": ""})
        assert response.status_code == 400
    
    def test_chat_rejects_whitespace_question(self, client):
        """Chat rejects whitespace-only question."""
        response = client.post("/api/chat", json={"question": "   "})
        assert response.status_code == 400
    
    def test_chat_accepts_mode_parameter(self, client):
        """Chat accepts mode parameter."""
        response = client.post(
            "/api/chat", 
            json={"question": "Help me", "mode": "docs"}
        )
        assert response.status_code == 200
    
    def test_chat_accepts_days_parameter(self, client):
        """Chat accepts days parameter."""
        response = client.post(
            "/api/chat", 
            json={"question": "Recent activity", "days": 7}
        )
        assert response.status_code == 200
    
    def test_chat_returns_follow_ups(self, client):
        """Chat returns follow-up suggestions."""
        response = client.post("/api/chat", json={"question": "Tell me about Acme"})
        data = response.json()
        assert "follow_up_suggestions" in data
        assert isinstance(data["follow_up_suggestions"], list)


# =============================================================================
# Route Registration Tests
# =============================================================================

class TestRouteRegistration:
    """Tests that all routes are properly registered."""
    
    def test_api_routes_are_prefixed(self, client):
        """All API routes have /api prefix."""
        # These should work
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/info").status_code == 200
        assert client.get("/api/data/companies").status_code == 200
    
    def test_unknown_routes_return_404(self, client):
        """Unknown routes return 404."""
        response = client.get("/api/unknown")
        assert response.status_code == 404
    
    def test_wrong_method_returns_405(self, client):
        """Wrong HTTP method returns 405."""
        response = client.get("/api/chat")  # Should be POST
        assert response.status_code == 405
