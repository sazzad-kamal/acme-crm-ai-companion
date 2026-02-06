"""
Tests for backend.api.email module.

Tests the email generation and cache endpoints.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Set mock mode before imports
os.environ["MOCK_LLM"] = "1"

from backend.main import app


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Questions Endpoint Tests
# =============================================================================


class TestEmailQuestionsEndpoint:
    """Tests for GET /api/email/questions."""

    def test_returns_questions_list(self, client):
        """Test that endpoint returns list of questions."""
        response = client.get("/api/email/questions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 5

    def test_questions_have_required_fields(self, client):
        """Test that each question has id and label fields."""
        response = client.get("/api/email/questions")
        data = response.json()
        for q in data:
            assert "id" in q
            assert "label" in q
            assert isinstance(q["id"], str)
            assert isinstance(q["label"], str)


# =============================================================================
# Contacts Endpoint Tests
# =============================================================================


class TestEmailContactsEndpoint:
    """Tests for GET /api/email/contacts."""

    def test_rejects_unknown_category(self, client):
        """Test that endpoint rejects unknown categories."""
        response = client.get("/api/email/contacts?category=unknown_category")
        assert response.status_code == 400
        assert "Unknown category" in response.json()["detail"]

    def test_returns_contacts_response_structure(self, client):
        """Test that endpoint returns proper response structure."""
        with patch("backend.api.email.get_contacts_for_category", new_callable=AsyncMock) as mock_contacts:
            mock_contacts.return_value = [
                {
                    "contactId": "123",
                    "name": "John Doe",
                    "company": "Acme Inc",
                    "lastContact": "2026-01-15",
                    "lastContactAgo": "3 weeks ago",
                    "reason": "Follow up needed",
                }
            ]

            with patch("backend.api.email.get_cache_age") as mock_age:
                mock_age.return_value = 120

                response = client.get("/api/email/contacts?category=quotes")

                assert response.status_code == 200
                data = response.json()
                assert data["category"] == "quotes"
                assert isinstance(data["contacts"], list)
                assert data["cachedSecondsAgo"] == 120

    def test_includes_cache_age_in_response(self, client):
        """Test that response includes cachedSecondsAgo field."""
        with patch("backend.api.email.get_contacts_for_category", new_callable=AsyncMock) as mock_contacts:
            mock_contacts.return_value = []

            with patch("backend.api.email.get_cache_age") as mock_age:
                mock_age.return_value = 60

                response = client.get("/api/email/contacts?category=quotes")

                assert response.status_code == 200
                assert response.json()["cachedSecondsAgo"] == 60


# =============================================================================
# Generate Email Endpoint Tests
# =============================================================================


class TestEmailGenerateEndpoint:
    """Tests for POST /api/email/generate."""

    def test_rejects_unknown_category(self, client):
        """Test that endpoint rejects unknown categories."""
        response = client.post(
            "/api/email/generate",
            json={"contactId": "123", "category": "unknown_category"},
        )
        assert response.status_code == 400
        assert "Unknown category" in response.json()["detail"]

    def test_returns_generated_email_structure(self, client):
        """Test that endpoint returns proper email structure."""
        with patch("backend.api.email.generate_email", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "subject": "Following up on our quote",
                "body": "Hi John,\n\nI wanted to follow up...",
                "mailtoLink": "mailto:john@acme.com?subject=...",
                "contact": {"id": "123", "name": "John Doe"},
            }

            response = client.post(
                "/api/email/generate",
                json={"contactId": "123", "category": "quotes"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "subject" in data
            assert "body" in data
            assert "mailtoLink" in data
            assert "contact" in data


# =============================================================================
# Warmup Endpoint Tests
# =============================================================================


class TestEmailWarmupEndpoint:
    """Tests for POST /api/email/warmup."""

    def test_warmup_returns_ok_status(self, client):
        """Test that warmup endpoint returns ok status."""
        with patch("backend.api.email.warmup_cache", new_callable=AsyncMock) as mock_warmup:
            mock_warmup.return_value = 100

            response = client.post("/api/email/warmup")

            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_warmup_calls_warmup_cache(self, client):
        """Test that warmup endpoint calls warmup_cache function."""
        with patch("backend.api.email.warmup_cache", new_callable=AsyncMock) as mock_warmup:
            mock_warmup.return_value = 50

            client.post("/api/email/warmup")

            mock_warmup.assert_called_once()


# =============================================================================
# Refresh Endpoint Tests
# =============================================================================


class TestEmailRefreshEndpoint:
    """Tests for POST /api/email/refresh."""

    def test_refresh_returns_ok_status_with_records(self, client):
        """Test that refresh endpoint returns ok status with record count."""
        with patch("backend.api.email._clear_cache") as mock_clear:
            with patch("backend.api.email.warmup_cache", new_callable=AsyncMock) as mock_warmup:
                mock_warmup.return_value = 150

                response = client.post("/api/email/refresh")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"
                assert data["records"] == 150

    def test_refresh_clears_cache_before_warming(self, client):
        """Test that refresh endpoint clears cache before warming."""
        call_order = []

        with patch("backend.api.email._clear_cache") as mock_clear:
            mock_clear.side_effect = lambda: call_order.append("clear")

            with patch("backend.api.email.warmup_cache", new_callable=AsyncMock) as mock_warmup:
                async def track_warmup():
                    call_order.append("warmup")
                    return 100

                mock_warmup.side_effect = track_warmup

                client.post("/api/email/refresh")

                assert call_order == ["clear", "warmup"]
