# backend.api - API Routes
"""
API route modules for the CRM AI Companion.

Modules:
- chat: Main chat endpoint for CRM questions
- health: Health check and system info endpoints
- data: Data explorer endpoints for CRM tables
"""

from fastapi import APIRouter

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.data import router as data_router

# Combined router with /api prefix
router = APIRouter(prefix="/api")
router.include_router(chat_router, tags=["chat"])
router.include_router(health_router, tags=["health"])
router.include_router(data_router, tags=["data"])

__all__ = ["router", "chat_router", "health_router", "data_router"]
