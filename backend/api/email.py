"""Email API endpoints for contextual CRM follow-ups."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agent.email.generator import (
    CATEGORY_DESCRIPTIONS,
    _clear_cache,
    generate_email,
    get_cache_age,
    get_contacts_for_category,
    get_questions,
    warmup_cache,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["email"])


class GenerateEmailRequest(BaseModel):
    """Request body for email generation."""

    contactId: str
    category: str


class EmailContact(BaseModel):
    """Contact returned for a category."""

    contactId: str
    name: str
    company: str | None = None
    lastContact: str | None = None
    lastContactAgo: str | None = None
    reason: str


class ContactsResponse(BaseModel):
    """Response for contacts endpoint."""

    category: str
    contacts: list[EmailContact]
    cachedSecondsAgo: int | None = None  # How old the underlying history cache is


class GeneratedEmail(BaseModel):
    """Generated email response."""

    subject: str
    body: str
    mailtoLink: str
    contact: dict[str, Any]


@router.get("/questions")
async def get_email_questions() -> list[dict[str, str]]:
    """Return the 5 question categories."""
    questions: list[dict[str, str]] = get_questions()
    return questions


@router.get("/contacts")
async def get_email_contacts(category: str) -> ContactsResponse:
    """Return contacts for a category with AI-generated reasons.

    This endpoint:
    1. Fetches history from Act! API (cached 5 min)
    2. Uses LLM to classify contacts needing follow-up
    3. Returns contacts sorted by oldest first (most likely needs follow-up)
    """
    if category not in CATEGORY_DESCRIPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category: {category}. Valid: {list(CATEGORY_DESCRIPTIONS.keys())}",
        )

    try:
        contacts = await get_contacts_for_category(category)
        return ContactsResponse(
            category=category,
            contacts=[EmailContact(**c) for c in contacts],
            cachedSecondsAgo=get_cache_age(),
        )
    except Exception as e:
        logger.exception("Failed to get contacts for category %s", category)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/generate")
async def generate_email_endpoint(request: GenerateEmailRequest) -> GeneratedEmail:
    """Generate a follow-up email for a specific contact.

    This endpoint:
    1. Fetches contact by ID to get email address
    2. Uses cached history for context
    3. Uses LLM to generate personalized email
    4. Returns subject, body, and mailto: link
    """
    if request.category not in CATEGORY_DESCRIPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category: {request.category}",
        )

    try:
        result = await generate_email(request.contactId, request.category)
        return GeneratedEmail(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to generate email for contact %s", request.contactId)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/warmup")
async def warmup_history() -> dict[str, str]:
    """Prefetch history to warm the cache. Fire-and-forget from frontend."""
    await warmup_cache()
    return {"status": "ok"}


@router.post("/refresh")
async def refresh_history() -> dict[str, Any]:
    """Force refresh the history cache."""
    _clear_cache()
    count = await warmup_cache()
    return {"status": "ok", "records": count}
