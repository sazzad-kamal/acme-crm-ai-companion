"""Chat endpoint for CRM AI Companion."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.act_fetch import (
    AVAILABLE_DATABASES,
    DEMO_MODE,
    DEMO_STARTERS,
    get_database,
    set_database,
)
from backend.agent.followup.tree import get_starters
from backend.agent.streaming import stream_agent

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None

class DatabaseRequest(BaseModel):
    database: str = Field(..., description="Database name: KQC or W31322003119")

@router.get("/chat/starter-questions", summary="Get starter questions")
def get_starter_questions() -> list[str]:
    return DEMO_STARTERS if DEMO_MODE else get_starters()

@router.get("/chat/databases", summary="Get available databases")
def get_databases() -> dict:
    """Get available databases and current selection."""
    return {
        "databases": AVAILABLE_DATABASES,
        "current": get_database(),
    }

@router.post("/chat/database", summary="Switch database")
def switch_database(payload: DatabaseRequest) -> dict:
    """Switch to a different database."""
    set_database(payload.database)
    return {"database": get_database()}

@router.post("/chat/stream", summary="Stream a chat response (SSE)")
def chat_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_agent(payload.question, session_id=payload.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
