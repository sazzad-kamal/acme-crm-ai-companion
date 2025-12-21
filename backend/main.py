# =============================================================================
# Acme CRM AI Companion - Backend API
# =============================================================================
# Run the app:
#   python main.py
#
# Call the endpoint with curl:
#   curl -X POST http://localhost:8000/api/chat \
#     -H "Content-Type: application/json" \
#     -d '{"question": "What'\''s going on with Acme Manufacturing in the last 90 days?"}'
# =============================================================================

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file from project root
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import schemas from the centralized schemas module
from project2_agentic.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
    Step,
    RawData,
    MetaInfo,
)

# Import the agent
from project2_agentic.agent import answer_question

# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Acme CRM AI Companion API",
    description="Talk to your CRM data using natural language",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Endpoint Implementation
# -----------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint - answers questions using CRM data and/or docs.
    
    The agent will:
    1. Route the question to determine mode (docs, data, data+docs)
    2. Fetch relevant CRM data if needed
    3. Query documentation if needed  
    4. Generate a grounded answer using LLM
    """
    # Call the agent
    result = answer_question(
        question=payload.question,
        mode=payload.mode or "auto",
        company_id=payload.company_id,
        session_id=payload.session_id,
        user_id=payload.user_id,
    )
    
    # Build response matching the contract
    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        steps=[Step(**s) for s in result["steps"]],
        raw_data=RawData(**result["raw_data"]),
        meta=MetaInfo(**result["meta"]),
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


# -----------------------------------------------------------------------------
# Simple Test (can be called manually from REPL or converted to pytest later)
# -----------------------------------------------------------------------------


def test_chat_mock() -> None:
    """Test the /api/chat endpoint with a mock question."""
    import os
    os.environ["MOCK_LLM"] = "1"  # Use mock mode for testing
    
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"question": "What's going on with Acme Manufacturing in the last 90 days?"},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    assert "answer" in data, "Response should contain 'answer'"
    assert "sources" in data, "Response should contain 'sources'"
    assert "steps" in data, "Response should contain 'steps'"
    assert "raw_data" in data, "Response should contain 'raw_data'"
    assert "meta" in data, "Response should contain 'meta'"
    
    # Check that we got company data
    if data["raw_data"].get("companies"):
        assert (
            data["raw_data"]["companies"][0]["company_id"] == "ACME-MFG"
        ), "First company should be ACME-MFG"

    print("All tests passed!")


# -----------------------------------------------------------------------------
# Local Development Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
