# =============================================================================
# Acme CRM AI Companion - Backend API
# =============================================================================
"""
FastAPI backend for the CRM AI Companion.

Run the app:
    uvicorn backend.main:app --reload
    # or
    python -m backend.main

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Example curl:
    curl -X POST http://localhost:8000/api/chat \
        -H "Content-Type: application/json" \
        -d '{"question": "What is going on with Acme Manufacturing?"}'
"""

import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

# =============================================================================
# Environment Setup
# =============================================================================

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# =============================================================================
# Local Imports
# =============================================================================

from backend.config import get_settings
from backend.api import router
from backend.middleware import RequestLoggingMiddleware, CacheControlMiddleware, RateLimitMiddleware
from backend.exceptions import APIError, ErrorResponse

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# Middleware
# =============================================================================

class RateLimitStore:
    """Simple in-memory rate limit store."""

    def __init__(self) -> None:
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_rate_limited(self, client_id: str, max_requests: int, window_seconds: int) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - window_seconds
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        if len(self.requests[client_id]) >= max_requests:
            return True
        self.requests[client_id].append(now)
        return False

    def get_remaining(self, client_id: str, max_requests: int, window_seconds: int) -> int:
        """Get remaining requests for client."""
        now = time.time()
        window_start = now - window_seconds
        recent_requests = len([
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ])
        return max(0, max_requests - recent_requests)


_rate_limit_store = RateLimitStore()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return await call_next(request)
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        if request.url.path in ["/api/health", "/api/info"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        if _rate_limit_store.is_rate_limited(
            client_ip, settings.rate_limit_requests, settings.rate_limit_window
        ):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": True, "message": "Rate limit exceeded."},
                headers={
                    "Retry-After": str(settings.rate_limit_window),
                    "X-RateLimit-Limit": str(settings.rate_limit_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        remaining = _rate_limit_store.get_remaining(
            client_ip, settings.rate_limit_requests, settings.rate_limit_window
        )
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests with timing and request ID."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start_time = time.time()

        if settings.log_requests:
            logger.info(f"[{request_id}] {request.method} {request.url.path} - Started")

        try:
            response = await call_next(request)
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[{request_id}] Error after {elapsed_ms}ms: {e}", exc_info=True)
            raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

        if settings.log_requests:
            log_fn = logger.info if response.status_code < 400 else logger.warning
            log_fn(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({elapsed_ms}ms)")

        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Add cache control headers to prevent caching of API responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


# =============================================================================
# RAG Collection Setup
# =============================================================================

def ensure_rag_collections_exist() -> None:
    """
    Ensure RAG Qdrant collections exist, create if missing.

    This runs at startup to auto-ingest data if collections don't exist.
    """
    from qdrant_client import QdrantClient
    from backend.agent.rag_tools import (
        QDRANT_PATH, DOCS_COLLECTION, PRIVATE_COLLECTION,
        ingest_docs, ingest_private_texts,
    )

    qdrant = QdrantClient(path=str(QDRANT_PATH))

    try:
        # Check docs collection
        if not qdrant.collection_exists(DOCS_COLLECTION):
            logger.info(f"Collection '{DOCS_COLLECTION}' not found, ingesting docs...")
            qdrant.close()
            ingest_docs()
            qdrant = QdrantClient(path=str(QDRANT_PATH))
        else:
            info = qdrant.get_collection(DOCS_COLLECTION)
            if info.points_count == 0:
                logger.info(f"Collection '{DOCS_COLLECTION}' is empty, ingesting docs...")
                qdrant.close()
                ingest_docs()
                qdrant = QdrantClient(path=str(QDRANT_PATH))
            else:
                logger.info(f"Docs collection ready with {info.points_count} points")

        # Check private collection
        if not qdrant.collection_exists(PRIVATE_COLLECTION):
            logger.info(f"Collection '{PRIVATE_COLLECTION}' not found, ingesting private texts...")
            qdrant.close()
            ingest_private_texts()
            qdrant = QdrantClient(path=str(QDRANT_PATH))
        else:
            info = qdrant.get_collection(PRIVATE_COLLECTION)
            if info.points_count == 0:
                logger.info(f"Collection '{PRIVATE_COLLECTION}' is empty, ingesting private texts...")
                qdrant.close()
                ingest_private_texts()
                qdrant = QdrantClient(path=str(QDRANT_PATH))
            else:
                logger.info(f"Private collection ready with {info.points_count} points")

        qdrant.close()
    except Exception as e:
        qdrant.close()
        logger.error(f"Failed to ensure RAG collections: {e}")
        raise


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    settings = get_settings()

    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")

    # Ensure RAG collections exist
    ensure_rag_collections_exist()

    yield

    # Shutdown
    logger.info("Shutting down...")


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI app.

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    # Create app
    app = FastAPI(
        title=settings.app_name,
        description="""
## Acme CRM AI Companion API

Talk to your CRM data using natural language.

### Features
- **Natural Language Queries** - Ask questions about accounts, activities, pipeline
- **Smart Routing** - Automatically determines if you need CRM data, docs, or both
- **Grounded Answers** - Responses cite specific data sources
- **Follow-up Suggestions** - Get intelligent suggestions for next questions

### Example Questions
- "What's going on with Acme Manufacturing in the last 90 days?"
- "Which renewals are coming up this month?"
- "How do I import contacts into the CRM?"
        """,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ==========================================================================
    # Middleware (order matters - first added = last executed)
    # ==========================================================================

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    # Custom middleware (order matters - first added = last executed)
    app.add_middleware(CacheControlMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ==========================================================================
    # Exception Handlers
    # ==========================================================================

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        """Handle custom API errors."""
        request_id = getattr(request.state, "request_id", None)

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status_code=exc.status_code,
                message=exc.detail,
                details=exc.details,
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, "request_id", None)

        logger.error(
            f"Unhandled exception: {exc}",
            extra={"request_id": request_id},
            exc_info=True,
        )

        # Don't expose internal errors in production
        message = str(exc) if settings.debug else "Internal server error"

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status_code=500,
                message=message,
                request_id=request_id,
            ).model_dump(),
        )

    # ==========================================================================
    # Routes
    # ==========================================================================

    app.include_router(router)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Redirect root to API docs."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    return app


# =============================================================================
# App Instance
# =============================================================================

app = create_app()


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
