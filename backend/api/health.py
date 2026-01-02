"""
Health check and system info endpoints.

Provides endpoints for monitoring and diagnostics.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.config import get_settings, Settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    services: dict[str, str] = Field(default_factory=dict, description="Service statuses")


class SystemInfo(BaseModel):
    """System information for diagnostics."""
    app_name: str
    version: str
    debug: bool
    cors_origins: list[str]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and dependent services are healthy.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Health check endpoint for monitoring and load balancers.

    Returns status of the API and its dependencies.
    """
    services = {
        "api": "healthy",
        "agent": "healthy",
    }

    # Check if CSV data is available
    try:
        csv_dir = settings.data_dir / "csv"
        if csv_dir.exists() and any(csv_dir.glob("*.csv")):
            services["data"] = "healthy"
        else:
            services["data"] = "missing"
    except Exception:
        services["data"] = "error"

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        services=services,
    )


@router.get(
    "/info",
    response_model=SystemInfo,
    summary="System information",
    description="Get information about the API configuration.",
)
async def system_info(
    settings: Settings = Depends(get_settings),
) -> SystemInfo:
    """
    Get system configuration information.

    Useful for debugging and verifying deployment configuration.
    """
    return SystemInfo(
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        cors_origins=settings.cors_origins_list,
    )
