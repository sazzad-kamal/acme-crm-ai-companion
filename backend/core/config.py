"""Centralized configuration for the backend API."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    app_name: str = "Acme CRM AI Companion API"
    app_version: str = "2.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # CORS
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # Logging
    log_level: str = "INFO"
    log_requests: bool = True

    # Paths
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        return Path(__file__).parent.parent / "data"

    model_config = {"env_prefix": "ACME_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
