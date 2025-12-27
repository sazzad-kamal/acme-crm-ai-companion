"""
Performance utilities for the backend.

Provides:
- Timeout configuration for external calls
- Context manager helpers for resource cleanup
- Async parallel execution utilities
- Simple performance metrics

Usage:
    from backend.common.performance import (
        get_timeout_config,
        run_parallel,
        timed,
    )
"""

import asyncio
import time
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Timeout Configuration
# =============================================================================

@dataclass(frozen=True)
class TimeoutConfig:
    """
    Timeout configuration for external service calls.

    All values in seconds.
    """
    # LLM/OpenAI calls
    llm_timeout: float = 60.0
    llm_connect_timeout: float = 10.0

    # Embedding model inference
    embedding_timeout: float = 30.0

    # Reranker inference
    reranker_timeout: float = 30.0

    # Database queries
    db_timeout: float = 10.0

    # Vector store (Qdrant) operations
    vector_store_timeout: float = 15.0

    # HTTP client defaults
    http_timeout: float = 30.0
    http_connect_timeout: float = 5.0


# Singleton config instance
_timeout_config: TimeoutConfig | None = None


def get_timeout_config() -> TimeoutConfig:
    """
    Get the timeout configuration.

    Values can be overridden via environment variables:
    - PERF_LLM_TIMEOUT
    - PERF_EMBEDDING_TIMEOUT
    - PERF_DB_TIMEOUT
    - etc.
    """
    global _timeout_config
    if _timeout_config is None:
        _timeout_config = TimeoutConfig(
            llm_timeout=float(os.environ.get("PERF_LLM_TIMEOUT", "60.0")),
            llm_connect_timeout=float(os.environ.get("PERF_LLM_CONNECT_TIMEOUT", "10.0")),
            embedding_timeout=float(os.environ.get("PERF_EMBEDDING_TIMEOUT", "30.0")),
            reranker_timeout=float(os.environ.get("PERF_RERANKER_TIMEOUT", "30.0")),
            db_timeout=float(os.environ.get("PERF_DB_TIMEOUT", "10.0")),
            vector_store_timeout=float(os.environ.get("PERF_VECTOR_STORE_TIMEOUT", "15.0")),
            http_timeout=float(os.environ.get("PERF_HTTP_TIMEOUT", "30.0")),
            http_connect_timeout=float(os.environ.get("PERF_HTTP_CONNECT_TIMEOUT", "5.0")),
        )
    return _timeout_config


def reset_timeout_config() -> None:
    """Reset timeout config (useful for testing)."""
    global _timeout_config
    _timeout_config = None


# =============================================================================
# Timing Utilities
# =============================================================================

@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    elapsed_ms: float
    success: bool
    error: str | None = None


@contextmanager
def timed(name: str, log_level: int = logging.DEBUG) -> "Generator[TimingResult, None, None]":
    """
    Context manager for timing code blocks.

    Usage:
        with timed("embedding_generation"):
            embeddings = model.encode(texts)
    """
    start = time.perf_counter()
    result = TimingResult(name=name, elapsed_ms=0, success=True)
    try:
        yield result
    except Exception as e:
        result.success = False
        result.error = str(e)
        raise
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        logger.log(
            log_level,
            f"[{name}] {'OK' if result.success else 'FAILED'} in {result.elapsed_ms:.1f}ms"
        )


def timed_decorator(name: str | None = None) -> Callable[[Callable], Callable]:
    """
    Decorator for timing function execution.

    Usage:
        @timed_decorator("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        operation_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            with timed(operation_name):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: object, **kwargs: object) -> object:
            with timed(operation_name):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# Async Parallel Execution
# =============================================================================

T = TypeVar("T")


async def run_parallel(*coros) -> tuple:
    """
    Run multiple coroutines in parallel and return results.

    Usage:
        data, docs = await run_parallel(
            fetch_data(company_id),
            fetch_docs(query),
        )
    """
    return await asyncio.gather(*coros, return_exceptions=False)


async def run_parallel_safe(*coros) -> list[tuple[bool, Any]]:
    """
    Run multiple coroutines in parallel, returning (success, result) tuples.

    Exceptions are caught and returned as (False, exception).

    Usage:
        results = await run_parallel_safe(
            fetch_data(company_id),
            fetch_docs(query),
        )
        for success, value in results:
            if success:
                process(value)
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [
        (False, r) if isinstance(r, Exception) else (True, r)
        for r in results
    ]


def run_sync_parallel(*funcs: Callable[[], T]) -> list[T]:
    """
    Run multiple synchronous functions in parallel using threads.

    For CPU-bound work, consider using ProcessPoolExecutor instead.

    Usage:
        data, docs = run_sync_parallel(
            lambda: fetch_data(company_id),
            lambda: fetch_docs(query),
        )
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(funcs)) as executor:
        futures = [executor.submit(func) for func in funcs]
        return [f.result() for f in futures]


# =============================================================================
# Resource Management
# =============================================================================

class ResourceManager:
    """
    Simple resource manager for tracking and cleaning up resources.

    Usage:
        manager = ResourceManager()
        manager.register("db", connection, cleanup=connection.close)
        ...
        manager.cleanup_all()
    """

    def __init__(self):
        self._resources: dict[str, tuple[Any, Callable | None]] = {}

    def register(
        self,
        name: str,
        resource: Any,
        cleanup: Callable | None = None,
    ) -> None:
        """Register a resource with optional cleanup function."""
        self._resources[name] = (resource, cleanup)

    def get(self, name: str) -> Any | None:
        """Get a registered resource by name."""
        if name in self._resources:
            return self._resources[name][0]
        return None

    def cleanup(self, name: str) -> None:
        """Clean up a specific resource."""
        if name in self._resources:
            resource, cleanup_fn = self._resources.pop(name)
            if cleanup_fn:
                try:
                    cleanup_fn()
                except Exception as e:
                    logger.warning(f"Error cleaning up resource '{name}': {e}")

    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        for name in list(self._resources.keys()):
            self.cleanup(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
        return False


# =============================================================================
# Simple Metrics
# =============================================================================

@dataclass
class OperationMetrics:
    """Simple metrics for tracking operation performance."""
    total_calls: int = 0
    total_time_ms: float = 0.0
    errors: int = 0
    _timings: list[float] = field(default_factory=list)

    def record(self, elapsed_ms: float, success: bool = True) -> None:
        """Record an operation's timing."""
        self.total_calls += 1
        self.total_time_ms += elapsed_ms
        self._timings.append(elapsed_ms)
        if not success:
            self.errors += 1
        # Keep only last 100 timings for percentile calculation
        if len(self._timings) > 100:
            self._timings = self._timings[-100:]

    @property
    def avg_time_ms(self) -> float:
        """Average time in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_time_ms / self.total_calls

    @property
    def p95_time_ms(self) -> float:
        """95th percentile time in milliseconds."""
        if not self._timings:
            return 0.0
        sorted_timings = sorted(self._timings)
        idx = int(len(sorted_timings) * 0.95)
        return sorted_timings[min(idx, len(sorted_timings) - 1)]

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction (0.0 to 1.0)."""
        if self.total_calls == 0:
            return 0.0
        return self.errors / self.total_calls


# Global metrics registry
_metrics: dict[str, OperationMetrics] = {}


def get_metrics(operation: str) -> OperationMetrics:
    """Get or create metrics for an operation."""
    if operation not in _metrics:
        _metrics[operation] = OperationMetrics()
    return _metrics[operation]


def get_all_metrics() -> dict[str, OperationMetrics]:
    """Get all registered metrics."""
    return _metrics.copy()


def clear_metrics() -> None:
    """Clear all metrics (useful for testing)."""
    _metrics.clear()


__all__ = [
    "TimeoutConfig",
    "get_timeout_config",
    "reset_timeout_config",
    "TimingResult",
    "timed",
    "timed_decorator",
    "run_parallel",
    "run_parallel_safe",
    "run_sync_parallel",
    "ResourceManager",
    "OperationMetrics",
    "get_metrics",
    "get_all_metrics",
    "clear_metrics",
]
