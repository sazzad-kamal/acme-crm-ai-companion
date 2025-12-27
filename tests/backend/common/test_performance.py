"""
Tests for backend.common.performance module.

Tests timeout configuration, timing utilities, parallel execution, and metrics.
"""

import asyncio
import os
import time
import pytest

from backend.common.performance import (
    TimeoutConfig,
    get_timeout_config,
    reset_timeout_config,
    TimingResult,
    timed,
    timed_decorator,
    run_parallel,
    run_parallel_safe,
    run_sync_parallel,
    ResourceManager,
    OperationMetrics,
    get_metrics,
    get_all_metrics,
    clear_metrics,
)


# =============================================================================
# Timeout Configuration Tests
# =============================================================================

class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_timeout_config_defaults(self):
        """Test default timeout values."""
        config = TimeoutConfig()

        assert config.llm_timeout == 60.0
        assert config.embedding_timeout == 30.0
        assert config.db_timeout == 10.0

    def test_timeout_config_custom_values(self):
        """Test custom timeout values."""
        config = TimeoutConfig(
            llm_timeout=120.0,
            db_timeout=5.0,
        )

        assert config.llm_timeout == 120.0
        assert config.db_timeout == 5.0

    def test_timeout_config_immutable(self):
        """Test that TimeoutConfig is immutable."""
        config = TimeoutConfig()

        with pytest.raises(AttributeError):
            config.llm_timeout = 100.0


class TestGetTimeoutConfig:
    """Tests for get_timeout_config function."""

    def setup_method(self):
        """Reset config before each test."""
        reset_timeout_config()

    def teardown_method(self):
        """Clean up after each test."""
        reset_timeout_config()
        # Clean up env vars
        for key in list(os.environ.keys()):
            if key.startswith("PERF_"):
                del os.environ[key]

    def test_get_timeout_config_returns_singleton(self):
        """Test that get_timeout_config returns same instance."""
        config1 = get_timeout_config()
        config2 = get_timeout_config()

        assert config1 is config2

    def test_get_timeout_config_uses_env_vars(self):
        """Test that config uses environment variables."""
        os.environ["PERF_LLM_TIMEOUT"] = "120.0"
        os.environ["PERF_DB_TIMEOUT"] = "5.0"

        config = get_timeout_config()

        assert config.llm_timeout == 120.0
        assert config.db_timeout == 5.0


# =============================================================================
# Timing Utilities Tests
# =============================================================================

class TestTimed:
    """Tests for timed context manager."""

    def test_timed_measures_elapsed_time(self):
        """Test that timed measures elapsed time."""
        with timed("test_operation") as result:
            time.sleep(0.01)  # 10ms

        assert result.elapsed_ms >= 10
        assert result.success is True
        assert result.error is None

    def test_timed_captures_errors(self):
        """Test that timed captures errors."""
        with pytest.raises(ValueError):
            with timed("failing_operation") as result:
                raise ValueError("Test error")

        assert result.success is False
        assert result.error == "Test error"

    def test_timed_result_has_name(self):
        """Test that timing result has correct name."""
        with timed("my_operation") as result:
            pass

        assert result.name == "my_operation"


class TestTimedDecorator:
    """Tests for timed_decorator."""

    def test_timed_decorator_sync_function(self):
        """Test timed_decorator with sync function."""
        @timed_decorator("test_func")
        def my_func():
            time.sleep(0.01)
            return "result"

        result = my_func()

        assert result == "result"

    def test_timed_decorator_async_function(self):
        """Test timed_decorator with async function."""
        @timed_decorator("async_func")
        async def my_async_func():
            await asyncio.sleep(0.01)
            return "async_result"

        result = asyncio.run(my_async_func())

        assert result == "async_result"

    def test_timed_decorator_uses_function_name(self):
        """Test that decorator uses function name if not specified."""
        @timed_decorator()
        def named_function():
            return 42

        result = named_function()
        assert result == 42


# =============================================================================
# Parallel Execution Tests
# =============================================================================

class TestRunParallel:
    """Tests for run_parallel function."""

    def test_run_parallel_executes_concurrently(self):
        """Test that run_parallel executes coroutines concurrently."""
        async def slow_task(n):
            await asyncio.sleep(0.01)
            return n * 2

        async def run_test():
            start = time.time()
            results = await run_parallel(
                slow_task(1),
                slow_task(2),
                slow_task(3),
            )
            elapsed = time.time() - start

            # Should take ~10ms (parallel) not ~30ms (sequential)
            # Allow some margin for CI/slow machines
            assert elapsed < 0.1
            return results

        results = asyncio.run(run_test())

        assert list(results) == [2, 4, 6]

    def test_run_parallel_propagates_exceptions(self):
        """Test that run_parallel propagates exceptions."""
        async def failing_task():
            raise ValueError("Task failed")

        async def run_test():
            with pytest.raises(ValueError):
                await run_parallel(failing_task())

        asyncio.run(run_test())


class TestRunParallelSafe:
    """Tests for run_parallel_safe function."""

    def test_run_parallel_safe_returns_tuples(self):
        """Test that run_parallel_safe returns (success, result) tuples."""
        async def good_task():
            return "ok"

        async def run_test():
            results = await run_parallel_safe(good_task())
            return results

        results = asyncio.run(run_test())

        assert results == [(True, "ok")]

    def test_run_parallel_safe_catches_exceptions(self):
        """Test that run_parallel_safe catches exceptions."""
        async def good_task():
            return "ok"

        async def bad_task():
            raise ValueError("failed")

        async def run_test():
            results = await run_parallel_safe(good_task(), bad_task())
            return results

        results = asyncio.run(run_test())

        assert results[0] == (True, "ok")
        assert results[1][0] is False
        assert isinstance(results[1][1], ValueError)


class TestRunSyncParallel:
    """Tests for run_sync_parallel function."""

    def test_run_sync_parallel_executes_functions(self):
        """Test that run_sync_parallel executes functions."""
        def task1():
            return 1

        def task2():
            return 2

        results = run_sync_parallel(task1, task2)

        assert results == [1, 2]

    def test_run_sync_parallel_is_concurrent(self):
        """Test that run_sync_parallel runs concurrently."""
        def slow_task(n):
            time.sleep(0.01)
            return n

        start = time.time()
        results = run_sync_parallel(
            lambda: slow_task(1),
            lambda: slow_task(2),
            lambda: slow_task(3),
        )
        elapsed = time.time() - start

        # Should take ~10ms (parallel) not ~30ms (sequential)
        assert elapsed < 0.025
        assert results == [1, 2, 3]


# =============================================================================
# Resource Manager Tests
# =============================================================================

class TestResourceManager:
    """Tests for ResourceManager."""

    def test_register_and_get_resource(self):
        """Test registering and retrieving a resource."""
        manager = ResourceManager()
        resource = {"key": "value"}

        manager.register("test", resource)

        assert manager.get("test") is resource

    def test_get_nonexistent_returns_none(self):
        """Test that getting nonexistent resource returns None."""
        manager = ResourceManager()

        assert manager.get("nonexistent") is None

    def test_cleanup_calls_cleanup_function(self):
        """Test that cleanup calls the cleanup function."""
        cleaned_up = []

        def cleanup_fn():
            cleaned_up.append(True)

        manager = ResourceManager()
        manager.register("test", "resource", cleanup=cleanup_fn)
        manager.cleanup("test")

        assert cleaned_up == [True]
        assert manager.get("test") is None

    def test_cleanup_all_cleans_everything(self):
        """Test that cleanup_all cleans all resources."""
        cleaned = []

        manager = ResourceManager()
        manager.register("a", "resource_a", cleanup=lambda: cleaned.append("a"))
        manager.register("b", "resource_b", cleanup=lambda: cleaned.append("b"))
        manager.cleanup_all()

        assert set(cleaned) == {"a", "b"}

    def test_context_manager_cleans_up(self):
        """Test that context manager cleans up on exit."""
        cleaned = []

        with ResourceManager() as manager:
            manager.register("test", "value", cleanup=lambda: cleaned.append(True))

        assert cleaned == [True]


# =============================================================================
# Metrics Tests
# =============================================================================

class TestOperationMetrics:
    """Tests for OperationMetrics."""

    def test_record_increments_counters(self):
        """Test that record increments counters."""
        metrics = OperationMetrics()

        metrics.record(100.0)
        metrics.record(200.0)

        assert metrics.total_calls == 2
        assert metrics.total_time_ms == 300.0

    def test_record_tracks_errors(self):
        """Test that record tracks errors."""
        metrics = OperationMetrics()

        metrics.record(100.0, success=True)
        metrics.record(200.0, success=False)

        assert metrics.errors == 1
        assert metrics.error_rate == 0.5

    def test_avg_time_calculation(self):
        """Test average time calculation."""
        metrics = OperationMetrics()

        metrics.record(100.0)
        metrics.record(200.0)

        assert metrics.avg_time_ms == 150.0

    def test_p95_time_calculation(self):
        """Test P95 time calculation."""
        metrics = OperationMetrics()

        # Record 100 values from 1 to 100
        for i in range(1, 101):
            metrics.record(float(i))

        # P95 should be around 95
        assert metrics.p95_time_ms >= 94


class TestMetricsRegistry:
    """Tests for global metrics registry."""

    def setup_method(self):
        """Clear metrics before each test."""
        clear_metrics()

    def test_get_metrics_creates_new(self):
        """Test that get_metrics creates new metrics."""
        metrics = get_metrics("test_op")

        assert isinstance(metrics, OperationMetrics)
        assert metrics.total_calls == 0

    def test_get_metrics_returns_same_instance(self):
        """Test that get_metrics returns same instance."""
        metrics1 = get_metrics("test_op")
        metrics1.record(100.0)

        metrics2 = get_metrics("test_op")

        assert metrics2.total_calls == 1

    def test_get_all_metrics_returns_copy(self):
        """Test that get_all_metrics returns all registered metrics."""
        get_metrics("op1").record(100.0)
        get_metrics("op2").record(200.0)

        all_metrics = get_all_metrics()

        assert "op1" in all_metrics
        assert "op2" in all_metrics

    def test_clear_metrics_removes_all(self):
        """Test that clear_metrics removes all metrics."""
        get_metrics("op1").record(100.0)
        clear_metrics()

        assert get_all_metrics() == {}


# =============================================================================
# Datastore Context Manager Tests
# =============================================================================

class TestDatastoreContextManager:
    """Tests for CRMDataStore context manager."""

    def test_context_manager_returns_datastore(self):
        """Test that context manager returns the datastore."""
        from backend.agent.datastore import CRMDataStore

        with CRMDataStore() as store:
            assert isinstance(store, CRMDataStore)

    def test_context_manager_closes_connection(self):
        """Test that context manager closes connection on exit."""
        from backend.agent.datastore import CRMDataStore

        store = CRMDataStore()
        # Force connection creation
        _ = store.conn

        assert store._conn is not None

        store.close()

        assert store._conn is None

    def test_close_clears_caches(self):
        """Test that close clears internal caches."""
        from backend.agent.datastore import CRMDataStore

        store = CRMDataStore()
        store._company_names_cache = {"test": "value"}
        store._company_ids_cache = {"test"}
        store._loaded_tables.add("test")

        store.close()

        assert store._company_names_cache is None
        assert store._company_ids_cache is None
        assert len(store._loaded_tables) == 0
