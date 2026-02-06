"""Thread-safe progress queue for real-time streaming.

Uses a module-level queue with thread-safe access for simplicity.
The queue is cleared at the start of each request.
"""

import queue
import threading
from typing import Any

# Module-level queue and lock for thread-safe access
_progress_queue: queue.Queue[dict[str, Any]] | None = None
_queue_lock = threading.Lock()


def create_progress_queue() -> queue.Queue[dict[str, Any]]:
    """Create a new progress queue for streaming.

    Call this at the start of streaming to set up progress collection.
    Clears any existing queue from previous requests.
    """
    global _progress_queue
    with _queue_lock:
        _progress_queue = queue.Queue()
        return _progress_queue


def emit_progress(step: str, status: str) -> None:
    """Emit a progress event to the queue.

    Thread-safe - can be called from ThreadPoolExecutor workers.
    """
    with _queue_lock:
        if _progress_queue is not None:
            _progress_queue.put({"step": step, "status": status})


def clear_progress_queue() -> None:
    """Clear the progress queue."""
    global _progress_queue
    with _queue_lock:
        _progress_queue = None


async def drain_progress_queue(
    q: queue.Queue[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drain all available progress events from the queue (non-blocking).

    Returns list of progress events that were in the queue.
    """
    events: list[dict[str, Any]] = []
    while True:
        try:
            event = q.get_nowait()
            events.append(event)
        except queue.Empty:
            break
    return events


__all__ = [
    "create_progress_queue",
    "emit_progress",
    "clear_progress_queue",
    "drain_progress_queue",
]
