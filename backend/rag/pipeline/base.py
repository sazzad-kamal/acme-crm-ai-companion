"""
Shared pipeline utilities and base classes.

Contains:
- PipelineProgress: Progress tracking for RAG pipelines

For other utilities, import from:
- backend.common.context_builder: Context building (build_context, ContextBuilder)
- backend.rag.pipeline.gating: Chunk filtering (apply_lexical_gate, etc.)
"""

import logging
import time
from collections.abc import Callable


logger = logging.getLogger(__name__)


# =============================================================================
# Progress Tracking
# =============================================================================

class PipelineProgress:
    """
    Tracks and logs pipeline step progress.
    
    Useful for UI progress indicators and debugging.
    """
    
    def __init__(self, callback: Callable[[str, str, float], None] | None = None):
        """
        Initialize progress tracker.
        
        Args:
            callback: Optional function called with (step_id, label, elapsed_ms)
        """
        self.steps: list[dict] = []
        self.callback = callback
        self._start_time = time.time()
        self._step_start: float | None = None
    
    def start_step(self, step_id: str, label: str) -> None:
        """Start tracking a new step."""
        self._step_start = time.time()
        logger.info(f"[STEP] Starting: {label}")
        if self.callback:
            self.callback(step_id, f"Starting: {label}", 0)
    
    def complete_step(self, step_id: str, label: str, status: str = "done") -> None:
        """Mark a step as complete."""
        elapsed_ms = (time.time() - self._step_start) * 1000 if self._step_start else 0
        self.steps.append({
            "id": step_id,
            "label": label,
            "status": status,
            "elapsed_ms": elapsed_ms,
        })
        logger.info(f"[STEP] Completed: {label} ({elapsed_ms:.0f}ms) - {status}")
        if self.callback:
            self.callback(step_id, label, elapsed_ms)
    
    def get_steps(self) -> list[dict]:
        """Get all completed steps."""
        return self.steps
    
    def total_elapsed_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (time.time() - self._start_time) * 1000


# NOTE: Context building functions have been moved to backend.common.context_builder
# They are re-exported at the top of this module for backwards compatibility.
