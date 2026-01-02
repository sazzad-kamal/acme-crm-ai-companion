"""
Progress tracking for agent pipeline execution.

Provides a dataclass for tracking and logging pipeline steps.
"""

import logging
import time
from dataclasses import dataclass, field

from backend.agent.schemas import Step

logger = logging.getLogger(__name__)


@dataclass
class AgentProgress:
    """Tracks progress through the agent pipeline."""
    steps: list[Step] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add_step(self, step_id: str, label: str, status: str = "done") -> None:
        """Add a completed step."""
        self.steps.append(Step(id=step_id, label=label, status=status))
        logger.debug(f"Step: {step_id} - {label} [{status}]")

    def get_elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)

    def to_list(self) -> list[dict]:
        """Convert steps to list of dicts."""
        return [step.model_dump() for step in self.steps]
