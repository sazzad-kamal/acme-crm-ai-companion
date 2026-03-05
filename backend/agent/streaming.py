"""SSE streaming adapter for LangGraph agent execution."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from backend.agent.followup.tree.loader import get_starters
from backend.agent.graph import (
    agent_graph,
    build_thread_config,
)
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


class StreamEvent:
    FETCH_START = "fetch_start"
    FETCH_PROGRESS = "fetch_progress"
    ANSWER_CHUNK = "answer_chunk"
    ACTION_CHUNK = "action_chunk"
    DATA_READY = "data_ready"
    ACTION_READY = "action_ready"
    FOLLOWUP_READY = "followup_ready"
    DONE = "done"
    ERROR = "error"


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_agent(question: str, session_id: str | None = None) -> AsyncGenerator[str, None]:  # pragma: no cover
    """Stream agent execution as SSE events.

    Uses ainvoke for reliability - astream_events has threading issues on Railway.
    """
    config = build_thread_config(session_id)
    state: AgentState = {"question": question}

    print(f"[Stream] Starting graph for: {question[:50]}...", flush=True)

    try:
        # Use ainvoke instead of astream_events for reliability
        # astream_events has GIL/threading issues on Railway
        result = await asyncio.to_thread(
            lambda: agent_graph.invoke(state, config=config)
        )

        print(f"[Stream] Graph completed", flush=True)

        # Extract results
        sql_results = result.get("sql_results", {})
        answer = result.get("answer", "")
        follow_ups = result.get("follow_up_suggestions", [])
        action = result.get("suggested_action")

        # Emit data ready event
        yield _format_sse(StreamEvent.DATA_READY, {"sql_results": sql_results})

        # If no data, use starters for follow-ups
        if not sql_results.get("data") and not follow_ups:
            follow_ups = get_starters()

        # Emit action ready
        yield _format_sse(StreamEvent.ACTION_READY, {"suggested_action": action})

        # Emit followup ready
        yield _format_sse(StreamEvent.FOLLOWUP_READY, {"follow_up_suggestions": follow_ups})

        # Emit done with full result
        yield _format_sse(StreamEvent.DONE, {
            "answer": answer,
            "follow_up_suggestions": follow_ups,
            "suggested_action": action,
            "sql_results": sql_results,
        })

    except Exception as ex:
        logger.error("[Stream] %s", ex)
        print(f"[Stream] Error: {ex}", flush=True)
        yield _format_sse(StreamEvent.ERROR, {"message": str(ex)})


__all__ = ["stream_agent"]
