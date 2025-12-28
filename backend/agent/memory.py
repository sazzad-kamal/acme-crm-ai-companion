"""
Conversation memory for multi-turn conversations.

Provides in-memory storage of conversation history keyed by session_id.
Can be extended to use Redis or database for production persistence.
"""

import logging
from collections import defaultdict
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agent.state import Message


logger = logging.getLogger(__name__)

# In-memory storage: session_id -> list of messages
_memory_store: dict[str, list["Message"]] = defaultdict(list)
_memory_lock = Lock()

# Max messages to keep per session (to prevent unbounded growth)
MAX_MESSAGES_PER_SESSION = 20


def get_conversation_history(session_id: str | None) -> list["Message"]:
    """
    Retrieve conversation history for a session.

    Args:
        session_id: The session identifier (None returns empty list)

    Returns:
        List of messages in chronological order
    """
    if not session_id:
        return []

    with _memory_lock:
        messages = _memory_store.get(session_id, [])
        logger.debug(f"[Memory] Retrieved {len(messages)} messages for session {session_id}")
        return list(messages)  # Return a copy


def add_message(
    session_id: str | None,
    role: str,
    content: str,
    company_id: str | None = None,
) -> None:
    """
    Add a message to conversation history.

    Args:
        session_id: The session identifier (None = no-op)
        role: "user" or "assistant"
        content: The message content
        company_id: Optional company context for this message
    """
    if not session_id:
        return

    message: "Message" = {
        "role": role,
        "content": content,
        "company_id": company_id,
    }

    with _memory_lock:
        messages = _memory_store[session_id]
        messages.append(message)

        # Trim to max size (keep most recent)
        if len(messages) > MAX_MESSAGES_PER_SESSION:
            _memory_store[session_id] = messages[-MAX_MESSAGES_PER_SESSION:]
            logger.debug(f"[Memory] Trimmed session {session_id} to {MAX_MESSAGES_PER_SESSION} messages")

        logger.debug(f"[Memory] Added {role} message to session {session_id}")


def clear_session(session_id: str | None) -> None:
    """
    Clear all messages for a session.

    Args:
        session_id: The session to clear
    """
    if not session_id:
        return

    with _memory_lock:
        if session_id in _memory_store:
            del _memory_store[session_id]
            logger.debug(f"[Memory] Cleared session {session_id}")


def get_last_company_context(session_id: str | None) -> str | None:
    """
    Get the most recent company context from conversation history.

    Useful for resolving pronouns like "their" or "them" to the last
    mentioned company.

    Args:
        session_id: The session identifier

    Returns:
        The last company_id mentioned, or None
    """
    if not session_id:
        return None

    with _memory_lock:
        messages = _memory_store.get(session_id, [])
        # Walk backwards to find the last message with a company context
        for msg in reversed(messages):
            if msg.get("company_id"):
                return msg["company_id"]
        return None


def format_history_for_prompt(
    messages: list["Message"],
    max_messages: int = 6,
) -> str:
    """
    Format conversation history for inclusion in LLM prompts.

    Args:
        messages: List of messages
        max_messages: Maximum number of recent messages to include

    Returns:
        Formatted string for prompt inclusion
    """
    if not messages:
        return ""

    recent = messages[-max_messages:]
    lines = []

    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


__all__ = [
    "get_conversation_history",
    "add_message",
    "clear_session",
    "get_last_company_context",
    "format_history_for_prompt",
    "MAX_MESSAGES_PER_SESSION",
]
