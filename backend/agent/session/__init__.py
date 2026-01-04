"""
Session and conversation management module.

Provides conversation persistence, memory management, and caching
for agent sessions.
"""

from backend.agent.session.conversation import (
    get_checkpointer,
    get_session_state,
    get_session_messages,
    build_thread_config,
)
from backend.agent.session.memory import (
    clear_session,
    format_history_for_prompt,
)
from backend.agent.session.cache import (
    make_cache_key,
    get_cached_result,
    set_cached_result,
    clear_query_cache,
)

__all__ = [
    # Conversation
    "get_checkpointer",
    "get_session_state",
    "get_session_messages",
    "build_thread_config",
    # Memory
    "clear_session",
    "format_history_for_prompt",
    # Cache
    "make_cache_key",
    "get_cached_result",
    "set_cached_result",
    "clear_query_cache",
]
