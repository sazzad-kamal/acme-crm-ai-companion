"""Answer node LLM chain functions."""

import json
import logging
import os
from datetime import datetime
from typing import Any

from backend.core.llm import LONG_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

# Check for demo mode
_DEMO_MODE = os.getenv("ACME_DEMO_MODE", "").lower() in ("true", "1")

_SYSTEM_PROMPT_BASE = """You are a CRM assistant. Answer questions using ONLY the provided context.
Today: {today}

RULES:
- Use exact numbers/dates from context
- Only say "data not available" if the answer cannot be found in the CRM DATA
- The CRM DATA contains SQL query results that directly answer the question - interpret and present them
- Use ALL provided data to formulate a complete answer
- Lead with key answer point
- Keep answers concise (2-4 sentences max, bullets when needed for details)

FORMAT: Currency $1,250,000 | Dates: March 31, 2026

EXAMPLES:
User: "What opportunities does Beta Tech have?"
Good: "Beta Tech has 3 open opportunities totaling $245,000.
- Largest: Enterprise renewal ($150,000, closes March 31)
- Champion: Sarah Chen (VP Engineering)
- Risk: Competitor evaluation in progress"
Bad: "They have several opportunities"
Bad: "Based on the provided data, I can confirm..."

User: "What's the renewal amount for Acme Corp?"
Good: "Renewal amount is not available in the current data."
Bad: "I don't have that information; amounts are tracked in the system but..."
"""

# Extended prompt for Act! demo mode with schema context
_ACT_SCHEMA_CONTEXT = """
## Act! CRM Data Context
You are answering questions about Act! CRM data. Key field mappings:

### Contacts
- fullName, firstName, lastName: Contact name
- company, companyID: Associated company
- emailAddress, businessPhone, mobilePhone: Contact info
- lastReach, lastMeeting, lastEmail: Activity tracking
- created, edited: Timestamps

### Opportunities (Deals)
- name: Deal name
- weightedTotal, productTotal: Deal value (NOT estimatedValue)
- estimatedCloseDate: When deal should close (NOT closeDate)
- probability: Win probability (0-100)
- status, statusName, stage: Pipeline position
- daysOpen, daysInStage: Time tracking
- contacts[]: Linked contacts array
- contactNames: Comma-separated contact names

### History (Past Activities)
- regarding: Activity subject/type
- details: Activity description
- startTime, endTime, duration: Timing
- historyType: Activity category
- contacts[]: Linked contacts

### Calendar (Upcoming Activities)
- date: Calendar date
- items[]: Activities on that date
- totalActivities, totalHistories: Counts
"""


def _get_system_prompt(today: str) -> str:
    """Get system prompt with optional Act! schema context."""
    base = _SYSTEM_PROMPT_BASE.format(today=today)
    if _DEMO_MODE:
        return base + _ACT_SCHEMA_CONTEXT
    return base

_HUMAN_PROMPT = """User's question: {question}

{conversation_history_section}

{sql_results_section}"""


def _get_answer_chain() -> Any:
    """Get answer chain with current date in system prompt."""
    today = datetime.now().strftime("%Y-%m-%d")
    chain = create_openai_chain(
        system_prompt=_get_system_prompt(today),
        human_prompt=_HUMAN_PROMPT,
        max_tokens=LONG_RESPONSE_MAX_TOKENS,
    )
    logger.debug("Created answer chain (demo_mode=%s)", _DEMO_MODE)
    return chain


def call_answer_chain(
    question: str,
    sql_results: dict[str, Any] | None = None,
    conversation_history: str = "",
    guidance: str = "",
) -> str:
    """Call the answer chain and return the answer string.

    Args:
        question: The user's question
        sql_results: SQL query results to use as context
        conversation_history: Previous conversation for context
        guidance: Optional guidance for how to interpret and present the data
    """
    # If guidance provided, append it to the question
    question_with_guidance = f"{question}\n\n[Guidance: {guidance}]" if guidance else question

    result: str = _get_answer_chain().invoke({
        "question": question_with_guidance,
        "conversation_history_section": f"=== RECENT CONVERSATION ===\n{conversation_history}\n" if conversation_history else "",
        "sql_results_section": f"=== CRM DATA ===\n{json.dumps(sql_results, indent=2, default=str)}\n" if sql_results else "",
    })
    return result


__all__ = ["call_answer_chain"]
