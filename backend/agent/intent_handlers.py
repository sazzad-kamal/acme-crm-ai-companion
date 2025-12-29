"""
Intent handlers for data_node dispatch.

Each handler fetches CRM data for a specific intent type.
Follows Open/Closed principle - add new intents without modifying data_node.
"""

import logging
from dataclasses import dataclass, field

from backend.agent.schemas import Source
from backend.agent.extractors import (
    extract_role_from_question,
    extract_company_criteria,
    extract_group_id,
    extract_attachment_query,
    extract_activity_type,
)
from backend.agent.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
    tool_search_contacts,
    tool_search_companies,
    tool_group_members,
    tool_list_groups,
    tool_search_attachments,
    tool_pipeline_summary,
    tool_search_activities,
    tool_analytics,
)


logger = logging.getLogger(__name__)


def _safe_extend(target_list: list, source_list: list | None) -> None:
    """Safely extend a list, handling None sources."""
    if source_list:
        target_list.extend(source_list)


@dataclass
class IntentContext:
    """Context passed to intent handlers."""
    question: str
    resolved_company_id: str | None
    days: int
    router_result: object | None = None


@dataclass
class IntentResult:
    """Result from an intent handler."""
    raw_data: dict = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    company_data: dict | None = None
    activities_data: dict | None = None
    history_data: dict | None = None
    pipeline_data: dict | None = None
    renewals_data: dict | None = None
    contacts_data: dict | None = None
    groups_data: dict | None = None
    attachments_data: dict | None = None
    analytics_data: dict | None = None
    resolved_company_id: str | None = None


def _empty_raw_data() -> dict:
    """Create empty raw_data structure."""
    return {
        "companies": [],
        "contacts": [],
        "activities": [],
        "opportunities": [],
        "history": [],
        "renewals": [],
        "groups": [],
        "attachments": [],
        "pipeline_summary": None,
        "analytics": None,
    }


def handle_pipeline_summary(ctx: IntentContext) -> IntentResult:
    """Handle pipeline_summary intent."""
    logger.debug("[Data] Fetching aggregate pipeline summary")
    result = IntentResult(raw_data=_empty_raw_data())

    summary_result = tool_pipeline_summary()
    result.pipeline_data = summary_result.data
    _safe_extend(result.sources, summary_result.sources)
    result.raw_data["pipeline_summary"] = {
        "total_count": result.pipeline_data.get("total_count"),
        "total_value": result.pipeline_data.get("total_value"),
        "by_stage": result.pipeline_data.get("by_stage", []),
    }
    result.raw_data["opportunities"] = result.pipeline_data.get("top_opportunities", [])[:8]
    return result


def handle_renewals(ctx: IntentContext) -> IntentResult:
    """Handle renewals intent."""
    result = IntentResult(raw_data=_empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    if ctx.resolved_company_id:
        company_result = tool_company_lookup(ctx.resolved_company_id)
        if company_result.data.get("found"):
            result.company_data = company_result.data
            _safe_extend(result.sources, company_result.sources)
            result.raw_data["companies"] = [result.company_data["company"]]

    logger.debug(f"[Data] Fetching renewals for next {ctx.days} days")
    renewals_result = tool_upcoming_renewals(days=ctx.days)
    result.renewals_data = renewals_result.data
    _safe_extend(result.sources, renewals_result.sources)
    result.raw_data["renewals"] = result.renewals_data.get("renewals", [])[:8]
    return result


def handle_contacts(ctx: IntentContext) -> IntentResult:
    """Handle contact_lookup and contact_search intents."""
    logger.debug("[Data] Handling contact query")
    result = IntentResult(raw_data=_empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    role = extract_role_from_question(ctx.question)
    if ctx.resolved_company_id:
        contacts_result = tool_search_contacts(company_id=ctx.resolved_company_id, role=role)
    else:
        contacts_result = tool_search_contacts(role=role)

    result.contacts_data = contacts_result.data
    _safe_extend(result.sources, contacts_result.sources)
    result.raw_data["contacts"] = result.contacts_data.get("contacts", [])[:10]
    return result


def handle_company_search(ctx: IntentContext) -> IntentResult:
    """Handle company_search intent."""
    logger.debug("[Data] Searching companies")
    result = IntentResult(raw_data=_empty_raw_data())

    segment, industry = extract_company_criteria(ctx.question)
    companies_result = tool_search_companies(segment=segment, industry=industry)
    result.company_data = companies_result.data
    _safe_extend(result.sources, companies_result.sources)
    result.raw_data["companies"] = result.company_data.get("companies", [])[:10]
    return result


def handle_groups(ctx: IntentContext) -> IntentResult:
    """Handle groups intent."""
    logger.debug("[Data] Handling groups query")
    result = IntentResult(raw_data=_empty_raw_data())

    group_id = extract_group_id(ctx.question)
    if group_id:
        members_result = tool_group_members(group_id)
        result.groups_data = members_result.data
        _safe_extend(result.sources, members_result.sources)
        result.raw_data["groups"] = [{
            "group_id": group_id,
            "name": result.groups_data.get("group_name"),
            "members": result.groups_data.get("members", [])[:10],
        }]
    else:
        groups_result = tool_list_groups()
        result.groups_data = groups_result.data
        _safe_extend(result.sources, groups_result.sources)
        result.raw_data["groups"] = result.groups_data.get("groups", [])[:10]
    return result


def handle_attachments(ctx: IntentContext) -> IntentResult:
    """Handle attachments intent."""
    logger.debug("[Data] Searching attachments")
    result = IntentResult(raw_data=_empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    query = extract_attachment_query(ctx.question)
    attachments_result = tool_search_attachments(query=query, company_id=ctx.resolved_company_id)
    result.attachments_data = attachments_result.data
    _safe_extend(result.sources, attachments_result.sources)
    result.raw_data["attachments"] = result.attachments_data.get("attachments", [])[:10]
    return result


def handle_activities(ctx: IntentContext) -> IntentResult:
    """Handle activities intent (cross-company search)."""
    logger.debug("[Data] Searching activities across all companies")
    result = IntentResult(raw_data=_empty_raw_data())

    activity_type = extract_activity_type(ctx.question)
    activities_result = tool_search_activities(activity_type=activity_type, days=ctx.days)
    result.activities_data = activities_result.data
    _safe_extend(result.sources, activities_result.sources)
    result.raw_data["activities"] = result.activities_data.get("activities", [])[:10]
    return result


def _detect_analytics_metric(question: str) -> tuple[str, str, str]:
    """
    Detect the analytics metric type from the question.

    Uses general patterns, not exact question matching.

    Returns:
        (metric, group_by, activity_type) tuple
    """
    q = question.lower()

    # Pattern: counting/breakdown keywords
    is_count_query = any(w in q for w in ["how many", "count", "total", "number of"])
    is_breakdown_query = any(w in q for w in ["breakdown", "distribution", "percentage", "split", "ratio"])
    is_comparison = any(w in q for w in ["most", "highest", "lowest", "compare", "common"])

    # Detect entity type
    has_contact = "contact" in q
    has_activity = "activit" in q
    has_account = "account" in q or "compan" in q
    has_group = "group" in q
    has_pipeline = "pipeline" in q or "deal" in q or "value" in q

    # Detect specific activity types
    activity_type = ""
    for atype in ["email", "call", "meeting", "demo", "task"]:
        if atype in q:
            activity_type = atype
            break

    # Decision logic based on entities
    if has_contact and (is_breakdown_query or "role" in q):
        return "contact_breakdown", "role", ""

    if has_activity:
        if activity_type and is_count_query:
            return "activity_count", "", activity_type
        if is_breakdown_query or is_comparison or "type" in q:
            return "activity_breakdown", "type", ""
        if is_count_query:
            return "activity_count", "", ""

    if has_group:
        if has_pipeline or has_account and ("value" in q or is_comparison):
            return "pipeline_by_group", "", ""
        if has_account or is_count_query or is_breakdown_query:
            return "accounts_by_group", "", ""

    if has_pipeline and has_group:
        return "pipeline_by_group", "", ""

    # Default based on query type
    if is_count_query:
        return "activity_count", "", ""
    if is_breakdown_query:
        return "activity_breakdown", "type", ""

    return "activity_breakdown", "type", ""


def _extract_group_id_for_analytics(question: str) -> str:
    """Extract group ID from analytics question."""
    q = question.lower()
    if "at-risk" in q or "at risk" in q:
        return "at-risk"
    if "enterprise" in q:
        return "enterprise"
    if "churn" in q:
        return "churned"
    if "strategic" in q:
        return "strategic"
    return ""


def handle_analytics(ctx: IntentContext) -> IntentResult:
    """Handle analytics intent (counts, breakdowns, aggregations)."""
    logger.debug("[Data] Processing analytics query")
    result = IntentResult(raw_data=_empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    # Detect the metric type from the question
    metric, group_by, activity_type = _detect_analytics_metric(ctx.question)

    # Extract group_id if needed
    group_id = _extract_group_id_for_analytics(ctx.question) if "group" in metric else ""

    logger.debug(f"[Analytics] metric={metric}, group_by={group_by}, activity_type={activity_type}, company={ctx.resolved_company_id}")

    analytics_result = tool_analytics(
        metric=metric,
        group_by=group_by,
        company_id=ctx.resolved_company_id or "",
        group_id=group_id,
        activity_type=activity_type,
        days=ctx.days,
    )

    result.analytics_data = analytics_result.data
    _safe_extend(result.sources, analytics_result.sources)
    result.raw_data["analytics"] = analytics_result.data

    return result


def handle_company_status(ctx: IntentContext) -> IntentResult:
    """Handle company_status and company-specific intents."""
    result = IntentResult(raw_data=_empty_raw_data())

    query = ctx.resolved_company_id
    if not query and ctx.router_result:
        query = getattr(ctx.router_result, 'company_name_query', None)

    logger.debug(f"[Data] Looking up company: {query}")
    company_result = tool_company_lookup(query or "")

    if company_result.data.get("found"):
        result.company_data = company_result.data
        _safe_extend(result.sources, company_result.sources)
        result.resolved_company_id = result.company_data["company"]["company_id"]
        result.raw_data["companies"] = [result.company_data["company"]]

        logger.debug(f"[Data] Fetching data for {result.resolved_company_id}")

        # Get activities
        activities_result = tool_recent_activity(result.resolved_company_id, days=ctx.days)
        result.activities_data = activities_result.data
        _safe_extend(result.sources, activities_result.sources)
        result.raw_data["activities"] = result.activities_data.get("activities", [])[:8]

        # Get history
        history_result = tool_recent_history(result.resolved_company_id, days=ctx.days)
        result.history_data = history_result.data
        _safe_extend(result.sources, history_result.sources)
        result.raw_data["history"] = result.history_data.get("history", [])[:8]

        # Get pipeline
        pipeline_result = tool_pipeline(result.resolved_company_id)
        result.pipeline_data = pipeline_result.data
        _safe_extend(result.sources, pipeline_result.sources)
        result.raw_data["opportunities"] = result.pipeline_data.get("opportunities", [])[:8]
        result.raw_data["pipeline_summary"] = result.pipeline_data.get("summary")

        logger.info(
            f"[Data] Fetched: activities={len(result.activities_data.get('activities', []))}, "
            f"history={len(result.history_data.get('history', []))}, "
            f"opps={len(result.pipeline_data.get('opportunities', []))}"
        )
    else:
        result.company_data = company_result.data
        logger.info(f"[Data] Company not found: {query}")

    return result


def handle_fallback(ctx: IntentContext) -> IntentResult:
    """Fallback handler for unknown intents."""
    logger.debug("[Data] No specific intent, fetching general renewals")
    result = IntentResult(raw_data=_empty_raw_data())

    renewals_result = tool_upcoming_renewals(days=ctx.days)
    result.renewals_data = renewals_result.data
    _safe_extend(result.sources, renewals_result.sources)
    result.raw_data["renewals"] = result.renewals_data.get("renewals", [])[:8]
    return result


# Intent dispatcher - maps intent strings to handler functions
# Explicit mappings for all router intents (no implicit fallthrough)
INTENT_HANDLERS = {
    # Aggregate queries (no company_id required)
    "pipeline_summary": handle_pipeline_summary,
    "renewals": handle_renewals,
    "activities": handle_activities,
    "company_search": handle_company_search,
    "groups": handle_groups,
    "attachments": handle_attachments,
    "analytics": handle_analytics,  # Counts, breakdowns, aggregations
    # Contact queries
    "contact_lookup": handle_contacts,
    "contact_search": handle_contacts,
    # Company-specific queries (all route to handle_company_status)
    "company_status": handle_company_status,
    "pipeline": handle_company_status,
    "history": handle_company_status,  # Explicit: was implicit fallthrough
    "account_context": handle_company_status,  # Explicit: triggers Account RAG in parallel node
    "general": handle_company_status,  # Explicit: fallback for ambiguous queries
}


def dispatch_intent(intent: str, ctx: IntentContext) -> IntentResult:
    """
    Dispatch to the appropriate intent handler.

    Args:
        intent: The intent string from router
        ctx: Context with question, company_id, days, etc.

    Returns:
        IntentResult with fetched data
    """
    # Special case: activities without company goes to activities handler
    if intent == "activities" and not ctx.resolved_company_id:
        return handle_activities(ctx)

    # Special case: analytics always goes to analytics handler (even with company_id)
    if intent == "analytics":
        return handle_analytics(ctx)

    # Special case: company-specific queries
    if ctx.resolved_company_id or (ctx.router_result and getattr(ctx.router_result, 'company_name_query', None)):
        if intent not in ("pipeline_summary", "company_search", "analytics"):
            return handle_company_status(ctx)

    # Normal dispatch
    handler = INTENT_HANDLERS.get(intent)
    if handler:
        return handler(ctx)

    return handle_fallback(ctx)


__all__ = [
    "IntentContext",
    "IntentResult",
    "dispatch_intent",
    "INTENT_HANDLERS",
]
