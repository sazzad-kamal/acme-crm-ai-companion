"""
Context formatting functions for agent prompts.

These functions format CRM data into text sections
for inclusion in LLM prompts.
"""

from typing import Optional


def format_company_section(company_data: Optional[dict]) -> str:
    """Format company data for the prompt."""
    if not company_data:
        return ""
    
    company = company_data.get("company")
    if not company:
        return ""
    
    lines = [
        "=== COMPANY INFO ===",
        f"Name: {company.get('name', 'N/A')}",
        f"ID: {company.get('company_id', 'N/A')}",
        f"Status: {company.get('status', 'N/A')}",
        f"Plan: {company.get('plan', 'N/A')}",
        f"Industry: {company.get('industry', 'N/A')}",
        f"Region: {company.get('region', 'N/A')}",
        f"Account Owner: {company.get('account_owner', 'N/A')}",
        f"Renewal Date: {company.get('renewal_date', 'N/A')}",
        f"Health: {company.get('health_flags', 'N/A')}",
    ]
    
    # Add contacts if present
    contacts = company_data.get("contacts", [])
    if contacts:
        lines.append("\nKey Contacts:")
        for c in contacts[:3]:
            name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()
            lines.append(f"  - {name} ({c.get('job_title', 'N/A')}): {c.get('email', 'N/A')}")
    
    return "\n".join(lines)


def format_activities_section(activities_data: Optional[dict]) -> str:
    """Format activities data for the prompt."""
    if not activities_data:
        return ""
    
    activities = activities_data.get("activities", [])
    if not activities:
        return "=== RECENT ACTIVITIES ===\nNo recent activities found."
    
    lines = [
        f"=== RECENT ACTIVITIES ({activities_data.get('count', 0)} found, last {activities_data.get('days', 90)} days) ==="
    ]
    
    for act in activities[:8]:
        due = act.get("due_datetime", act.get("created_at", "N/A"))
        if due and "T" in str(due):
            due = str(due).split("T")[0]
        lines.append(
            f"- [{act.get('type', 'N/A')}] {act.get('subject', 'N/A')} "
            f"(Owner: {act.get('owner', 'N/A')}, Due: {due}, Status: {act.get('status', 'N/A')})"
        )
    
    return "\n".join(lines)


def format_history_section(history_data: Optional[dict]) -> str:
    """Format history data for the prompt."""
    if not history_data:
        return ""
    
    history = history_data.get("history", [])
    if not history:
        return "=== HISTORY LOG ===\nNo recent history entries."
    
    lines = [
        f"=== HISTORY LOG ({history_data.get('count', 0)} entries, last {history_data.get('days', 90)} days) ==="
    ]
    
    for h in history[:8]:
        occurred = h.get("occurred_at", "N/A")
        if occurred and "T" in str(occurred):
            occurred = str(occurred).split("T")[0]
        lines.append(
            f"- [{h.get('type', 'N/A')}] {h.get('subject', 'N/A')} "
            f"(Date: {occurred}, Owner: {h.get('owner', 'N/A')})"
        )
        if h.get("description"):
            desc = str(h.get("description", ""))[:100]
            if len(str(h.get("description", ""))) > 100:
                desc += "..."
            lines.append(f"    Note: {desc}")
    
    return "\n".join(lines)


def format_pipeline_section(pipeline_data: Optional[dict]) -> str:
    """Format pipeline data for the prompt."""
    if not pipeline_data:
        return ""
    
    summary = pipeline_data.get("summary", {})
    opps = pipeline_data.get("opportunities", [])
    
    if not summary.get("total_count"):
        return "=== PIPELINE ===\nNo open opportunities."
    
    lines = [
        f"=== PIPELINE SUMMARY ===",
        f"Total Open Deals: {summary.get('total_count', 0)}",
        f"Total Value: ${summary.get('total_value', 0):,.0f}",
        "\nBy Stage:"
    ]
    
    for stage, data in summary.get("stages", {}).items():
        lines.append(f"  - {stage}: {data.get('count', 0)} deals (${data.get('total_value', 0):,.0f})")
    
    if opps:
        lines.append("\nOpen Opportunities:")
        for opp in opps[:5]:
            close_date = opp.get("expected_close_date", "N/A")
            lines.append(
                f"  - {opp.get('name', 'N/A')}: {opp.get('stage', 'N/A')} - "
                f"${opp.get('value', 0):,} (Close: {close_date})"
            )
    
    return "\n".join(lines)


def format_renewals_section(renewals_data: Optional[dict]) -> str:
    """Format renewals data for the prompt."""
    if not renewals_data:
        return ""
    
    renewals = renewals_data.get("renewals", [])
    if not renewals:
        return "=== UPCOMING RENEWALS ===\nNo renewals in the specified timeframe."
    
    lines = [
        f"=== UPCOMING RENEWALS ({renewals_data.get('count', 0)} accounts, next {renewals_data.get('days', 90)} days) ==="
    ]
    
    for r in renewals[:10]:
        lines.append(
            f"- {r.get('name', 'N/A')} ({r.get('company_id', 'N/A')}): "
            f"Renewal {r.get('renewal_date', 'N/A')} | Plan: {r.get('plan', 'N/A')} | "
            f"Health: {r.get('health_flags', 'N/A')}"
        )
    
    return "\n".join(lines)


def format_docs_section(docs_answer: str) -> str:
    """Format docs RAG answer for the prompt."""
    if not docs_answer:
        return ""
    
    return f"=== DOCUMENTATION GUIDANCE ===\n{docs_answer}"
