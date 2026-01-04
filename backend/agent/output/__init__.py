"""
Output processing module.

Provides formatting, extraction, and audit logging for agent responses.

Note: streaming is imported directly from backend.agent.output.streaming
to avoid circular imports with graph.py.
"""

from backend.agent.output.formatters import (
    SectionFormatter,
    FORMATTERS,
    format_section,
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_contacts_section,
    format_groups_section,
    format_attachments_section,
    format_docs_section,
    format_account_context_section,
    format_conversation_history_section,
)
from backend.agent.output.extractors import (
    extract_role_from_question,
    extract_company_criteria,
    extract_group_id,
    extract_attachment_query,
    extract_activity_type,
)
from backend.agent.output.audit import (
    AgentAuditEntry,
    AgentAuditLogger,
    get_audit_logger,
)

__all__ = [
    # Formatters
    "SectionFormatter",
    "FORMATTERS",
    "format_section",
    "format_company_section",
    "format_activities_section",
    "format_history_section",
    "format_pipeline_section",
    "format_renewals_section",
    "format_contacts_section",
    "format_groups_section",
    "format_attachments_section",
    "format_docs_section",
    "format_account_context_section",
    "format_conversation_history_section",
    # Extractors
    "extract_role_from_question",
    "extract_company_criteria",
    "extract_group_id",
    "extract_attachment_query",
    "extract_activity_type",
    # Audit
    "AgentAuditEntry",
    "AgentAuditLogger",
    "get_audit_logger",
]
