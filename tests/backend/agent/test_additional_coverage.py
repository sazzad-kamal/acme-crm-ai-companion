"""
Additional coverage tests for tools/company.py and nodes/fetching.py.

Covers edge cases and error paths not hit by existing tests.
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Tests for tools/company.py
# =============================================================================


class TestToolContactLookup:
    """Tests for tool_contact_lookup function."""

    def test_contact_lookup_not_found(self):
        """Test contact lookup when contact doesn't exist."""
        from backend.agent.tools.company import tool_contact_lookup

        mock_ds = MagicMock()
        mock_ds.get_contact.return_value = None

        # Pass datastore directly (decorator accepts it)
        result = tool_contact_lookup("NONEXISTENT", datastore=mock_ds)

        assert result.data["found"] is False
        assert result.data["contact_id"] == "NONEXISTENT"
        assert "not found" in result.error

    def test_contact_lookup_found_with_company(self):
        """Test contact lookup with associated company."""
        from backend.agent.tools.company import tool_contact_lookup

        mock_contact = {
            "contact_id": "CONT001",
            "first_name": "John",
            "last_name": "Doe",
            "company_id": "COMP001",
        }
        mock_company = {"company_id": "COMP001", "name": "Acme Corp"}

        mock_ds = MagicMock()
        mock_ds.get_contact.return_value = mock_contact
        mock_ds.get_company.return_value = mock_company

        result = tool_contact_lookup("CONT001", datastore=mock_ds)

        assert result.data["found"] is True
        assert result.data["contact"]["first_name"] == "John"
        assert result.data["company"]["name"] == "Acme Corp"
        assert len(result.sources) == 1
        assert result.sources[0].type == "contact"


class TestToolGroupMembers:
    """Tests for tool_group_members function."""

    def test_group_members_not_found(self):
        """Test group members when group doesn't exist."""
        from backend.agent.tools.company import tool_group_members

        mock_ds = MagicMock()
        mock_ds.get_group.return_value = None
        mock_ds.get_all_groups.return_value = [
            {"group_id": "GRP1", "name": "Group 1"},
            {"group_id": "GRP2", "name": "Group 2"},
        ]

        result = tool_group_members("NONEXISTENT", datastore=mock_ds)

        assert result.data["found"] is False
        assert result.data["group_id"] == "NONEXISTENT"
        assert len(result.data["available_groups"]) == 2
        assert "not found" in result.error

    def test_group_members_found_with_members(self):
        """Test group members with existing group."""
        from backend.agent.tools.company import tool_group_members

        mock_group = {"group_id": "GRP1", "name": "Enterprise"}
        mock_members = [
            {"company_id": "COMP1", "name": "Company 1"},
            {"company_id": "COMP2", "name": "Company 2"},
        ]

        mock_ds = MagicMock()
        mock_ds.get_group.return_value = mock_group
        mock_ds.get_group_members.return_value = mock_members

        result = tool_group_members("GRP1", datastore=mock_ds)

        assert result.data["found"] is True
        assert result.data["count"] == 2
        assert len(result.sources) == 1
        assert result.sources[0].type == "group"

    def test_group_members_found_empty(self):
        """Test group members when group exists but has no members."""
        from backend.agent.tools.company import tool_group_members

        mock_group = {"group_id": "GRP1", "name": "Empty Group"}

        mock_ds = MagicMock()
        mock_ds.get_group.return_value = mock_group
        mock_ds.get_group_members.return_value = []

        result = tool_group_members("GRP1", datastore=mock_ds)

        assert result.data["found"] is True
        assert result.data["count"] == 0
        assert result.sources == []  # No sources when no members


class TestToolAccountsNeedingAttention:
    """Tests for tool_accounts_needing_attention function."""

    def test_accounts_needing_attention_no_filter(self):
        """Test getting accounts needing attention without owner filter."""
        from backend.agent.tools.company import tool_accounts_needing_attention

        mock_accounts = [
            {"company_id": "COMP1", "name": "At Risk Corp", "status": "at_risk"},
            {"company_id": "COMP2", "name": "Trial Co", "status": "trial"},
        ]

        mock_ds = MagicMock()
        mock_ds.get_accounts_needing_attention.return_value = mock_accounts

        result = tool_accounts_needing_attention(datastore=mock_ds)

        assert result.data["count"] == 2
        assert result.data["owner_filter"] is None
        mock_ds.get_accounts_needing_attention.assert_called_once_with(owner=None, limit=20)

    def test_accounts_needing_attention_with_owner(self):
        """Test getting accounts needing attention with owner filter."""
        from backend.agent.tools.company import tool_accounts_needing_attention

        mock_accounts = [{"company_id": "COMP1", "name": "At Risk Corp"}]

        mock_ds = MagicMock()
        mock_ds.get_accounts_needing_attention.return_value = mock_accounts

        result = tool_accounts_needing_attention(owner="John Smith", limit=10, datastore=mock_ds)

        assert result.data["count"] == 1
        assert result.data["owner_filter"] == "John Smith"
        mock_ds.get_accounts_needing_attention.assert_called_once_with(owner="John Smith", limit=10)


class TestToolSearchAttachmentsEdgeCases:
    """Tests for tool_search_attachments edge cases."""

    def test_search_attachments_with_file_type(self):
        """Test searching attachments by file type."""
        from backend.agent.tools.company import tool_search_attachments

        mock_attachments = [
            {"attachment_id": "ATT1", "file_name": "proposal.pdf", "file_type": "pdf"},
        ]

        mock_ds = MagicMock()
        mock_ds.search_attachments.return_value = mock_attachments

        result = tool_search_attachments(file_type="pdf", datastore=mock_ds)

        assert result.data["count"] == 1
        assert result.data["filters"]["file_type"] == "pdf"


# =============================================================================
# Tests for nodes/fetching.py
# =============================================================================


class TestFetchCrmData:
    """Tests for _fetch_crm_data function."""

    def test_fetch_crm_data_success(self):
        """Test successful CRM data fetch."""
        from backend.agent.nodes.fetching import _fetch_crm_data

        mock_result = MagicMock()
        mock_result.company_data = {"name": "Acme"}
        mock_result.activities_data = []
        mock_result.history_data = []
        mock_result.pipeline_data = {}
        mock_result.renewals_data = []
        mock_result.contacts_data = []
        mock_result.groups_data = []
        mock_result.attachments_data = []
        mock_result.resolved_company_id = "COMP001"
        mock_result.sources = []
        mock_result.raw_data = {}

        with patch("backend.agent.nodes.fetching.dispatch_intent", return_value=mock_result):
            result = _fetch_crm_data(
                question="What is Acme status?",
                intent="company_status",
                resolved_company_id="COMP001",
                days=30,
                router_result=None,
                owner=None,
            )

        assert result["company_data"]["name"] == "Acme"
        assert result["resolved_company_id"] == "COMP001"

    def test_fetch_crm_data_exception(self):
        """Test CRM data fetch handles exceptions."""
        from backend.agent.nodes.fetching import _fetch_crm_data

        with patch(
            "backend.agent.nodes.fetching.dispatch_intent",
            side_effect=Exception("Database error"),
        ):
            result = _fetch_crm_data(
                question="What is Acme status?",
                intent="company_status",
                resolved_company_id="COMP001",
                days=30,
                router_result=None,
            )

        assert "error" in result
        assert "Database error" in result["error"]


class TestFetchNode:
    """Tests for fetch_node function."""

    def test_fetch_node_basic(self):
        """Test basic fetch node execution."""
        from backend.agent.nodes.fetching import fetch_node

        state = {
            "question": "What is pipeline status?",
            "intent": "pipeline",
            "resolved_company_id": None,
            "days": 30,
            "router_result": None,
        }

        mock_crm_result = {
            "company_data": None,
            "activities_data": [],
            "history_data": [],
            "pipeline_data": {"total": 100000},
            "renewals_data": [],
            "contacts_data": [],
            "groups_data": [],
            "attachments_data": [],
            "resolved_company_id": None,
            "sources": [],
            "raw_data": {},
        }

        mock_docs_result = {
            "docs_answer": "Pipeline documentation",
            "docs_sources": [],
        }

        with patch("backend.agent.nodes.fetching._fetch_crm_data", return_value=mock_crm_result), \
             patch("backend.agent.nodes.fetching._fetch_docs", return_value=mock_docs_result), \
             patch("backend.agent.nodes.fetching.get_config") as mock_config:
            mock_config.return_value.default_days = 30
            mock_config.return_value.fetch_timeout_seconds = 30

            result = fetch_node(state)

        assert result["pipeline_data"]["total"] == 100000
        assert result["docs_answer"] == "Pipeline documentation"
        assert "fetch_latency_ms" in result

    def test_fetch_node_with_account_context(self):
        """Test fetch node with account context RAG."""
        from backend.agent.nodes.fetching import fetch_node

        state = {
            "question": "What is company status?",
            "intent": "company_status",  # In ACCOUNT_RAG_INTENTS
            "resolved_company_id": "COMP001",
            "days": 30,
            "router_result": None,
        }

        mock_crm_result = {
            "company_data": {"name": "Acme"},
            "activities_data": [],
            "history_data": [],
            "pipeline_data": {},
            "renewals_data": [],
            "contacts_data": [],
            "groups_data": [],
            "attachments_data": [],
            "resolved_company_id": "COMP001",
            "sources": [],
            "raw_data": {},
        }

        mock_docs_result = {"docs_answer": "", "docs_sources": []}

        mock_account_result = {
            "account_context_answer": "Recent notes about Acme",
            "account_context_sources": [],
        }

        with patch("backend.agent.nodes.fetching._fetch_crm_data", return_value=mock_crm_result), \
             patch("backend.agent.nodes.fetching._fetch_docs", return_value=mock_docs_result), \
             patch("backend.agent.nodes.fetching._fetch_account_context", return_value=mock_account_result), \
             patch("backend.agent.nodes.fetching.get_config") as mock_config:
            mock_config.return_value.default_days = 30
            mock_config.return_value.fetch_timeout_seconds = 30

            result = fetch_node(state)

        assert result["company_data"]["name"] == "Acme"
        assert result["account_context_answer"] == "Recent notes about Acme"
        # Should have 3 steps (data, docs, account_context)
        assert len(result["steps"]) == 3

    def test_fetch_node_with_timeout(self):
        """Test fetch node handles timeout."""
        from backend.agent.nodes.fetching import fetch_node
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        state = {
            "question": "What is pipeline?",
            "intent": "pipeline",
            "resolved_company_id": None,
            "days": 30,
            "router_result": None,
        }

        def slow_fetch(*args, **kwargs):
            raise FuturesTimeoutError()

        with patch("backend.agent.nodes.fetching._fetch_crm_data", side_effect=slow_fetch), \
             patch("backend.agent.nodes.fetching._fetch_docs", return_value={"docs_answer": "", "docs_sources": []}), \
             patch("backend.agent.nodes.fetching.get_config") as mock_config:
            mock_config.return_value.default_days = 30
            mock_config.return_value.fetch_timeout_seconds = 1

            result = fetch_node(state)

        assert "error" in result
        assert "timeout" in result["error"]

    def test_fetch_node_with_router_owner(self):
        """Test fetch node extracts owner from router result."""
        from backend.agent.nodes.fetching import fetch_node

        mock_router = MagicMock()
        mock_router.owner = "John Smith"

        state = {
            "question": "What is my pipeline?",
            "intent": "pipeline",
            "resolved_company_id": None,
            "days": 30,
            "router_result": mock_router,
        }

        mock_crm_result = {
            "company_data": None,
            "activities_data": [],
            "history_data": [],
            "pipeline_data": {},
            "renewals_data": [],
            "contacts_data": [],
            "groups_data": [],
            "attachments_data": [],
            "resolved_company_id": None,
            "sources": [],
            "raw_data": {},
        }

        with patch("backend.agent.nodes.fetching._fetch_crm_data", return_value=mock_crm_result) as mock_fetch, \
             patch("backend.agent.nodes.fetching._fetch_docs", return_value={"docs_answer": "", "docs_sources": []}), \
             patch("backend.agent.nodes.fetching.get_config") as mock_config:
            mock_config.return_value.default_days = 30
            mock_config.return_value.fetch_timeout_seconds = 30

            fetch_node(state)

        # Verify owner was passed to _fetch_crm_data
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["owner"] == "John Smith"

    def test_fetch_node_error_in_results(self):
        """Test fetch node handles errors in individual fetch results."""
        from backend.agent.nodes.fetching import fetch_node

        state = {
            "question": "What is pipeline?",
            "intent": "pipeline",
            "resolved_company_id": None,
            "days": 30,
            "router_result": None,
        }

        mock_crm_result = {
            "error": "CRM fetch failed",
            "sources": [],
        }

        mock_docs_result = {
            "docs_answer": "",
            "docs_sources": [],
            "error": "Docs fetch failed",
        }

        with patch("backend.agent.nodes.fetching._fetch_crm_data", return_value=mock_crm_result), \
             patch("backend.agent.nodes.fetching._fetch_docs", return_value=mock_docs_result), \
             patch("backend.agent.nodes.fetching.get_config") as mock_config:
            mock_config.return_value.default_days = 30
            mock_config.return_value.fetch_timeout_seconds = 30

            result = fetch_node(state)

        # Check steps show error status
        data_step = next(s for s in result["steps"] if s["id"] == "data")
        docs_step = next(s for s in result["steps"] if s["id"] == "docs")
        assert data_step["status"] == "error"
        assert docs_step["status"] == "error"


# =============================================================================
# Additional Branch Coverage Tests
# =============================================================================


class TestDatastoreSearchBranches:
    """Tests for datastore search filter branches."""

    def test_search_contacts_with_job_title(self):
        """Test search_contacts with job_title filter (lines 61-62)."""
        from backend.agent.datastore import CRMDataStore

        ds = CRMDataStore()
        # This should exercise the job_title branch
        contacts = ds.search_contacts(job_title="Engineer", limit=5)
        assert isinstance(contacts, list)

    def test_search_contacts_get_contact(self):
        """Test get_contact method (lines 20-21)."""
        from backend.agent.datastore import CRMDataStore

        ds = CRMDataStore()
        # Try to get a non-existent contact
        contact = ds.get_contact("NONEXISTENT_CONTACT")
        assert contact is None

    def test_search_companies_with_region(self):
        """Test search_companies with region filter (lines 115-116)."""
        from backend.agent.datastore import CRMDataStore

        ds = CRMDataStore()
        companies = ds.search_companies(region="West", limit=5)
        assert isinstance(companies, list)

    def test_search_activities_with_company(self):
        """Test search_activities with company_id filter (lines 74-75)."""
        from backend.agent.datastore import CRMDataStore

        ds = CRMDataStore()
        activities = ds.search_activities(company_id="COMP001", limit=5)
        assert isinstance(activities, list)


class TestToolBranches:
    """Tests for tool function branches."""

    def test_search_activities_with_type_filter(self):
        """Test tool_search_activities with activity_type filter (line 84)."""
        from backend.agent.tools.activity import tool_search_activities

        mock_ds = MagicMock()
        mock_ds.search_activities.return_value = [
            {"activity_id": "ACT001", "type": "Call"}
        ]

        result = tool_search_activities(activity_type="Call", datastore=mock_ds)
        assert "type='Call'" in result.sources[0].label

    def test_resolve_company_name_not_found(self):
        """Test _resolve_company_name when company not resolved (line 110)."""
        from backend.agent.tools.activity import _resolve_company_name

        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = None

        resolved_id, name = _resolve_company_name(mock_ds, "UNKNOWN")
        assert resolved_id is None
        assert name == ""

    def test_analytics_accounts_by_group(self):
        """Test tool_analytics with accounts_by_group metric (line 160)."""
        from backend.agent.tools.activity import tool_analytics

        mock_ds = MagicMock()
        mock_ds.get_accounts_by_group.return_value = {"total_groups": 5, "breakdown": []}
        mock_ds.resolve_company_id.return_value = None

        result = tool_analytics(metric="accounts_by_group", datastore=mock_ds)
        assert result.data["total_groups"] == 5

    def test_analytics_pipeline_by_group(self):
        """Test tool_analytics with pipeline_by_group metric (lines 173-174)."""
        from backend.agent.tools.activity import tool_analytics

        mock_ds = MagicMock()
        mock_ds.get_pipeline_by_group.return_value = {"breakdown": []}
        mock_ds.resolve_company_id.return_value = None

        result = tool_analytics(metric="pipeline_by_group", group_id="GRP001", datastore=mock_ds)
        assert "pipeline_GRP001" in result.sources[0].id

    def test_search_companies_with_all_filters(self):
        """Test tool_search_companies with industry, segment, status, region (lines 62,68,70,139)."""
        from backend.agent.tools.company import tool_search_companies

        mock_ds = MagicMock()
        mock_ds.search_companies.return_value = [{"company_id": "COMP001", "name": "Test"}]

        result = tool_search_companies(
            industry="Tech",
            segment="Enterprise",
            status="Active",
            region="West",
            datastore=mock_ds,
        )

        # Check all filters appear in the label
        label = result.sources[0].label
        assert "industry='Tech'" in label
        assert "segment='Enterprise'" in label
        assert "status='Active'" in label
        assert "region='West'" in label

    def test_forecast_accuracy_tool(self):
        """Test tool_forecast_accuracy (lines 151-157)."""
        from backend.agent.tools.pipeline import tool_forecast_accuracy

        mock_ds = MagicMock()
        mock_ds.get_forecast_accuracy.return_value = {
            "overall_win_rate": 65.5,
            "total_won": 100,
            "total_lost": 50,
            "total_closed": 150,  # Needed for sources to be generated
        }

        result = tool_forecast_accuracy(owner="John", datastore=mock_ds)
        assert result.data["overall_win_rate"] == 65.5
        assert len(result.sources) == 1
        assert "John" in result.sources[0].label

    def test_forecast_accuracy_tool_no_data(self):
        """Test tool_forecast_accuracy with no closed deals."""
        from backend.agent.tools.pipeline import tool_forecast_accuracy

        mock_ds = MagicMock()
        mock_ds.get_forecast_accuracy.return_value = {
            "overall_win_rate": 0,
            "total_closed": 0,
        }

        result = tool_forecast_accuracy(datastore=mock_ds)
        assert result.sources == []  # No sources when no data

