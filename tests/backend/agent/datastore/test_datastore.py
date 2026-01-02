"""Tests for CRM DataStore."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from backend.agent.datastore import (
    CRMDataStore,
    get_csv_base_path,
    CSV_TABLES,
    REQUIRED_TABLES,
)


class TestGetCsvBasePath:
    """Tests for get_csv_base_path function."""

    def test_returns_path(self):
        """Test returns a Path object."""
        path = get_csv_base_path()
        assert isinstance(path, Path)

    def test_path_exists(self):
        """Test returned path exists."""
        path = get_csv_base_path()
        assert path.exists()

    def test_path_is_directory(self):
        """Test returned path is a directory."""
        path = get_csv_base_path()
        assert path.is_dir()


class TestCsvTables:
    """Tests for CSV table configuration."""

    def test_csv_tables_defined(self):
        """Test CSV_TABLES has expected tables."""
        expected = ["companies", "contacts", "activities", "history", "opportunities"]
        for table in expected:
            assert table in CSV_TABLES

    def test_required_tables_subset_of_csv_tables(self):
        """Test required tables are in CSV_TABLES."""
        for table in REQUIRED_TABLES:
            assert table in CSV_TABLES


class TestCRMDataStoreInit:
    """Tests for CRMDataStore initialization."""

    def test_init_without_path(self):
        """Test init without explicit path uses auto-detection."""
        store = CRMDataStore()
        assert store._csv_path is None  # Lazy loaded

    def test_init_with_path(self):
        """Test init with explicit path."""
        test_path = Path("/tmp/test")
        store = CRMDataStore(csv_path=test_path)
        assert store._csv_path == test_path

    def test_lazy_connection(self):
        """Test connection is lazy-loaded."""
        store = CRMDataStore()
        assert store._conn is None  # Not created yet


class TestCRMDataStoreProperties:
    """Tests for CRMDataStore properties."""

    def test_csv_path_property(self):
        """Test csv_path property returns Path."""
        store = CRMDataStore()
        assert isinstance(store.csv_path, Path)

    def test_conn_property_creates_connection(self):
        """Test conn property creates DuckDB connection."""
        store = CRMDataStore()
        conn = store.conn
        assert conn is not None
        assert store._conn is not None


class TestCRMDataStoreResolveCompany:
    """Tests for resolve_company_id method."""

    def test_resolve_empty_returns_none(self):
        """Test empty input returns None."""
        store = CRMDataStore()
        assert store.resolve_company_id("") is None
        assert store.resolve_company_id(None) is None

    def test_resolve_by_id(self):
        """Test resolving by exact company ID."""
        store = CRMDataStore()
        # First call builds cache
        result = store.resolve_company_id("C001")
        # Either finds it or returns None, but should not error
        assert result is None or isinstance(result, str)

    def test_resolve_case_insensitive(self):
        """Test name matching is case-insensitive."""
        store = CRMDataStore()
        # The method should handle case-insensitive lookups
        result1 = store.resolve_company_id("acme")
        result2 = store.resolve_company_id("ACME")
        # Both should return same result (or None if not found)
        assert result1 == result2


class TestCRMDataStoreGetCompany:
    """Tests for get_company method."""

    def test_get_company_returns_dict_or_none(self):
        """Test get_company returns dict or None."""
        store = CRMDataStore()
        result = store.get_company("C001")
        assert result is None or isinstance(result, dict)

    def test_get_company_not_found(self):
        """Test get_company with invalid ID returns None."""
        store = CRMDataStore()
        result = store.get_company("INVALID_ID_XYZ123")
        assert result is None


class TestCRMDataStoreGetActivities:
    """Tests for get_recent_activities method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_recent_activities("C001", days=30, limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_recent_activities("C001", days=90, limit=5)
        assert len(result) <= 5

    def test_returns_dicts(self):
        """Test returns list of dicts."""
        store = CRMDataStore()
        result = store.get_recent_activities("C001", days=90, limit=10)
        for item in result:
            assert isinstance(item, dict)


class TestCRMDataStoreGetHistory:
    """Tests for get_recent_history method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_recent_history("C001", days=30, limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_recent_history("C001", days=90, limit=5)
        assert len(result) <= 5


class TestCRMDataStoreGetOpportunities:
    """Tests for get_open_opportunities method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_open_opportunities("C001", limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_open_opportunities("C001", limit=3)
        assert len(result) <= 3


class TestCRMDataStoreGetPipelineSummary:
    """Tests for get_pipeline_summary method."""

    def test_returns_dict(self):
        """Test returns a dict."""
        store = CRMDataStore()
        result = store.get_pipeline_summary("C001")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test result has expected keys."""
        store = CRMDataStore()
        result = store.get_pipeline_summary("C001")
        assert "stages" in result
        assert "total_count" in result
        assert "total_value" in result

    def test_stages_is_dict(self):
        """Test stages is a dict."""
        store = CRMDataStore()
        result = store.get_pipeline_summary("C001")
        assert isinstance(result["stages"], dict)


class TestCRMDataStoreGetRenewals:
    """Tests for get_upcoming_renewals method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=90, limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=365, limit=5)
        assert len(result) <= 5

    def test_returns_dicts(self):
        """Test returns list of dicts."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=90, limit=10)
        for item in result:
            assert isinstance(item, dict)


class TestCRMDataStoreGetContacts:
    """Tests for get_contacts_for_company method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_contacts_for_company("C001", limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_contacts_for_company("C001", limit=3)
        assert len(result) <= 3


class TestCRMDataStoreSearchContacts:
    """Tests for search_contacts method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.search_contacts(query="john", limit=10)
        assert isinstance(result, list)

    def test_with_role_filter(self):
        """Test with role filter."""
        store = CRMDataStore()
        result = store.search_contacts(role="Decision Maker", limit=10)
        assert isinstance(result, list)

    def test_with_company_filter(self):
        """Test with company_id filter."""
        store = CRMDataStore()
        result = store.search_contacts(company_id="C001", limit=10)
        assert isinstance(result, list)


class TestCRMDataStoreSearchCompanies:
    """Tests for search_companies method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.search_companies(query="acme", limit=10)
        assert isinstance(result, list)

    def test_with_industry_filter(self):
        """Test with industry filter."""
        store = CRMDataStore()
        result = store.search_companies(industry="Technology", limit=10)
        assert isinstance(result, list)

    def test_with_status_filter(self):
        """Test with status filter."""
        store = CRMDataStore()
        result = store.search_companies(status="Active", limit=10)
        assert isinstance(result, list)

    def test_empty_query_returns_all(self):
        """Test empty query returns results."""
        store = CRMDataStore()
        result = store.search_companies(limit=5)
        assert isinstance(result, list)


class TestCRMDataStoreCompanyNameMatches:
    """Tests for get_company_name_matches method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_company_name_matches("acme", limit=5)
        assert isinstance(result, list)

    def test_empty_query_returns_empty(self):
        """Test empty query returns empty list."""
        store = CRMDataStore()
        result = store.get_company_name_matches("", limit=5)
        assert result == []

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_company_name_matches("tech", limit=3)
        assert len(result) <= 3


class TestGetDatastoreThreadLocal:
    """Tests for thread-local get_datastore function."""

    def test_returns_datastore(self):
        """Test returns a CRMDataStore instance."""
        from backend.agent.datastore import get_datastore
        store = get_datastore()
        assert isinstance(store, CRMDataStore)

    def test_same_thread_same_instance(self):
        """Test same thread gets same instance."""
        from backend.agent.datastore import get_datastore
        store1 = get_datastore()
        store2 = get_datastore()
        assert store1 is store2

    def test_different_threads_different_instances(self):
        """Test different threads get different instances."""
        import threading
        from backend.agent.datastore import get_datastore

        results = {}

        def get_store_id(thread_name):
            store = get_datastore()
            results[thread_name] = id(store)

        # Get instance in main thread
        main_store = get_datastore()
        results["main"] = id(main_store)

        # Get instance in different thread
        thread = threading.Thread(target=get_store_id, args=("thread1",))
        thread.start()
        thread.join()

        # Different threads should have different instances
        assert results["main"] != results["thread1"]


# =============================================================================
# Pipeline Mixin Tests (coverage improvement)
# =============================================================================


class TestCRMDataStoreGetAllPipelineSummary:
    """Tests for get_all_pipeline_summary method."""

    def test_returns_dict(self):
        """Test returns a dict."""
        store = CRMDataStore()
        result = store.get_all_pipeline_summary()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test result has expected keys."""
        store = CRMDataStore()
        result = store.get_all_pipeline_summary()
        assert "total_count" in result
        assert "total_value" in result
        assert "by_stage" in result
        assert "top_companies" in result

    def test_total_count_is_int(self):
        """Test total_count is an integer."""
        store = CRMDataStore()
        result = store.get_all_pipeline_summary()
        assert isinstance(result["total_count"], int)

    def test_total_value_is_numeric(self):
        """Test total_value is numeric."""
        store = CRMDataStore()
        result = store.get_all_pipeline_summary()
        assert isinstance(result["total_value"], (int, float))

    def test_by_stage_is_dict(self):
        """Test by_stage is a dict."""
        store = CRMDataStore()
        result = store.get_all_pipeline_summary()
        assert isinstance(result["by_stage"], dict)

    def test_top_companies_is_list(self):
        """Test top_companies is a list."""
        store = CRMDataStore()
        result = store.get_all_pipeline_summary()
        assert isinstance(result["top_companies"], list)


class TestCRMDataStoreGetPipelineByOwner:
    """Tests for get_pipeline_by_owner method."""

    def test_returns_dict(self):
        """Test returns a dict."""
        store = CRMDataStore()
        result = store.get_pipeline_by_owner()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test result has expected keys."""
        store = CRMDataStore()
        result = store.get_pipeline_by_owner()
        assert "total_count" in result
        assert "total_value" in result
        assert "breakdown" in result
        assert "owner_filter" in result

    def test_with_owner_filter(self):
        """Test filtering by owner."""
        store = CRMDataStore()
        result = store.get_pipeline_by_owner(owner="John Doe")
        assert result["owner_filter"] == "John Doe"

    def test_breakdown_is_list(self):
        """Test breakdown is a list."""
        store = CRMDataStore()
        result = store.get_pipeline_by_owner()
        assert isinstance(result["breakdown"], list)

    def test_breakdown_item_has_expected_keys(self):
        """Test breakdown items have expected keys."""
        store = CRMDataStore()
        result = store.get_pipeline_by_owner()
        if result["breakdown"]:
            item = result["breakdown"][0]
            assert "owner" in item
            assert "deal_count" in item
            assert "total_value" in item
            assert "avg_days_in_stage" in item
            assert "percentage" in item


class TestCRMDataStoreGetDealsAtRisk:
    """Tests for get_deals_at_risk method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_deals_at_risk()
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_deals_at_risk(limit=5)
        assert len(result) <= 5

    def test_with_owner_filter(self):
        """Test filtering by owner."""
        store = CRMDataStore()
        result = store.get_deals_at_risk(owner="Jane Smith")
        assert isinstance(result, list)

    def test_with_custom_days_threshold(self):
        """Test with custom days threshold."""
        store = CRMDataStore()
        result = store.get_deals_at_risk(days_threshold=30)
        assert isinstance(result, list)

    def test_returns_dicts(self):
        """Test returns list of dicts."""
        store = CRMDataStore()
        result = store.get_deals_at_risk(limit=10)
        for item in result:
            assert isinstance(item, dict)


class TestCRMDataStoreGetForecast:
    """Tests for get_forecast method."""

    def test_returns_dict(self):
        """Test returns a dict."""
        store = CRMDataStore()
        result = store.get_forecast()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test result has expected keys."""
        store = CRMDataStore()
        result = store.get_forecast()
        assert "total_pipeline" in result
        assert "total_weighted" in result
        assert "by_stage" in result
        assert "by_owner" in result
        assert "owner_filter" in result

    def test_with_owner_filter(self):
        """Test filtering by owner."""
        store = CRMDataStore()
        result = store.get_forecast(owner="Bob Jones")
        assert result["owner_filter"] == "Bob Jones"

    def test_by_stage_is_dict(self):
        """Test by_stage is a dict."""
        store = CRMDataStore()
        result = store.get_forecast()
        assert isinstance(result["by_stage"], dict)

    def test_by_owner_is_dict(self):
        """Test by_owner is a dict."""
        store = CRMDataStore()
        result = store.get_forecast()
        assert isinstance(result["by_owner"], dict)

    def test_total_weighted_is_numeric(self):
        """Test total_weighted is numeric."""
        store = CRMDataStore()
        result = store.get_forecast()
        assert isinstance(result["total_weighted"], (int, float))

    def test_stage_probabilities_applied(self):
        """Test weighted values are less than or equal to pipeline."""
        store = CRMDataStore()
        result = store.get_forecast()
        # Weighted should be <= total (due to probability < 1)
        assert result["total_weighted"] <= result["total_pipeline"] or result["total_pipeline"] == 0


class TestCRMDataStoreGetForecastAccuracy:
    """Tests for get_forecast_accuracy method."""

    def test_returns_dict(self):
        """Test returns a dict."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test result has expected keys."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy()
        assert "overall_win_rate" in result
        assert "total_won" in result
        assert "total_lost" in result
        assert "total_won_value" in result
        assert "total_lost_value" in result
        assert "total_closed" in result
        assert "owner_filter" in result
        assert "by_owner" in result

    def test_with_owner_filter(self):
        """Test filtering by owner."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy(owner="Alice Smith")
        assert result["owner_filter"] == "Alice Smith"

    def test_overall_win_rate_is_numeric(self):
        """Test overall_win_rate is numeric."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy()
        assert isinstance(result["overall_win_rate"], (int, float))

    def test_overall_win_rate_is_percentage(self):
        """Test overall_win_rate is between 0 and 100."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy()
        assert 0 <= result["overall_win_rate"] <= 100

    def test_by_owner_is_dict(self):
        """Test by_owner is a dict."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy()
        assert isinstance(result["by_owner"], dict)

    def test_by_owner_item_has_expected_keys(self):
        """Test by_owner items have expected keys."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy()
        if result["by_owner"]:
            owner_id = list(result["by_owner"].keys())[0]
            item = result["by_owner"][owner_id]
            assert "won" in item
            assert "lost" in item
            assert "won_value" in item
            assert "lost_value" in item
            assert "win_rate" in item
            assert "total_closed" in item

    def test_total_closed_equals_won_plus_lost(self):
        """Test total_closed equals total_won + total_lost."""
        store = CRMDataStore()
        result = store.get_forecast_accuracy()
        assert result["total_closed"] == result["total_won"] + result["total_lost"]


class TestCRMDataStoreGetUpcomingRenewalsWithOwner:
    """Additional tests for get_upcoming_renewals with owner filter."""

    def test_with_owner_filter(self):
        """Test filtering by owner."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=90, owner="Test Owner", limit=10)
        assert isinstance(result, list)

    def test_with_large_days_window(self):
        """Test with large days window."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=365, limit=10)
        assert isinstance(result, list)


# =============================================================================
# Analytics Mixin Tests (coverage improvement)
# =============================================================================


class TestCRMDataStoreContactBreakdown:
    """Tests for get_contact_breakdown analytics method."""

    def test_returns_dict_with_expected_keys(self):
        """Test returns dict with expected keys."""
        store = CRMDataStore()
        result = store.get_contact_breakdown()
        assert "group_by" in result
        assert "total" in result
        assert "breakdown" in result

    def test_default_group_by_is_role(self):
        """Test default group_by is 'role'."""
        store = CRMDataStore()
        result = store.get_contact_breakdown()
        assert result["group_by"] == "role"

    def test_group_by_job_title(self):
        """Test grouping by job_title."""
        store = CRMDataStore()
        result = store.get_contact_breakdown(group_by="job_title")
        assert result["group_by"] == "job_title"

    def test_with_company_filter(self):
        """Test filtering by company_id."""
        store = CRMDataStore()
        result = store.get_contact_breakdown(company_id="C001")
        assert result["company_id"] == "C001"

    def test_breakdown_has_percentage(self):
        """Test breakdown items have percentage."""
        store = CRMDataStore()
        result = store.get_contact_breakdown()
        if result["breakdown"]:
            assert "percentage" in result["breakdown"][0]
            assert "count" in result["breakdown"][0]
            assert "category" in result["breakdown"][0]


class TestCRMDataStoreActivityBreakdown:
    """Tests for get_activity_breakdown analytics method."""

    def test_returns_dict_with_expected_keys(self):
        """Test returns dict with expected keys."""
        store = CRMDataStore()
        result = store.get_activity_breakdown()
        assert "group_by" in result
        assert "days" in result
        assert "total" in result
        assert "breakdown" in result

    def test_default_group_by_is_type(self):
        """Test default group_by is 'type'."""
        store = CRMDataStore()
        result = store.get_activity_breakdown()
        assert result["group_by"] == "type"

    def test_group_by_status(self):
        """Test grouping by status."""
        store = CRMDataStore()
        result = store.get_activity_breakdown(group_by="status")
        assert result["group_by"] == "status"

    def test_with_company_filter(self):
        """Test filtering by company_id."""
        store = CRMDataStore()
        result = store.get_activity_breakdown(company_id="C001", days=30)
        assert result["company_id"] == "C001"
        assert result["days"] == 30

    def test_respects_days_parameter(self):
        """Test respects days parameter."""
        store = CRMDataStore()
        result = store.get_activity_breakdown(days=90)
        assert result["days"] == 90


class TestCRMDataStoreActivityCountByFilter:
    """Tests for get_activity_count_by_filter method."""

    def test_returns_count(self):
        """Test returns count."""
        store = CRMDataStore()
        result = store.get_activity_count_by_filter()
        assert "count" in result
        assert isinstance(result["count"], int)

    def test_with_activity_type_filter(self):
        """Test filtering by activity_type."""
        store = CRMDataStore()
        result = store.get_activity_count_by_filter(activity_type="Call")
        assert result["activity_type"] == "Call"

    def test_with_company_filter(self):
        """Test filtering by company_id."""
        store = CRMDataStore()
        result = store.get_activity_count_by_filter(company_id="C001")
        assert result["company_id"] == "C001"

    def test_with_days_filter(self):
        """Test filtering by days."""
        store = CRMDataStore()
        result = store.get_activity_count_by_filter(days=60)
        assert result["days"] == 60

    def test_with_all_filters(self):
        """Test with all filters combined."""
        store = CRMDataStore()
        result = store.get_activity_count_by_filter(
            activity_type="Meeting",
            company_id="C002",
            days=30,
        )
        assert result["activity_type"] == "Meeting"
        assert result["company_id"] == "C002"
        assert result["days"] == 30


class TestCRMDataStoreAccountsNeedingAttention:
    """Tests for get_accounts_needing_attention method."""

    def test_returns_list(self):
        """Test returns list."""
        store = CRMDataStore()
        result = store.get_accounts_needing_attention()
        assert isinstance(result, list)

    def test_with_owner_filter(self):
        """Test with owner filter."""
        store = CRMDataStore()
        result = store.get_accounts_needing_attention(owner="jsmith")
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_accounts_needing_attention(limit=5)
        assert len(result) <= 5


class TestCRMDataStoreGroups:
    """Tests for group methods."""

    def test_get_group_returns_dict_or_none(self):
        """Test get_group returns dict or None."""
        store = CRMDataStore()
        result = store.get_group("nonexistent")
        # Either finds group or returns None
        assert result is None or isinstance(result, dict)

    def test_get_all_groups_returns_list(self):
        """Test get_all_groups returns list."""
        store = CRMDataStore()
        result = store.get_all_groups()
        assert isinstance(result, list)

    def test_get_group_members_returns_list(self):
        """Test get_group_members returns list."""
        store = CRMDataStore()
        result = store.get_group_members("G001")
        assert isinstance(result, list)

    def test_get_group_members_respects_limit(self):
        """Test get_group_members respects limit."""
        store = CRMDataStore()
        result = store.get_group_members("G001", limit=5)
        assert len(result) <= 5


class TestCRMDataStoreAccountsByGroup:
    """Tests for get_accounts_by_group method."""

    def test_returns_dict_with_expected_keys(self):
        """Test returns dict with expected keys."""
        store = CRMDataStore()
        result = store.get_accounts_by_group()
        assert "total_groups" in result
        assert "breakdown" in result

    def test_breakdown_is_list(self):
        """Test breakdown is a list."""
        store = CRMDataStore()
        result = store.get_accounts_by_group()
        assert isinstance(result["breakdown"], list)


class TestCRMDataStorePipelineByGroup:
    """Tests for get_pipeline_by_group method."""

    def test_returns_dict(self):
        """Test returns dict."""
        store = CRMDataStore()
        result = store.get_pipeline_by_group()
        assert isinstance(result, dict)

    def test_with_specific_group(self):
        """Test with specific group_id."""
        store = CRMDataStore()
        result = store.get_pipeline_by_group(group_id="G001")
        assert isinstance(result, dict)
        # Either returns group data or error
        assert "group_id" in result or "error" in result

    def test_breakdown_for_all_groups(self):
        """Test breakdown when no group specified."""
        store = CRMDataStore()
        result = store.get_pipeline_by_group()
        assert "breakdown" in result


class TestCRMDataStoreSearchAttachments:
    """Tests for search_attachments method."""

    def test_returns_list(self):
        """Test returns list."""
        store = CRMDataStore()
        result = store.search_attachments()
        assert isinstance(result, list)

    def test_with_query(self):
        """Test searching with query."""
        store = CRMDataStore()
        result = store.search_attachments(query="proposal")
        assert isinstance(result, list)

    def test_with_company_filter(self):
        """Test filtering by company_id."""
        store = CRMDataStore()
        result = store.search_attachments(company_id="C001")
        assert isinstance(result, list)

    def test_with_file_type_filter(self):
        """Test filtering by file_type."""
        store = CRMDataStore()
        result = store.search_attachments(file_type="pdf")
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.search_attachments(limit=5)
        assert len(result) <= 5

    def test_with_all_filters(self):
        """Test with all filters combined."""
        store = CRMDataStore()
        result = store.search_attachments(
            query="contract",
            company_id="C001",
            file_type="pdf",
            limit=10,
        )
        assert isinstance(result, list)
        assert len(result) <= 10
