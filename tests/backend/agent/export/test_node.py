"""Tests for export node."""

import os
import pytest

from backend.agent.export.node import (
    _detect_export_format,
    _extract_data_query,
)


class TestDetectExportFormat:
    """Tests for export format detection."""

    def test_detects_csv(self):
        """Should detect CSV format."""
        assert _detect_export_format("Export deals to CSV") == "csv"
        assert _detect_export_format("Download as spreadsheet") == "csv"
        assert _detect_export_format("Get Excel file") == "csv"

    def test_detects_pdf(self):
        """Should detect PDF format."""
        assert _detect_export_format("Export to PDF") == "pdf"
        assert _detect_export_format("Generate document") == "pdf"
        assert _detect_export_format("Create report") == "pdf"

    def test_detects_json(self):
        """Should detect JSON format."""
        assert _detect_export_format("Export as JSON") == "json"
        assert _detect_export_format("Get API response") == "json"

    def test_defaults_to_csv(self):
        """Should default to CSV for ambiguous queries."""
        assert _detect_export_format("Export all deals") == "csv"
        assert _detect_export_format("Download data") == "csv"


class TestExtractDataQuery:
    """Tests for data query extraction."""

    def test_extracts_from_export_query(self):
        """Should remove export keywords to get data query."""
        query = _extract_data_query("Export deals to CSV")
        assert "export" not in query.lower()
        assert "csv" not in query.lower()
        assert "deals" in query.lower()

    def test_extracts_from_download_query(self):
        """Should handle download syntax."""
        query = _extract_data_query("Download all contacts as spreadsheet")
        assert "download" not in query.lower()
        assert "spreadsheet" not in query.lower()
        assert "contacts" in query.lower()

    def test_preserves_data_terms(self):
        """Should preserve the actual data request."""
        query = _extract_data_query("Export Q1 revenue to PDF")
        assert "q1" in query.lower()
        assert "revenue" in query.lower()

    def test_handles_simple_export(self):
        """Should handle minimal export queries."""
        query = _extract_data_query("Export deals")
        assert "deals" in query.lower()
