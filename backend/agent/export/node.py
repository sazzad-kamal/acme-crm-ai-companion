"""Export node - handles export requests for CSV/PDF generation."""

import csv
import io
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import Any, cast

from backend.agent.fetch.node import fetch_node
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)

# Export directory (configurable via env)
EXPORT_DIR = os.getenv("EXPORT_DIR", tempfile.gettempdir())

# Export format patterns
EXPORT_PATTERNS = {
    "csv": [r"csv", r"spreadsheet", r"excel"],
    "pdf": [r"pdf", r"document", r"report"],
    "json": [r"json", r"api"],
}


def _detect_export_format(question: str) -> str:
    """Detect requested export format from question."""
    q_lower = question.lower()

    for fmt, patterns in EXPORT_PATTERNS.items():
        if any(re.search(pattern, q_lower) for pattern in patterns):
            return fmt

    # Default to CSV
    return "csv"


def _extract_data_query(question: str) -> str:
    """Extract the data query from an export request."""
    # Remove export-related words to get the underlying data query
    export_words = [
        r"export\s*(to\s*)?(as\s*)?",
        r"download\s*(as\s*)?",
        r"generate\s*(a\s*)?",
        r"create\s*(a\s*)?",
        r"save\s*(as\s*)?",
        r"(as\s*)?(csv|pdf|json|excel|spreadsheet|document|report)",
        r"file",
    ]

    data_query = question
    for pattern in export_words:
        data_query = re.sub(pattern, "", data_query, flags=re.IGNORECASE)

    # Clean up extra spaces
    data_query = " ".join(data_query.split())

    # If nothing left, use original question
    return data_query.strip() if data_query.strip() else question


def _generate_csv(data: list[dict[str, Any]], filename: str) -> str:
    """Generate CSV file and return file path."""
    if not data:
        return ""

    filepath = os.path.join(EXPORT_DIR, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        # Get all unique keys from data
        all_keys = set()
        for row in data:
            all_keys.update(row.keys())

        # Filter out internal fields
        columns = sorted([k for k in all_keys if not k.startswith("_")])

        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"[Export] Generated CSV: {filepath}")
    return filepath


def _generate_json(data: list[dict[str, Any]], filename: str) -> str:
    """Generate JSON file and return file path."""
    import json

    if not data:
        return ""

    filepath = os.path.join(EXPORT_DIR, filename)

    # Filter out internal fields
    clean_data = []
    for row in data:
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        clean_data.append(clean_row)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, default=str)

    logger.info(f"[Export] Generated JSON: {filepath}")
    return filepath


def _generate_pdf_placeholder(data: list[dict[str, Any]], filename: str) -> str:
    """Generate a placeholder for PDF (actual implementation would use reportlab)."""
    # For now, generate a text-based report as a placeholder
    # In production, this would use reportlab or weasyprint
    filepath = os.path.join(EXPORT_DIR, filename.replace(".pdf", ".txt"))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("CRM Data Export Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total Records: {len(data)}\n")
        f.write("=" * 50 + "\n\n")

        for i, row in enumerate(data[:50]):  # Limit to 50 rows for readability
            f.write(f"Record {i + 1}:\n")
            for key, value in row.items():
                if not key.startswith("_"):
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        if len(data) > 50:
            f.write(f"... and {len(data) - 50} more records\n")

    logger.info(f"[Export] Generated report: {filepath}")
    return filepath


def _generate_export_file(
    data: list[dict[str, Any]],
    export_format: str,
    question: str,
) -> tuple[str, str]:
    """Generate export file and return (filepath, download_url)."""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    # Create safe filename from question
    safe_name = re.sub(r"[^\w\s-]", "", question)[:30].strip()
    safe_name = re.sub(r"[-\s]+", "_", safe_name).lower()

    filename = f"export_{safe_name}_{timestamp}_{unique_id}.{export_format}"

    if export_format == "csv":
        filepath = _generate_csv(data, filename)
    elif export_format == "json":
        filepath = _generate_json(data, filename)
    elif export_format == "pdf":
        filepath = _generate_pdf_placeholder(data, filename)
    else:
        filepath = _generate_csv(data, filename.replace(export_format, "csv"))

    # Generate download URL (would be a real URL in production)
    download_url = f"/api/exports/{os.path.basename(filepath)}" if filepath else ""

    return filepath, download_url


def export_node(state: AgentState) -> AgentState:
    """Export node that generates downloadable files from query results."""
    question = state["question"]
    logger.info(f"[Export] Processing: {question[:50]}...")

    result: dict[str, Any] = {
        "sql_results": {},
    }

    # Step 1: Detect export format
    export_format = _detect_export_format(question)
    logger.info(f"[Export] Format: {export_format}")

    # Step 2: Extract the underlying data query
    data_query = _extract_data_query(question)
    logger.info(f"[Export] Data query: {data_query[:40]}...")

    # Step 3: Fetch the data using fetch node
    fetch_state = cast(
        AgentState,
        {
            "question": data_query,
            "messages": state.get("messages", []),
        },
    )

    fetch_result = fetch_node(fetch_state)
    data = fetch_result.get("sql_results", {}).get("data", [])

    if not data:
        result["error"] = "No data available to export"
        result["sql_results"] = {
            "export": {
                "status": "failed",
                "reason": "No data found",
            }
        }
        return cast(AgentState, result)

    # Step 4: Generate export file
    filepath, download_url = _generate_export_file(data, export_format, data_query)

    # Build result
    sql_results: dict[str, Any] = {
        "_debug": {
            "original_query": question,
            "data_query": data_query,
            "row_count": len(data),
        },
        "data": data,
        "export": {
            "status": "success",
            "format": export_format,
            "filepath": filepath,
            "download_url": download_url,
            "record_count": len(data),
        },
    }

    result["sql_results"] = sql_results
    logger.info(f"[Export] Complete: {len(data)} records exported to {export_format}")

    return cast(AgentState, result)


__all__ = ["export_node"]
