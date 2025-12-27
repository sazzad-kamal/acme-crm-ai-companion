"""
Tool evaluation harness.

Tests each agent tool for correctness:
- company_lookup: Does it find the right company?
- recent_activity: Does it return correct activities?
- recent_history: Does it return correct history?
- pipeline: Does it return correct opportunities?
- upcoming_renewals: Does it return correct renewals?

Usage:
    python -m backend.agent.eval.tool_eval
    python -m backend.agent.eval.tool_eval --verbose
"""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.progress import track

from backend.agent.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
    # New comprehensive tools
    tool_contact_lookup,
    tool_search_contacts,
    tool_search_companies,
    tool_group_members,
    tool_list_groups,
    tool_search_attachments,
    tool_pipeline_summary,
    tool_search_activities,
)
from backend.agent.datastore import get_datastore
from backend.agent.eval.models import ToolEvalResult, ToolEvalSummary
from backend.agent.eval.base import (
    console,
    create_summary_table,
    format_percentage,
    print_eval_header,
)


# =============================================================================
# Test Cases
# =============================================================================

TOOL_TEST_CASES = [
    # Company Lookup - by ID
    {
        "id": "lookup_by_id_1",
        "tool": "company_lookup",
        "input": {"company_id_or_name": "ACME-MFG"},
        "expected": {"found": True, "company_id": "ACME-MFG"},
    },
    {
        "id": "lookup_by_id_2",
        "tool": "company_lookup",
        "input": {"company_id_or_name": "GREEN-ENERGY"},
        "expected": {"found": True, "company_id": "GREEN-ENERGY"},
    },
    # Company Lookup - by name
    {
        "id": "lookup_by_name_1",
        "tool": "company_lookup",
        "input": {"company_id_or_name": "Acme Manufacturing"},
        "expected": {"found": True, "company_id": "ACME-MFG"},
    },
    {
        "id": "lookup_by_name_2",
        "tool": "company_lookup",
        "input": {"company_id_or_name": "Beta Tech Solutions"},
        "expected": {"found": True, "company_id": "BETA-TECH"},
    },
    {
        "id": "lookup_by_name_partial",
        "tool": "company_lookup",
        "input": {"company_id_or_name": "Crown"},
        "expected": {"found": True, "company_id": "CROWN-FOODS"},
    },
    # Company Lookup - not found
    {
        "id": "lookup_not_found",
        "tool": "company_lookup",
        "input": {"company_id_or_name": "Nonexistent Corp"},
        "expected": {"found": False},
    },
    # Recent Activity
    {
        "id": "activity_acme",
        "tool": "recent_activity",
        "input": {"company_id": "ACME-MFG", "days": 365},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "activity_green",
        "tool": "recent_activity",
        "input": {"company_id": "GREEN-ENERGY", "days": 365},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "activity_all_companies",
        "tool": "recent_activity",
        "input": {"company_id": "BETA-TECH", "days": 365},
        "expected": {"found": True, "min_count": 1},
    },
    # Recent History
    {
        "id": "history_acme",
        "tool": "recent_history",
        "input": {"company_id": "ACME-MFG", "days": 365},
        "expected": {"found": True, "min_count": 3},
    },
    {
        "id": "history_green_churn",
        "tool": "recent_history",
        "input": {"company_id": "GREEN-ENERGY", "days": 365},
        "expected": {"found": True, "min_count": 3},
    },
    {
        "id": "history_harbor",
        "tool": "recent_history",
        "input": {"company_id": "HARBOR-LOGISTICS", "days": 365},
        "expected": {"found": True, "min_count": 2},
    },
    # Pipeline
    {
        "id": "pipeline_acme",
        "tool": "pipeline",
        "input": {"company_id": "ACME-MFG"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "pipeline_green_churned",
        "tool": "pipeline",
        "input": {"company_id": "GREEN-ENERGY"},
        "expected": {"found": False},  # No OPEN opps - company churned
    },
    {
        "id": "pipeline_fusion",
        "tool": "pipeline",
        "input": {"company_id": "FUSION-RETAIL"},
        "expected": {"found": True, "min_count": 1},
    },
    # Upcoming Renewals
    {
        "id": "renewals_all",
        "tool": "upcoming_renewals",
        "input": {"days": 365},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "renewals_90d",
        "tool": "upcoming_renewals",
        "input": {"days": 90},
        "expected": {"found": True, "min_count": 0},  # May or may not have
    },
    # =========================================================================
    # NEW TOOLS - Contact Search
    # =========================================================================
    {
        "id": "search_contacts_by_company",
        "tool": "search_contacts",
        "input": {"company_id": "ACME-MFG"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_contacts_decision_makers",
        "tool": "search_contacts",
        "input": {"role": "Decision Maker"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_contacts_champions",
        "tool": "search_contacts",
        "input": {"role": "Champion"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_contacts_all",
        "tool": "search_contacts",
        "input": {},
        "expected": {"found": True, "min_count": 5},
    },
    # =========================================================================
    # NEW TOOLS - Company Search
    # =========================================================================
    {
        "id": "search_companies_midmarket",
        "tool": "search_companies",
        "input": {"segment": "Mid-market"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_companies_smb",
        "tool": "search_companies",
        "input": {"segment": "SMB"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_companies_software",
        "tool": "search_companies",
        "input": {"industry": "Software"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_companies_all",
        "tool": "search_companies",
        "input": {},
        "expected": {"found": True, "min_count": 5},
    },
    # =========================================================================
    # NEW TOOLS - Groups
    # =========================================================================
    {
        "id": "list_groups_all",
        "tool": "list_groups",
        "input": {},
        "expected": {"found": True, "min_count": 3},
    },
    {
        "id": "group_members_at_risk",
        "tool": "group_members",
        "input": {"group_id": "GRP-AT-RISK"},
        "expected": {"found": True, "min_count": 0},  # Group exists, may have members
    },
    {
        "id": "group_members_champions",
        "tool": "group_members",
        "input": {"group_id": "GRP-CHAMPIONS"},
        "expected": {"found": True, "min_count": 0},  # Group exists, may have members
    },
    # =========================================================================
    # NEW TOOLS - Pipeline Summary (Aggregate)
    # =========================================================================
    {
        "id": "pipeline_summary_all",
        "tool": "pipeline_summary",
        "input": {},
        "expected": {"found": True, "min_count": 1, "has_total": True},
    },
    # =========================================================================
    # NEW TOOLS - Attachments
    # =========================================================================
    {
        "id": "search_attachments_all",
        "tool": "search_attachments",
        "input": {},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_attachments_proposal",
        "tool": "search_attachments",
        "input": {"query": "proposal"},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_attachments_by_company",
        "tool": "search_attachments",
        "input": {"company_id": "ACME-MFG"},
        "expected": {"found": True, "min_count": 0},  # May or may not have
    },
    # =========================================================================
    # NEW TOOLS - Activity Search
    # =========================================================================
    {
        "id": "search_activities_all",
        "tool": "search_activities",
        "input": {"days": 365},
        "expected": {"found": True, "min_count": 1},
    },
    {
        "id": "search_activities_calls",
        "tool": "search_activities",
        "input": {"activity_type": "Call", "days": 365},
        "expected": {"found": True, "min_count": 0},
    },
    {
        "id": "search_activities_meetings",
        "tool": "search_activities",
        "input": {"activity_type": "Meeting", "days": 365},
        "expected": {"found": True, "min_count": 0},
    },
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def run_tool_test(test_case: dict, verbose: bool = False) -> ToolEvalResult:
    """Run a single tool test case."""
    tool_name = test_case["tool"]
    test_id = test_case["id"]
    input_params = test_case["input"]
    expected = test_case["expected"]
    
    start_time = time.time()
    error = None
    
    try:
        # Call the appropriate tool
        if tool_name == "company_lookup":
            result = tool_company_lookup(**input_params)
        elif tool_name == "recent_activity":
            result = tool_recent_activity(**input_params)
        elif tool_name == "recent_history":
            result = tool_recent_history(**input_params)
        elif tool_name == "pipeline":
            result = tool_pipeline(**input_params)
        elif tool_name == "upcoming_renewals":
            result = tool_upcoming_renewals(**input_params)
        # New tools
        elif tool_name == "contact_lookup":
            result = tool_contact_lookup(**input_params)
        elif tool_name == "search_contacts":
            result = tool_search_contacts(**input_params)
        elif tool_name == "search_companies":
            result = tool_search_companies(**input_params)
        elif tool_name == "group_members":
            result = tool_group_members(**input_params)
        elif tool_name == "list_groups":
            result = tool_list_groups(**input_params)
        elif tool_name == "search_attachments":
            result = tool_search_attachments(**input_params)
        elif tool_name == "pipeline_summary":
            result = tool_pipeline_summary(**input_params)
        elif tool_name == "search_activities":
            result = tool_search_activities(**input_params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
            
    except Exception as e:
        error = str(e)
        latency = (time.time() - start_time) * 1000
        return ToolEvalResult(
            tool_name=tool_name,
            test_case_id=test_id,
            input_params=input_params,
            expected_found=expected.get("found", True),
            actual_found=False,
            data_correct=False,
            sources_present=False,
            error=error,
            latency_ms=latency,
        )
    
    latency = (time.time() - start_time) * 1000
    
    # Evaluate results
    actual_found = False
    actual_count = 0
    actual_company_id = None
    data_correct = True
    
    if tool_name == "company_lookup":
        actual_found = result.data.get("found", False)
        if actual_found:
            actual_company_id = result.data.get("company", {}).get("company_id")
            if expected.get("company_id"):
                data_correct = actual_company_id == expected["company_id"]
    elif tool_name == "pipeline_summary":
        # Pipeline summary has different structure
        actual_count = result.data.get("total_count", 0)
        actual_found = actual_count > 0
        if expected.get("has_total"):
            data_correct = "total_value" in result.data and result.data.get("total_value", 0) >= 0
        if "min_count" in expected:
            data_correct = data_correct and actual_count >= expected["min_count"]
    elif tool_name == "list_groups":
        groups = result.data.get("groups", [])
        actual_count = len(groups)
        actual_found = actual_count > 0
        if "min_count" in expected:
            data_correct = actual_count >= expected["min_count"]
    elif tool_name == "group_members":
        members = result.data.get("members", [])
        actual_count = len(members)
        # Group is found if the response has found=True or has group info
        actual_found = result.data.get("found", False) or result.data.get("group") is not None
        if "min_count" in expected:
            data_correct = actual_count >= expected["min_count"]
    else:
        # Activity, history, pipeline, renewals, contacts, companies, attachments
        data = result.data
        if "activities" in data:
            actual_count = len(data["activities"])
        elif "history" in data:
            actual_count = len(data["history"])
        elif "opportunities" in data:
            actual_count = len(data["opportunities"])
        elif "renewals" in data:
            actual_count = len(data["renewals"])
        elif "contacts" in data:
            actual_count = len(data["contacts"])
        elif "companies" in data:
            actual_count = len(data["companies"])
        elif "attachments" in data:
            actual_count = len(data["attachments"])
        else:
            actual_count = data.get("count", 0)
        
        actual_found = actual_count > 0
        
        if "min_count" in expected:
            data_correct = actual_count >= expected["min_count"]
        elif "exact_count" in expected:
            data_correct = actual_count == expected["exact_count"]
    
    # Check if found matches expectation
    if expected.get("found") is not None:
        if actual_found != expected["found"]:
            data_correct = False
    
    sources_present = len(result.sources) > 0 if actual_found else True
    
    if verbose:
        status = "✓" if data_correct else "✗"
        console.print(f"  [{status}] {test_id}: found={actual_found}, count={actual_count}")
    
    return ToolEvalResult(
        tool_name=tool_name,
        test_case_id=test_id,
        input_params=input_params,
        expected_found=expected.get("found", True),
        actual_found=actual_found,
        expected_count=expected.get("min_count") or expected.get("exact_count"),
        actual_count=actual_count,
        expected_company_id=expected.get("company_id"),
        actual_company_id=actual_company_id,
        data_correct=data_correct,
        sources_present=sources_present,
        error=error,
        latency_ms=latency,
    )


def run_tool_eval(verbose: bool = False) -> tuple[list[ToolEvalResult], ToolEvalSummary]:
    """
    Run all tool evaluation tests.

    Returns:
        Tuple of (results list, summary)
    """
    print_eval_header(
        "[bold blue]Tool Evaluation[/bold blue]",
        "Testing agent tools for correctness",
    )
    
    results = []
    
    # Initialize datastore once
    _ = get_datastore()
    
    for test_case in track(TOOL_TEST_CASES, description="Testing tools..."):
        result = run_tool_test(test_case, verbose=verbose)
        results.append(result)
    
    # Compute summary
    total = len(results)
    passed = sum(1 for r in results if r.data_correct)
    failed = total - passed
    
    by_tool: dict[str, dict] = {}
    for r in results:
        if r.tool_name not in by_tool:
            by_tool[r.tool_name] = {"passed": 0, "failed": 0}
        if r.data_correct:
            by_tool[r.tool_name]["passed"] += 1
        else:
            by_tool[r.tool_name]["failed"] += 1
    
    for tool_name in by_tool:
        tool_total = by_tool[tool_name]["passed"] + by_tool[tool_name]["failed"]
        by_tool[tool_name]["accuracy"] = by_tool[tool_name]["passed"] / tool_total if tool_total > 0 else 0
    
    summary = ToolEvalSummary(
        total_tests=total,
        passed=passed,
        failed=failed,
        accuracy=passed / total if total > 0 else 0,
        by_tool=by_tool,
    )
    
    return results, summary


def print_tool_eval_results(results: list[ToolEvalResult], summary: ToolEvalSummary) -> None:
    """Print tool evaluation results."""
    # Summary table using shared helper
    table = create_summary_table("Tool Evaluation Summary")

    table.add_row("Total Tests", str(summary.total_tests))
    table.add_row("Passed", f"[green]{summary.passed}[/green]")
    table.add_row("Failed", f"[red]{summary.failed}[/red]")
    table.add_row("Accuracy", format_percentage(summary.accuracy))

    console.print(table)
    
    # By-tool breakdown
    tool_table = Table(title="Results by Tool", show_header=True)
    tool_table.add_column("Tool")
    tool_table.add_column("Passed", justify="right")
    tool_table.add_column("Failed", justify="right")
    tool_table.add_column("Accuracy", justify="right")
    
    for tool_name, stats in sorted(summary.by_tool.items()):
        acc_color = "green" if stats["accuracy"] >= 0.9 else "yellow" if stats["accuracy"] >= 0.7 else "red"
        tool_table.add_row(
            tool_name,
            str(stats["passed"]),
            str(stats["failed"]),
            f"[{acc_color}]{stats['accuracy']:.1%}[/{acc_color}]"
        )
    
    console.print(tool_table)
    
    # Failed tests detail
    failed = [r for r in results if not r.data_correct]
    if failed:
        console.print("\n[red bold]Failed Tests:[/red bold]")
        for r in failed:
            console.print(f"  • {r.test_case_id}: expected_found={r.expected_found}, actual_found={r.actual_found}")
            if r.error:
                console.print(f"    Error: {r.error}")


# =============================================================================
# CLI
# =============================================================================

app = typer.Typer()


@app.command()
def main(verbose: bool = False) -> None:
    """Run tool evaluation."""
    results, summary = run_tool_eval(verbose=verbose)
    print_tool_eval_results(results, summary)
    
    # Exit with error code if tests failed
    if summary.failed > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
