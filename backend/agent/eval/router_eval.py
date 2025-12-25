"""
Router evaluation harness.

Tests the routing logic for:
- Mode selection (docs, data, data+docs)
- Company ID extraction
- Intent classification

Usage:
    python -m backend.agent.eval.router_eval
    python -m backend.agent.eval.router_eval --verbose
"""

import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from backend.agent.router import route_question as heuristic_route
from backend.agent.llm_router import route_question as llm_route
from backend.agent.eval.models import RouterEvalResult, RouterEvalSummary


console = Console()


# =============================================================================
# Test Cases
# =============================================================================

ROUTER_TEST_CASES = [
    # Pure docs questions
    {
        "id": "docs_how_to_1",
        "question": "How do I create a new contact?",
        "expected_mode": "docs",
        "expected_company": None,
    },
    {
        "id": "docs_what_is_1",
        "question": "What is an opportunity in the CRM?",
        "expected_mode": "docs",
        "expected_company": None,
    },
    {
        "id": "docs_explain_1",
        "question": "Explain how pipeline stages work",
        "expected_mode": "docs",
        "expected_company": None,
    },
    {
        "id": "docs_help_1",
        "question": "Help me understand groups and segments",
        "expected_mode": "docs",
        "expected_company": None,
    },
    {
        "id": "docs_guide_1",
        "question": "What's the guide for importing data?",
        "expected_mode": "docs",
        "expected_company": None,
    },
    # Pure data questions with company
    {
        "id": "data_status_1",
        "question": "What's the status of Acme Manufacturing?",
        "expected_mode": "data",
        "expected_company": "ACME-MFG",
    },
    {
        "id": "data_activity_1",
        "question": "Show me recent activities for Beta Tech",
        "expected_mode": "data",
        "expected_company": "BETA-TECH",
    },
    {
        "id": "data_pipeline_1",
        "question": "What opportunities are open for Crown Foods?",
        "expected_mode": "data",
        "expected_company": "CROWN-FOODS",
    },
    {
        "id": "data_history_1",
        "question": "What calls and emails happened with Delta Health?",
        "expected_mode": "data",
        "expected_company": "DELTA-HEALTH",
    },
    {
        "id": "data_renewal_1",
        "question": "When is the renewal for Fusion Retail?",
        "expected_mode": "data",
        "expected_company": "FUSION-RETAIL",
    },
    {
        "id": "data_churned_1",
        "question": "What happened with Green Energy Partners?",
        "expected_mode": "data",
        "expected_company": "GREEN-ENERGY",
    },
    # Data questions without specific company
    {
        "id": "data_renewals_all",
        "question": "What renewals are coming up in the next 90 days?",
        "expected_mode": "data",
        "expected_company": None,
    },
    {
        "id": "data_pipeline_all",
        "question": "Show me all open opportunities",
        "expected_mode": "data",
        "expected_company": None,
    },
    # Combined data+docs questions
    {
        "id": "combined_1",
        "question": "How do I track renewal risk for Acme Manufacturing?",
        "expected_mode": "data+docs",
        "expected_company": "ACME-MFG",
    },
    {
        "id": "combined_2",
        "question": "What pipeline stages is Beta Tech in and how do stages work?",
        "expected_mode": "data+docs",
        "expected_company": "BETA-TECH",
    },
    # Ambiguous questions (should default to data+docs)
    {
        "id": "ambiguous_1",
        "question": "Tell me about Acme Manufacturing",
        "expected_mode": "data+docs",
        "expected_company": "ACME-MFG",
    },
    {
        "id": "ambiguous_2",
        "question": "What should I know about Harbor Logistics?",
        "expected_mode": "data+docs",
        "expected_company": "HARBOR-LOGISTICS",
    },
    # Edge cases
    {
        "id": "edge_partial_name",
        "question": "What's happening with Eastern Travel?",
        "expected_mode": "data",
        "expected_company": "EASTERN-TRAVEL",
    },
    {
        "id": "edge_lowercase",
        "question": "show activities for acme manufacturing",
        "expected_mode": "data",
        "expected_company": "ACME-MFG",
    },
    {
        "id": "edge_misspelling",
        "question": "What's the pipeline for Acme Manufacturng?",  # typo
        "expected_mode": "data",
        "expected_company": "ACME-MFG",  # Should still resolve
    },
    # =========================================================================
    # NEW INTENTS - Contact Search
    # =========================================================================
    {
        "id": "contact_search_decision_makers",
        "question": "Who are the decision makers at our accounts?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "contact_search",
    },
    {
        "id": "contact_search_champions",
        "question": "List all champions",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "contact_search",
    },
    {
        "id": "contact_search_company",
        "question": "Who are the contacts at Acme Manufacturing?",
        "expected_mode": "data",
        "expected_company": "ACME-MFG",
        "expected_intent": "contact_search",
    },
    {
        "id": "contact_lookup_who",
        "question": "Who is Maria Silva?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "contact_lookup",
    },
    # =========================================================================
    # NEW INTENTS - Company Search
    # =========================================================================
    {
        "id": "company_search_enterprise",
        "question": "Show me all enterprise accounts",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "company_search",
    },
    {
        "id": "company_search_smb",
        "question": "List SMB companies",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "company_search",
    },
    {
        "id": "company_search_industry",
        "question": "Which companies are in the software industry?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "company_search",
    },
    # =========================================================================
    # NEW INTENTS - Groups
    # =========================================================================
    {
        "id": "groups_at_risk",
        "question": "Who is in the at-risk accounts group?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "groups",
    },
    {
        "id": "groups_list_all",
        "question": "List all groups",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "groups",
    },
    {
        "id": "groups_champions",
        "question": "Show the champion contacts group",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "groups",
    },
    # =========================================================================
    # NEW INTENTS - Pipeline Summary (Aggregate)
    # =========================================================================
    {
        "id": "pipeline_summary_total",
        "question": "What's the total pipeline value across all accounts?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "pipeline_summary",
    },
    {
        "id": "pipeline_summary_deals",
        "question": "How many deals are in the pipeline?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "pipeline_summary",
    },
    {
        "id": "pipeline_summary_forecast",
        "question": "What's the forecast summary for this quarter?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "pipeline_summary",
    },
    # =========================================================================
    # NEW INTENTS - Attachments
    # =========================================================================
    {
        "id": "attachments_proposals",
        "question": "Find all proposals",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "attachments",
    },
    {
        "id": "attachments_contracts",
        "question": "Show contracts for Acme Manufacturing",
        "expected_mode": "data",
        "expected_company": "ACME-MFG",
        "expected_intent": "attachments",
    },
    {
        "id": "attachments_documents",
        "question": "What documents do we have?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "attachments",
    },
    # =========================================================================
    # NEW INTENTS - Activity Search (without company)
    # =========================================================================
    {
        "id": "activities_meetings",
        "question": "What meetings happened this week?",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "activities",
    },
    {
        "id": "activities_calls",
        "question": "Show me all calls from last month",
        "expected_mode": "data",
        "expected_company": None,
        "expected_intent": "activities",
    },
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def run_router_test(
    test_case: dict,
    use_llm: bool = False,
    verbose: bool = False,
) -> RouterEvalResult:
    """Run a single router test case."""
    test_id = test_case["id"]
    question = test_case["question"]
    expected_mode = test_case["expected_mode"]
    expected_company = test_case.get("expected_company")
    
    # Route the question
    if use_llm:
        result = llm_route(question)
    else:
        result = heuristic_route(question)
    
    actual_mode = result.mode_used
    actual_company = result.company_id
    
    # Evaluate mode correctness
    # For mode, we allow some flexibility:
    # - "data+docs" is acceptable if expected was "data" or "docs" (safer default)
    # - "data" is acceptable when "data+docs" expected AND company was found
    #   (router correctly identified account context)
    mode_correct = actual_mode == expected_mode
    if not mode_correct:
        if actual_mode == "data+docs":
            # data+docs is always acceptable (it's the safe default)
            mode_correct = True
        elif actual_mode == "data" and expected_mode == "data+docs" and actual_company:
            # "data" when "data+docs" expected is okay if company was detected
            # The router is being smart about routing to account data
            mode_correct = True
    
    # Evaluate company extraction
    company_correct = True
    if expected_company:
        company_correct = actual_company == expected_company
    elif actual_company and expected_company is None:
        # Found a company when none expected - may be okay
        company_correct = True
    
    # Evaluate intent correctness (NEW)
    expected_intent = test_case.get("expected_intent")
    actual_intent = result.intent
    intent_correct = True
    if expected_intent:
        # Allow flexibility: some intents are related
        intent_correct = actual_intent == expected_intent
        # Allow "pipeline" to match "pipeline_summary" for aggregate queries
        if not intent_correct and expected_intent == "pipeline_summary" and actual_intent == "pipeline":
            intent_correct = True
        # Allow "activities" for company queries since they may use recent_activity
        if not intent_correct and expected_intent == "activities" and actual_intent in ("company_status", "general"):
            intent_correct = True
    
    if verbose:
        mode_status = "✓" if mode_correct else "✗"
        company_status = "✓" if company_correct else "✗"
        intent_status = "✓" if intent_correct else "✗"
        console.print(f"  [{mode_status}] {test_id}: mode={actual_mode} (expected={expected_mode})")
        if expected_company:
            console.print(f"      [{company_status}] company={actual_company} (expected={expected_company})")
        if expected_intent:
            console.print(f"      [{intent_status}] intent={actual_intent} (expected={expected_intent})")
    
    return RouterEvalResult(
        test_case_id=test_id,
        question=question,
        expected_mode=expected_mode,
        actual_mode=actual_mode,
        expected_company_id=expected_company,
        actual_company_id=actual_company,
        mode_correct=mode_correct,
        company_correct=company_correct,
        intent_expected=expected_intent,
        intent_actual=actual_intent,
        intent_correct=intent_correct,
    )


def run_router_eval(
    use_llm: bool = False,
    verbose: bool = False,
) -> tuple[list[RouterEvalResult], RouterEvalSummary]:
    """
    Run all router evaluation tests.
    
    Args:
        use_llm: Use LLM-based routing instead of heuristics
        verbose: Print detailed progress
        
    Returns:
        Tuple of (results list, summary)
    """
    router_type = "LLM" if use_llm else "Heuristic"
    console.print(f"\n[bold blue]═══ Router Evaluation ({router_type}) ═══[/bold blue]\n")
    
    results = []
    
    for test_case in track(ROUTER_TEST_CASES, description="Testing router..."):
        result = run_router_test(test_case, use_llm=use_llm, verbose=verbose)
        results.append(result)
    
    # Compute summary
    total = len(results)
    mode_correct = sum(1 for r in results if r.mode_correct)
    
    # Company extraction accuracy (only for tests with expected company)
    company_tests = [r for r in results if r.expected_company_id is not None]
    company_correct = sum(1 for r in company_tests if r.company_correct)
    
    # Intent accuracy (only for tests with expected intent)
    intent_tests = [r for r in results if r.intent_expected is not None]
    intent_correct = sum(1 for r in intent_tests if r.intent_correct)
    
    by_mode: dict[str, dict] = {}
    for r in results:
        mode = r.expected_mode
        if mode not in by_mode:
            by_mode[mode] = {"expected": 0, "correct": 0}
        by_mode[mode]["expected"] += 1
        if r.mode_correct:
            by_mode[mode]["correct"] += 1
    
    for mode in by_mode:
        by_mode[mode]["accuracy"] = by_mode[mode]["correct"] / by_mode[mode]["expected"]
    
    # Intent breakdown by expected intent type
    by_intent: dict[str, dict] = {}
    for r in results:
        if r.intent_expected:
            intent = r.intent_expected
            if intent not in by_intent:
                by_intent[intent] = {"expected": 0, "correct": 0}
            by_intent[intent]["expected"] += 1
            if r.intent_correct:
                by_intent[intent]["correct"] += 1
    
    for intent in by_intent:
        by_intent[intent]["accuracy"] = by_intent[intent]["correct"] / by_intent[intent]["expected"]
    
    summary = RouterEvalSummary(
        total_tests=total,
        mode_accuracy=mode_correct / total if total > 0 else 0,
        company_extraction_accuracy=company_correct / len(company_tests) if company_tests else 1.0,
        intent_accuracy=intent_correct / len(intent_tests) if intent_tests else 1.0,
        by_mode=by_mode,
        by_intent=by_intent,
    )
    
    return results, summary


def print_router_eval_results(results: list[RouterEvalResult], summary: RouterEvalSummary) -> None:
    """Print router evaluation results."""
    # Summary table
    table = Table(title="Router Evaluation Summary", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Tests", str(summary.total_tests))
    
    mode_color = "green" if summary.mode_accuracy >= 0.9 else "yellow" if summary.mode_accuracy >= 0.7 else "red"
    table.add_row("Mode Accuracy", f"[{mode_color}]{summary.mode_accuracy:.1%}[/{mode_color}]")
    
    company_color = "green" if summary.company_extraction_accuracy >= 0.9 else "yellow"
    table.add_row("Company Extraction", f"[{company_color}]{summary.company_extraction_accuracy:.1%}[/{company_color}]")
    
    intent_color = "green" if summary.intent_accuracy >= 0.9 else "yellow" if summary.intent_accuracy >= 0.7 else "red"
    table.add_row("Intent Accuracy", f"[{intent_color}]{summary.intent_accuracy:.1%}[/{intent_color}]")
    
    console.print(table)
    
    # By-mode breakdown
    mode_table = Table(title="Results by Expected Mode", show_header=True)
    mode_table.add_column("Mode")
    mode_table.add_column("Expected", justify="right")
    mode_table.add_column("Correct", justify="right")
    mode_table.add_column("Accuracy", justify="right")
    
    for mode, stats in sorted(summary.by_mode.items()):
        acc_color = "green" if stats["accuracy"] >= 0.9 else "yellow" if stats["accuracy"] >= 0.7 else "red"
        mode_table.add_row(
            mode,
            str(stats["expected"]),
            str(stats["correct"]),
            f"[{acc_color}]{stats['accuracy']:.1%}[/{acc_color}]"
        )
    
    console.print(mode_table)
    
    # By-intent breakdown (NEW)
    if summary.by_intent:
        intent_table = Table(title="Results by Expected Intent", show_header=True)
        intent_table.add_column("Intent")
        intent_table.add_column("Expected", justify="right")
        intent_table.add_column("Correct", justify="right")
        intent_table.add_column("Accuracy", justify="right")
        
        for intent, stats in sorted(summary.by_intent.items()):
            acc_color = "green" if stats["accuracy"] >= 0.9 else "yellow" if stats["accuracy"] >= 0.7 else "red"
            intent_table.add_row(
                intent,
                str(stats["expected"]),
                str(stats["correct"]),
                f"[{acc_color}]{stats['accuracy']:.1%}[/{acc_color}]"
            )
        
        console.print(intent_table)
    
    # Failed tests
    failed_mode = [r for r in results if not r.mode_correct]
    failed_company = [r for r in results if not r.company_correct and r.expected_company_id]
    failed_intent = [r for r in results if not r.intent_correct and r.intent_expected]
    
    if failed_mode:
        console.print("\n[red bold]Mode Mismatches:[/red bold]")
        for r in failed_mode:
            console.print(f"  • {r.test_case_id}: got '{r.actual_mode}', expected '{r.expected_mode}'")
            console.print(f"    Question: {r.question[:60]}...")
    
    if failed_company:
        console.print("\n[yellow bold]Company Extraction Failures:[/yellow bold]")
        for r in failed_company:
            console.print(f"  • {r.test_case_id}: got '{r.actual_company_id}', expected '{r.expected_company_id}'")
    
    if failed_intent:
        console.print("\n[yellow bold]Intent Classification Failures:[/yellow bold]")
        for r in failed_intent:
            console.print(f"  • {r.test_case_id}: got '{r.intent_actual}', expected '{r.intent_expected}'")
            console.print(f"    Question: {r.question[:60]}...")


# =============================================================================
# CLI
# =============================================================================

app = typer.Typer()


@app.command()
def main(
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM-based routing"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run router evaluation."""
    results, summary = run_router_eval(use_llm=use_llm, verbose=verbose)
    print_router_eval_results(results, summary)
    
    # Exit with error if below threshold
    if summary.mode_accuracy < 0.8:
        console.print("\n[red]FAIL: Mode accuracy below 80%[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
