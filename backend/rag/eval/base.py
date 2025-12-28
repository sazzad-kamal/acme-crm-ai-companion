"""
Base evaluation utilities for RAG evaluation.

Re-exports shared utilities from backend.common.eval_base
and adds any RAG-specific helpers.
"""

# Re-export all shared utilities
from backend.common.eval_base import (
    console,
    create_summary_table,
    create_detail_table,
    create_comparison_table,
    format_check_mark,
    format_percentage,
    format_latency,
    format_delta,
    print_eval_header,
    print_issues_panel,
    print_success_panel,
    add_separator_row,
    add_metric_row,
    save_results_json,
    load_results_json,
    compute_p95,
    compute_pass_rate,
    compare_to_baseline,
    save_baseline,
    print_baseline_comparison,
    REGRESSION_THRESHOLD,
)

__all__ = [
    "console",
    "create_summary_table",
    "create_detail_table",
    "create_comparison_table",
    "format_check_mark",
    "format_percentage",
    "format_latency",
    "format_delta",
    "print_eval_header",
    "print_issues_panel",
    "print_success_panel",
    "add_separator_row",
    "add_metric_row",
    "save_results_json",
    "load_results_json",
    "compute_p95",
    "compute_pass_rate",
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    "REGRESSION_THRESHOLD",
]
