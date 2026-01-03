"""
Shared evaluation utilities - facade module.

This module re-exports utilities from focused submodules for backward compatibility.
New code should import directly from the submodules:
- formatting: console, format_check_mark, format_percentage, etc.
- baseline: compare_to_baseline, save_baseline, REGRESSION_THRESHOLD
- parallel: run_parallel_evaluation, calculate_p95_latency
- slo: create_slo_table, print_slo_result, get_failed_slos, determine_exit_code
- results: parse_json_response, run_llm_judge, save_eval_results, finalize_eval_cli
"""

import json
from pathlib import Path
from typing import Any, Callable

# Re-export from formatting module
from backend.agent.eval.formatting import (
    console,
    format_check_mark,
    format_percentage,
    create_summary_table,
    print_eval_header,
    print_overall_result_panel,
    print_debug_failures,
    build_eval_table,
)

# Re-export from baseline module
from backend.agent.eval.baseline import (
    REGRESSION_THRESHOLD,
    compare_to_baseline,
    save_baseline,
    print_baseline_comparison,
)

# Re-export from parallel module
from backend.agent.eval.parallel import (
    run_parallel_evaluation,
    calculate_p95_latency,
)

# Re-export from SLO module
from backend.agent.eval.slo import (
    create_slo_table,
    print_slo_result,
    get_failed_slos,
    determine_exit_code,
)


# =============================================================================
# LLM Judge Utilities (kept here for dependency on llm_client)
# =============================================================================


def parse_json_response(text: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


def run_llm_judge(
    prompt: str,
    system_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict:
    """
    Run LLM judge and parse JSON response.

    Args:
        prompt: The evaluation prompt
        system_prompt: System prompt for the judge
        model: Model to use
        temperature: Temperature for generation
        max_tokens: Max tokens for response

    Returns:
        Parsed JSON result, or dict with 'error' key on failure
    """
    from backend.agent.eval.llm_client import call_llm

    try:
        response = call_llm(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response or not response.strip():
            return {"error": "Empty response from judge LLM"}

        return parse_json_response(response)
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Compatibility wrappers for complex functions with different signatures
# =============================================================================


def save_eval_results(
    output_path: str,
    summary: Any,
    results: list[Any],
    result_mapper: Callable[[Any], dict],
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        output_path: Path to save JSON results
        summary: Summary object (must have model_dump() or be dict)
        results: List of result objects
        result_mapper: Function to convert result object to dict
    """
    summary_dict = summary.model_dump() if hasattr(summary, "model_dump") else summary
    output_data = {
        "summary": summary_dict,
        "results": [result_mapper(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    console.print(f"[dim]Results saved to {output_path}[/dim]")


def finalize_eval_cli(
    primary_score: float,
    slo_checks: list[tuple[str, bool, str, str]],
    baseline_path: Path,
    score_key: str,
    set_baseline: bool = False,
    baseline_data: dict | None = None,
    extra_failure_check: bool = False,
    extra_failure_reason: str = "",
) -> int:
    """
    Finalize evaluation CLI: handle baseline, SLOs, and exit code.

    This is a shared helper for eval CLIs to avoid duplication.

    Args:
        primary_score: The primary score to compare against baseline
        slo_checks: List of (name, passed, actual_value, target_value) tuples
        baseline_path: Path to baseline JSON file
        score_key: Key for score in baseline JSON
        set_baseline: Whether to save current results as baseline
        baseline_data: Data to save as baseline (required if set_baseline=True)
        extra_failure_check: Additional failure condition (e.g., paths_failed > 0)
        extra_failure_reason: Reason for extra failure

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Baseline comparison
    is_regression, baseline_score = compare_to_baseline(
        primary_score,
        baseline_path,
        score_key=score_key,
    )
    print_baseline_comparison(primary_score, baseline_score, is_regression)

    # Save as new baseline if requested
    if set_baseline and baseline_data:
        save_baseline(baseline_data, baseline_path)

    # Check SLOs
    failed_slos = get_failed_slos(slo_checks)
    all_slos_passed = len(failed_slos) == 0
    exit_code = determine_exit_code(all_slos_passed, is_regression)

    # Include extra failure condition
    if extra_failure_check:
        exit_code = 1

    # Build failure reasons
    failure_reasons = []
    if extra_failure_check and extra_failure_reason:
        failure_reasons.append(extra_failure_reason)
    if failed_slos:
        failure_reasons.append(f"{len(failed_slos)} SLOs failed: {', '.join(failed_slos)}")
    if is_regression:
        failure_reasons.append("Regression detected vs baseline")

    # Print overall result panel
    console.print()
    print_overall_result_panel(
        all_passed=exit_code == 0,
        failure_reasons=failure_reasons,
        success_message=f"All {len(slo_checks)} SLOs met, no regression detected",
    )

    return exit_code


__all__ = [
    # Formatting
    "console",
    "create_summary_table",
    "format_check_mark",
    "format_percentage",
    "print_eval_header",
    "print_overall_result_panel",
    "print_debug_failures",
    "build_eval_table",
    # Baseline
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    "REGRESSION_THRESHOLD",
    # Parallel
    "run_parallel_evaluation",
    "calculate_p95_latency",
    # SLO
    "create_slo_table",
    "print_slo_result",
    "get_failed_slos",
    "determine_exit_code",
    # Results / LLM Judge
    "parse_json_response",
    "run_llm_judge",
    "save_eval_results",
    "finalize_eval_cli",
]
