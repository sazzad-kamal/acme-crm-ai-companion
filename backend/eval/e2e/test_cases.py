"""
E2E evaluation test cases loader.

Loads test cases from YAML file for cleaner separation of data and code.
"""

from pathlib import Path

import yaml  # type: ignore[import-untyped]


def _load_test_cases() -> list[dict]:
    """Load test cases from YAML file and apply post-processing."""
    yaml_path = Path(__file__).parent / "test_cases.yaml"

    with open(yaml_path, encoding="utf-8") as f:
        cases = yaml.safe_load(f)

    # Post-process special cases
    for case in cases:
        # Handle the very long input test case
        if case["question"] == "__LONG_INPUT__":
            case["question"] = (
                "What is the status of Acme Manufacturing? " + "Please provide details. " * 100
            )

    return cases  # type: ignore[no-any-return]


E2E_TEST_CASES = _load_test_cases()
