"""Allow running as: python -m backend.eval.act"""

import argparse
import sys

from backend.eval.act.runner import (
    SLO_AVG_FAITHFULNESS,
    SLO_AVG_LATENCY_MS,
    SLO_AVG_RELEVANCY,
    run_act_eval,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Act! demo evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers")
    parser.add_argument("--question", "-q", type=int, help="Run single question by index (0-4)")
    args = parser.parse_args()

    summary = run_act_eval(verbose=args.verbose, question_index=args.question)

    # Exit with appropriate code
    slo_passed = (
        summary.all_passed
        and summary.avg_faithfulness >= SLO_AVG_FAITHFULNESS
        and summary.avg_relevancy >= SLO_AVG_RELEVANCY
        and summary.action_pass_rate == 1.0
        and summary.avg_latency_ms < SLO_AVG_LATENCY_MS
    )
    sys.exit(0 if slo_passed else 1)
