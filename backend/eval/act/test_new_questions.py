"""Test only the 2 new questions: Top opportunities and Deals stuck in stage.

Run with: python -m backend.eval.act.test_new_questions
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from backend.act_fetch import DEMO_PROMPTS, act_fetch, get_database
from backend.agent.action.suggester import call_action_chain
from backend.agent.answer.answerer import call_answer_chain
from backend.core.llm import create_openai_chain

# Only test these 2 questions
TEST_QUESTIONS = [
    "Top opportunities",
    "Deals stuck in stage",
]


@dataclass
class QuestionCapture:
    """Captured output for a single question."""

    question: str
    database: str
    timestamp: str

    # Raw data
    fetched_data: dict = field(default_factory=dict)
    fetch_error: str | None = None
    fetch_latency_ms: int = 0

    # Answer
    answer: str = ""
    answer_latency_ms: int = 0

    # Action
    action: str = ""
    action_latency_ms: int = 0

    # Prompts used
    answer_guidance: str = ""
    action_guidance: str = ""


class AnswerScore(BaseModel):
    """Scores for a single answer."""

    usefulness: int = Field(ge=0, le=5, description="Would this help a sales manager? 0=useless, 5=very helpful")
    accuracy: int = Field(ge=0, le=5, description="Does the answer match the fetched data? 0=wrong/hallucinated, 5=perfectly accurate")
    freshness: int = Field(ge=0, le=5, description="Is it about recent/relevant records? 0=stale/outdated, 5=current and timely")
    actionability: int = Field(ge=0, le=5, description="Can you do something concrete with this? 0=no action possible, 5=clear next steps")

    usefulness_reason: str = Field(description="Brief reason for usefulness score")
    accuracy_reason: str = Field(description="Brief reason for accuracy score")
    freshness_reason: str = Field(description="Brief reason for freshness score")
    actionability_reason: str = Field(description="Brief reason for actionability score")

    root_cause: str = Field(description="If total score < 12, what's the root cause? Options: 'wrong_question_for_data', 'bad_stale_data', 'fetch_logic_issue', 'answer_logic_issue', 'acceptable'")
    root_cause_explanation: str = Field(description="Detailed explanation of the root cause")
    recommended_fix: str = Field(description="What would fix this issue?")


_SCORER_SYSTEM_PROMPT = """You are evaluating CRM assistant answers for a sales manager.

Score each dimension 0-5:
- **Usefulness**: Would a sales manager find this helpful for their daily work? 0=useless, 5=very valuable
- **Accuracy**: Does the answer correctly reflect the fetched data? Check for hallucinations or mismatches. 0=wrong, 5=accurate
- **Freshness**: Are the records recent enough to be relevant? 0=stale, 5=current
- **Actionability**: Can they take concrete action based on this? 0=vague, 5=clear next steps

For root_cause, choose ONE of:
- **wrong_question_for_data**: The question asks for something the data doesn't support
- **bad_stale_data**: Data exists but is too old/empty to be useful
- **fetch_logic_issue**: Question is good, data could work, but we're fetching wrong things
- **answer_logic_issue**: Data is fine, but the LLM misinterpreted or hallucinated
- **acceptable**: Total score >= 12 and no major issues

Be critical. A sales manager needs actionable, current insights."""

_SCORER_HUMAN_PROMPT = """## Question
"{question}"

## Fetched Data (from Act! CRM API)
```json
{fetched_data}
```

## Generated Answer
"{answer}"

## Generated Action
"{action}"

Evaluate this answer. Today's date is {today}."""


def capture_question(question: str) -> QuestionCapture:
    """Capture all outputs for a single question."""
    capture = QuestionCapture(
        question=question,
        database=get_database(),
        timestamp=datetime.utcnow().isoformat(),
    )

    # Get prompts
    prompts = DEMO_PROMPTS.get(question, {})
    capture.answer_guidance = prompts.get("answer", "")
    capture.action_guidance = prompts.get("action", "")

    # Step 1: Fetch data
    fetch_start = time.time()
    try:
        result = act_fetch(question)
        capture.fetch_latency_ms = int((time.time() - fetch_start) * 1000)

        if result.get("error"):
            capture.fetch_error = result["error"]
            return capture

        capture.fetched_data = result.get("data", {})

    except Exception as e:
        capture.fetch_latency_ms = int((time.time() - fetch_start) * 1000)
        capture.fetch_error = str(e)
        return capture

    # Step 2: Generate answer
    answer_start = time.time()
    try:
        capture.answer = call_answer_chain(
            question=question,
            sql_results={"data": capture.fetched_data},
            guidance=capture.answer_guidance,
        )
        capture.answer_latency_ms = int((time.time() - answer_start) * 1000)
    except Exception as e:
        capture.answer_latency_ms = int((time.time() - answer_start) * 1000)
        capture.answer = f"[ERROR: {e}]"

    # Step 3: Generate action
    action_start = time.time()
    try:
        capture.action = call_action_chain(
            question=question,
            answer=capture.answer,
            guidance=capture.action_guidance,
        ) or ""
        capture.action_latency_ms = int((time.time() - action_start) * 1000)
    except Exception as e:
        capture.action_latency_ms = int((time.time() - action_start) * 1000)
        capture.action = f"[ERROR: {e}]"

    return capture


def score_answer(capture: QuestionCapture) -> AnswerScore:
    """Score a single answer using GPT-5.2-pro."""

    chain = create_openai_chain(
        system_prompt=_SCORER_SYSTEM_PROMPT,
        human_prompt=_SCORER_HUMAN_PROMPT,
        max_tokens=1024,
        structured_output=AnswerScore,
        streaming=False,
        model="gpt-5.2-pro",
    )

    # Wrap in "data" to match what the LLM actually saw
    # (call_answer_chain receives sql_results={"data": fetched_data})
    data_for_scorer = {"data": capture.fetched_data}
    data_str = json.dumps(data_for_scorer, default=str)
    if len(data_str) > 15000:
        data_str = data_str[:15000] + "\n... [truncated]"

    result: AnswerScore = chain.invoke({
        "question": capture.question,
        "fetched_data": data_str,
        "answer": capture.answer,
        "action": capture.action,
        "today": datetime.utcnow().strftime("%Y-%m-%d"),
    })

    return result


def main() -> None:
    """Run capture and scoring for the 2 new questions."""

    print("=" * 70)
    print(f"Testing New Questions - Database: {get_database()}")
    print("=" * 70)

    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        print("-" * 50)

        # Capture
        print("  Capturing...")
        capture = capture_question(question)

        if capture.fetch_error:
            print(f"  FETCH ERROR: {capture.fetch_error}")
            continue

        data_keys = list(capture.fetched_data.keys())
        print(f"  Data: {data_keys} ({capture.fetch_latency_ms}ms)")
        print(f"  Answer: {capture.answer[:100]}..." if len(capture.answer) > 100 else f"  Answer: {capture.answer}")
        print(f"  Action: {capture.action[:100]}..." if len(capture.action) > 100 else f"  Action: {capture.action}")

        # Score
        print("  Scoring with GPT-5.2-pro...")
        scores = score_answer(capture)

        total = scores.usefulness + scores.accuracy + scores.freshness + scores.actionability
        status = "PASS" if total >= 12 else "FAIL"

        print(f"  [{status}] Total: {total}/20")
        print(f"    Usefulness:    {scores.usefulness}/5 - {scores.usefulness_reason}")
        print(f"    Accuracy:      {scores.accuracy}/5 - {scores.accuracy_reason}")
        print(f"    Freshness:     {scores.freshness}/5 - {scores.freshness_reason}")
        print(f"    Actionability: {scores.actionability}/5 - {scores.actionability_reason}")
        print(f"    Root Cause: {scores.root_cause}")
        print(f"    Fix: {scores.recommended_fix}")

        results.append({
            "question": question,
            "capture": asdict(capture),
            "scores": scores.model_dump(),
            "total_score": total,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        status = "PASS" if r["total_score"] >= 12 else "FAIL"
        print(f"  [{status}] {r['question']}: {r['total_score']}/20")

    avg = sum(r["total_score"] for r in results) / len(results) if results else 0
    print(f"\n  Average: {avg:.1f}/20")

    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"new_questions_{get_database()}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
