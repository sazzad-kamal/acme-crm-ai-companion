"""Analyze CRM history in batches using LLM.

Splits 1000 records into batches of 100 for analysis.

Run with: python -m backend.eval.act.analyze_categories_batch
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.core.llm import create_openai_chain  # noqa: E402


def strip_html(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&nbsp;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


BATCH_PROMPT = """Categorize these {count} CRM history records into follow-up categories.

Categories:
1. awaiting_response - Waiting for response, sent email, no answer, follow-up needed
2. support - Support issues, problems, errors, help requests
3. renewals - Renewals, expirations, subscriptions, licenses
4. billing - Invoices, payments, charges, refunds
5. quotes - Quotes, proposals, pricing, estimates

Records:
{history_json}

For EACH record, output ONE line in format:
record_index|category|reason

Example:
0|support|Customer reported sync error
1|quotes|Quote 12345 was viewed
2|awaiting_response|Left message, no callback

Only categorize records that clearly match. Skip unclear ones."""


def analyze_in_batches() -> None:
    print("=" * 70)
    print("Batch Category Analysis (LLM)")
    print("=" * 70)

    # Load cached history
    cache_file = Path(__file__).parent / "output" / "history_cache.json"
    if not cache_file.exists():
        print("ERROR: No cached history. Run analyze_categories.py first.")
        return

    with open(cache_file, encoding="utf-8") as f:
        history = json.load(f)

    print(f"Total records: {len(history)}")

    # Extract samples
    samples = []
    for h in history:
        details = strip_html(h.get("details") or "")
        regarding = h.get("regarding") or ""
        if details or regarding:
            samples.append({
                "regarding": regarding[:150],
                "details": details[:200],
            })

    print(f"Records with text: {len(samples)}")

    # Process in batches
    batch_size = 100
    all_results = {
        "awaiting_response": [],
        "support": [],
        "renewals": [],
        "billing": [],
        "quotes": [],
    }

    chain = create_openai_chain(
        system_prompt="You are a CRM analyst categorizing interaction records. Be concise.",
        human_prompt=BATCH_PROMPT,
        model="gpt-5.2",  # Cheaper model for batches
        max_tokens=2000,
        streaming=False,
        timeout=120,
    )

    for batch_num in range(0, len(samples), batch_size):
        batch = samples[batch_num:batch_num + batch_size]
        batch_end = min(batch_num + batch_size, len(samples))

        print(f"\nBatch {batch_num//batch_size + 1}: records {batch_num}-{batch_end-1}")

        try:
            result = chain.invoke({
                "count": len(batch),
                "history_json": json.dumps(batch, indent=1),
            })

            # Parse results
            for line in result.strip().split("\n"):
                line = line.strip()
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        category = parts[1].strip().lower()
                        if category in all_results:
                            all_results[category].append(line)

            # Show progress
            for cat, items in all_results.items():
                print(f"  {cat}: {len(items)}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL COUNTS (LLM categorized)")
    print("=" * 70)

    total = 0
    for cat, items in sorted(all_results.items(), key=lambda x: -len(x[1])):
        count = len(items)
        total += count
        pct = (count / len(samples)) * 100
        print(f"{cat}: {count} ({pct:.1f}%)")
        # Show samples
        for item in items[:3]:
            print(f"  - {item}")

    print(f"\nTotal categorized: {total}/{len(samples)} ({total/len(samples)*100:.1f}%)")

    # Save results
    output_file = Path(__file__).parent / "output" / f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    analyze_in_batches()
