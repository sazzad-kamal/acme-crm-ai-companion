"""Analyze CRM history in batches using GPT-5.2-pro.

Splits 1000 records into batches of 50 for comprehensive analysis.
Goal: Find any additional categories we might have missed.

Run with: python -m backend.eval.act.analyze_categories_pro_batch
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


BATCH_PROMPT = """Analyze these {count} CRM history records and categorize them for email follow-up.

Current 5 categories:
1. awaiting_response - Waiting for response, sent email, no answer, follow-up needed
2. support - Support issues, problems, errors, help requests
3. renewals - Renewals, expirations, subscriptions, licenses
4. billing - Invoices, payments, charges, refunds
5. quotes - Quotes, proposals, pricing, estimates

Records:
{history_json}

For EACH record, output ONE line in format:
record_index|category|reason

Use one of the 5 categories above. If a record doesn't fit ANY category, use "other" and explain what category it SHOULD be.

Example:
0|support|Customer reported sync error
1|quotes|Quote 12345 was viewed
2|awaiting_response|Left message, no callback
3|other|Training request - suggest adding "training" category

Be thorough - categorize every record that has meaningful content."""


def analyze_in_batches_pro() -> None:
    print("=" * 70)
    print("Batch Category Analysis (GPT-5.2-pro)")
    print("=" * 70)

    # Load cached history
    cache_file = Path(__file__).parent / "output" / "history_cache.json"
    if not cache_file.exists():
        print("ERROR: No cached history. Run analyze_categories.py first.")
        return

    with open(cache_file, encoding="utf-8") as f:
        history = json.load(f)

    print(f"Total records: {len(history)}")

    # Extract samples with content
    samples = []
    for idx, h in enumerate(history):
        details = strip_html(h.get("details") or "")
        regarding = h.get("regarding") or ""
        if details or regarding:
            samples.append({
                "idx": idx,
                "regarding": regarding[:150],
                "details": details[:200],
            })

    print(f"Records with text: {len(samples)}")

    # Process in batches of 50
    batch_size = 50
    all_results = {
        "awaiting_response": [],
        "support": [],
        "renewals": [],
        "billing": [],
        "quotes": [],
        "other": [],  # Track records that don't fit
    }

    # Track suggested new categories
    suggested_categories: dict[str, int] = {}

    chain = create_openai_chain(
        system_prompt="You are a CRM analyst categorizing interaction records. Be thorough and categorize every record.",
        human_prompt=BATCH_PROMPT,
        model="gpt-5.2-pro",
        max_tokens=3000,
        streaming=False,
        timeout=180,  # 3 min timeout for pro
    )

    total_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_num in range(0, len(samples), batch_size):
        batch = samples[batch_num:batch_num + batch_size]
        batch_end = min(batch_num + batch_size, len(samples))
        current_batch = batch_num // batch_size + 1

        print(f"\nBatch {current_batch}/{total_batches}: records {batch_num}-{batch_end-1}")

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
                        reason = parts[2].strip() if len(parts) > 2 else ""

                        if category in all_results:
                            all_results[category].append({"line": line, "reason": reason})
                        else:
                            # Track as "other" and extract suggested category
                            all_results["other"].append({"line": line, "reason": reason, "suggested": category})
                            # Count suggested categories
                            if "suggest" in reason.lower() or category not in ["awaiting_response", "support", "renewals", "billing", "quotes"]:
                                # Extract category name from reason if mentioned
                                cat_match = re.search(r'suggest.*?"(\w+)"', reason.lower())
                                if cat_match:
                                    suggested_categories[cat_match.group(1)] = suggested_categories.get(cat_match.group(1), 0) + 1
                                else:
                                    suggested_categories[category] = suggested_categories.get(category, 0) + 1

            # Show progress
            print("  Current totals:")
            for cat, items in all_results.items():
                if items:
                    print(f"    {cat}: {len(items)}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL COUNTS (GPT-5.2-pro categorized)")
    print("=" * 70)

    total = 0
    for cat, items in sorted(all_results.items(), key=lambda x: -len(x[1])):
        if not items:
            continue
        count = len(items)
        total += count
        pct = (count / len(samples)) * 100
        print(f"\n{cat.upper()}: {count} ({pct:.1f}%)")
        # Show samples
        for item in items[:5]:
            print(f"  - {item['line'][:80]}")

    print(f"\n{'=' * 70}")
    print(f"Total categorized: {total}/{len(samples)} ({total/len(samples)*100:.1f}%)")

    # Show suggested new categories
    if suggested_categories:
        print(f"\n{'=' * 70}")
        print("SUGGESTED NEW CATEGORIES (from 'other')")
        print("=" * 70)
        for cat, count in sorted(suggested_categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    # Save results
    output_dir = Path(__file__).parent / "output"
    output_file = output_dir / f"pro_batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        "categories": {k: len(v) for k, v in all_results.items()},
        "suggested_new": suggested_categories,
        "details": dict(all_results.items()),
        "total_records": len(samples),
        "total_categorized": total,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    analyze_in_batches_pro()
