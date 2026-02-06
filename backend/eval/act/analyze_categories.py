"""Analyze CRM history to discover optimal email follow-up categories.

Uses GPT-5.2-pro to analyze actual history text and suggest categories.

Run with: python -m backend.eval.act.analyze_categories
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import httpx  # noqa: E402

# Monkey-patch act_fetch timeout BEFORE importing _get
import backend.act_fetch as act_fetch  # noqa: E402

act_fetch.TIMEOUT = httpx.Timeout(120.0, connect=30.0)  # Increase from 30s to 120s

from backend.core.llm import create_openai_chain  # noqa: E402


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&nbsp;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


def fetch_and_cache_history() -> list[dict]:
    """Fetch 1000 history records and cache to file."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    cache_file = output_dir / "history_cache.json"

    # Check if recent cache exists (less than 1 hour old)
    if cache_file.exists():
        age_seconds = (datetime.now().timestamp() - os.path.getmtime(cache_file))
        if age_seconds < 3600:  # 1 hour
            print(f"Using cached history (age: {age_seconds/60:.1f} minutes)")
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)

    # Fetch from API using act_fetch (handles auth)
    print("Fetching 1000 history records from Act! API...")
    print("Timeout: 120 seconds (please wait...)")

    history = act_fetch._get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})
    print(f"Fetched {len(history)} records")

    # Cache to file
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Cached to {cache_file}")

    return history


def analyze_history_categories() -> None:
    """Fetch history and ask GPT-5.2-pro to analyze categories."""

    print("=" * 70)
    print("CRM History Category Analysis")
    print("Using: GPT-5.2-pro")
    print("=" * 70)

    # Fetch/load history
    try:
        history = fetch_and_cache_history()
    except Exception as e:
        print(f"ERROR: Failed to fetch history: {e}")
        return

    print(f"\nTotal records: {len(history)}")

    # Extract text samples with content
    samples = []
    for h in history:
        details = strip_html(h.get("details") or "")
        regarding = h.get("regarding") or ""
        if details or regarding:
            samples.append({
                "regarding": regarding[:200],
                "details": details[:300],  # Shorter to fit more samples
                "date": str(h.get("startTime") or "")[:10],
            })

    print(f"Records with text content: {len(samples)}")

    if not samples:
        print("ERROR: No history records with text found")
        return

    # Take representative sample (100 records - reduced for faster processing)
    samples_for_llm = samples[:100]
    print(f"Sending {len(samples_for_llm)} samples to GPT-5.2-pro")

    # Build analysis prompt
    analysis_prompt = """Analyze these {count} CRM interaction history records and identify the BEST categories for email follow-up.

Goal: Help a sales manager quickly find contacts who need follow-up emails.

Current categories we're using:
1. quotes - Pending quotes/proposals
2. support - Unresolved support issues
3. renewals - Upcoming renewals/expirations
4. recent - Recent interactions (questionable - recent != needs follow-up)
5. technical - Technical issues (sync, database, server)

History records to analyze:
{history_json}

Based on these ACTUAL records, answer:

1. **What patterns do you see?** What types of interactions appear most frequently? Give counts.

2. **Which of our current 5 categories have good coverage?** (many records match)

3. **Which categories have poor coverage?** (few or no records match)

4. **What categories are MISSING?** Common patterns that don't fit our current categories?

5. **Recommended 5 categories:** Based on this data, what are the 5 BEST categories? For each:
   - id: short lowercase identifier
   - label: Question format ("Who has...?", "Who needs...?")
   - description: What contacts this finds
   - estimated_matches: approximate count from this dataset

Be specific and data-driven. Reference actual patterns you see."""

    print("\nAnalyzing with GPT-5.2-pro...")
    print("-" * 70)

    chain = create_openai_chain(
        system_prompt="You are a CRM data analyst. Analyze interaction history to identify optimal email follow-up categories. Be specific and data-driven.",
        human_prompt=analysis_prompt,
        model="gpt-5.2-pro",  # Back to pro
        max_tokens=3000,
        streaming=False,
        timeout=180,  # 3 minute timeout for pro model
    )

    try:
        result = chain.invoke({
            "count": len(samples_for_llm),
            "history_json": json.dumps(samples_for_llm, indent=2),
        })
        print(result)
    except Exception as e:
        print(f"ERROR: LLM analysis failed: {e}")
        return

    print("\n" + "=" * 70)
    print("Analysis complete")
    print("=" * 70)

    # Save result
    output_dir = Path(__file__).parent / "output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"category_analysis_{timestamp}.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\nSaved analysis to: {result_file}")


if __name__ == "__main__":
    analyze_history_categories()
