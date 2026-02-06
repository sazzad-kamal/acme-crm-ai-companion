"""Analyze CRM history locally to discover patterns (no LLM needed).

Run with: python -m backend.eval.act.analyze_categories_local
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&nbsp;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


# Category patterns to search for
CATEGORY_PATTERNS = {
    "awaiting_response": [
        r"waiting.*response", r"waiting.*reply", r"sent.*email",
        r"no answer", r"left message", r"follow.?up", r"pending",
        r"awaiting", r"get back", r"let me know"
    ],
    "quotes": [
        r"quote", r"proposal", r"pricing", r"price", r"cost",
        r"estimate", r"bid", r"offer", r"discount"
    ],
    "support": [
        r"support", r"help", r"issue", r"problem", r"error",
        r"bug", r"fix", r"resolve", r"ticket", r"case"
    ],
    "renewals": [
        r"renewal", r"renew", r"expir", r"subscription", r"annual",
        r"upgrade", r"license", r"maintenance"
    ],
    "technical": [
        r"sync", r"database", r"server", r"install", r"upgrade",
        r"backup", r"restore", r"crash", r"connect", r"login",
        r"password", r"user", r"permission", r"migration"
    ],
    "training": [
        r"train", r"learn", r"webinar", r"demo", r"tutorial",
        r"onboard", r"setup", r"getting started"
    ],
    "billing": [
        r"invoice", r"payment", r"bill", r"charge", r"credit",
        r"refund", r"account", r"paid"
    ],
    "opportunity": [
        r"opportunity", r"deal", r"prospect", r"lead", r"sales",
        r"pipeline", r"close", r"won", r"lost"
    ],
    "meeting": [
        r"meeting", r"call", r"schedule", r"appointment", r"phone",
        r"zoom", r"teams", r"conference"
    ],
}


def analyze_history() -> None:
    """Analyze cached history for patterns."""

    print("=" * 70)
    print("CRM History Pattern Analysis (Local)")
    print("=" * 70)

    # Load cached history
    cache_file = Path(__file__).parent / "output" / "history_cache.json"
    if not cache_file.exists():
        print("ERROR: No cached history. Run analyze_categories.py first.")
        return

    with open(cache_file, encoding="utf-8") as f:
        history = json.load(f)

    print(f"\nTotal records: {len(history)}")

    # Extract text and count patterns
    category_matches: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_PATTERNS}
    unmatched = []
    all_keywords = Counter()

    for h in history:
        details = strip_html(h.get("details") or "").lower()
        regarding = (h.get("regarding") or "").lower()
        full_text = f"{regarding} {details}"

        if not full_text.strip():
            continue

        # Extract keywords from text
        words = re.findall(r"\b[a-z]{4,}\b", full_text)
        all_keywords.update(words)

        # Check which categories match
        matched_any = False
        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    category_matches[category].append({
                        "regarding": (h.get("regarding") or "")[:80],
                        "details": strip_html(h.get("details") or "")[:100],
                    })
                    matched_any = True
                    break  # Only count once per category

        if not matched_any:
            unmatched.append({
                "regarding": (h.get("regarding") or "")[:80],
                "details": strip_html(h.get("details") or "")[:100],
            })

    # Print results
    print("\n" + "=" * 70)
    print("CATEGORY PATTERN MATCHES")
    print("=" * 70)

    for category, matches in sorted(category_matches.items(), key=lambda x: -len(x[1])):
        count = len(matches)
        pct = (count / len(history)) * 100
        status = "HIGH" if count > 50 else "MEDIUM" if count > 20 else "LOW"
        print(f"\n{category.upper()}: {count} matches ({pct:.1f}%) [{status}]")
        # Show sample matches
        for m in matches[:3]:
            print(f"  - {m['regarding'][:60]}")

    print("\n" + "=" * 70)
    print(f"UNMATCHED: {len(unmatched)} records ({len(unmatched)/len(history)*100:.1f}%)")
    print("=" * 70)
    for m in unmatched[:10]:
        print(f"  - {m['regarding'][:60] or m['details'][:60]}")

    print("\n" + "=" * 70)
    print("TOP 30 KEYWORDS (may suggest new categories)")
    print("=" * 70)
    for word, count in all_keywords.most_common(30):
        print(f"  {word}: {count}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    high_coverage = [c for c, m in category_matches.items() if len(m) > 50]
    medium_coverage = [c for c, m in category_matches.items() if 20 < len(m) <= 50]
    low_coverage = [c for c, m in category_matches.items() if len(m) <= 20]

    print(f"\nHIGH coverage categories (>50 matches): {', '.join(high_coverage) or 'None'}")
    print(f"MEDIUM coverage categories (20-50): {', '.join(medium_coverage) or 'None'}")
    print(f"LOW coverage categories (<20): {', '.join(low_coverage) or 'None'}")

    # Current 5 categories assessment
    current_cats = ["quotes", "support", "renewals", "technical", "recent"]
    print("\n\nCURRENT 5 CATEGORIES ASSESSMENT:")
    for cat in current_cats:
        if cat == "recent":
            print(f"  {cat}: N/A (time-based, not pattern-based)")
        else:
            count = len(category_matches.get(cat, []))
            status = "KEEP" if count > 20 else "CONSIDER REPLACING"
            print(f"  {cat}: {count} matches - {status}")


if __name__ == "__main__":
    analyze_history()
