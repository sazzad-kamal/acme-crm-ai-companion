"""Analyze content patterns in KQC to design data-driven questions."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.act_fetch import _get


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&nbsp;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


def analyze_content_patterns() -> None:
    """Categorize history content by type to find actionable patterns."""
    print("Fetching history data...")
    history = _get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})

    print(f"Total history items: {len(history)}")

    # Categorize by 'regarding' field (subject line)
    regarding_counter: Counter = Counter()
    for h in history:
        regarding = h.get("regarding", "")[:50] if h.get("regarding") else "(empty)"
        regarding_counter[regarding] += 1

    print("\n" + "=" * 70)
    print("TOP 'REGARDING' SUBJECTS (what types of activities exist)")
    print("=" * 70)
    for subject, count in regarding_counter.most_common(30):
        print(f"  {count:>4}x  {subject}")

    # Categorize by content patterns
    print("\n" + "=" * 70)
    print("CONTENT PATTERN ANALYSIS")
    print("=" * 70)

    patterns = {
        "Quote/Proposal": [r"quote", r"proposal", r"pricing", r"estimate"],
        "Support/Help": [r"support", r"help", r"issue", r"problem", r"error", r"fix"],
        "Renewal": [r"renew", r"expir", r"subscript"],
        "Call/Phone": [r"called", r"spoke", r"phone", r"voicemail", r"left message"],
        "Email": [r"email", r"sent.*email", r"replied"],
        "Meeting": [r"meeting", r"demo", r"appointment", r"scheduled"],
        "Invoice/Payment": [r"invoice", r"payment", r"paid", r"charge"],
        "Technical": [r"sync", r"database", r"server", r"install", r"upgrade"],
        "Field Changed": [r"field changed", r"status.*changed"],
        "Quote Viewed": [r"quote viewed", r"viewed your quote"],
    }

    pattern_counts: dict[str, list] = {k: [] for k in patterns}

    for h in history:
        details = strip_html(h.get("details", "")).lower()
        regarding = (h.get("regarding") or "").lower()
        combined = f"{regarding} {details}"

        for category, pats in patterns.items():
            if any(re.search(p, combined) for p in pats):
                pattern_counts[category].append(h)

    # Sort by count
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -len(x[1]))

    for category, items in sorted_patterns:
        pct = 100 * len(items) / len(history) if history else 0
        print(f"\n{category}: {len(items)} ({pct:.0f}%)")
        # Show samples
        for h in items[:3]:
            regarding = h.get("regarding", "")[:40]
            details = strip_html(h.get("details", ""))[:100]
            contacts = [c.get("displayName") for c in (h.get("contacts") or []) if isinstance(c, dict)]
            contact = contacts[0] if contacts else "No contact"
            print(f"    [{regarding}] ({contact})")
            print(f"      {details}...")

    # Find actionable history (not just system/field changes)
    print("\n" + "=" * 70)
    print("ACTIONABLE HISTORY (excluding system records)")
    print("=" * 70)

    system_patterns = [r"field changed", r"record created", r"record updated", r"quote viewed"]
    actionable = []
    for h in history:
        details = strip_html(h.get("details", "")).lower()
        regarding = (h.get("regarding") or "").lower()
        combined = f"{regarding} {details}"

        if not any(re.search(p, combined) for p in system_patterns):  # noqa: SIM102
            if len(details) > 50:  # Has meaningful content
                actionable.append(h)

    print(f"\nActionable records: {len(actionable)} / {len(history)} ({100*len(actionable)/len(history):.0f}%)")

    # Sample actionable
    print("\nSample actionable history:")
    for h in actionable[:10]:
        regarding = h.get("regarding", "")[:50]
        details = strip_html(h.get("details", ""))[:150]
        contacts = [c.get("displayName") for c in (h.get("contacts") or []) if isinstance(c, dict)]
        contact = contacts[0] if contacts else "No contact"
        print(f"\n  [{regarding}] - {contact}")
        print(f"    {details}...")

    # Suggest questions based on data
    print("\n" + "=" * 70)
    print("SUGGESTED QUESTIONS BASED ON DATA")
    print("=" * 70)

    suggestions = []

    if len(pattern_counts["Quote/Proposal"]) > 20:
        suggestions.append(("Who has open quotes?", len(pattern_counts["Quote/Proposal"]), "Quote follow-ups"))

    if len(pattern_counts["Support/Help"]) > 20:
        suggestions.append(("Who needs support follow-up?", len(pattern_counts["Support/Help"]), "Support tickets"))

    if len(pattern_counts["Renewal"]) > 10:
        suggestions.append(("Who should I call about renewals?", len(pattern_counts["Renewal"]), "Renewal records"))

    if len(pattern_counts["Call/Phone"]) > 20:
        suggestions.append(("Who did I talk to recently?", len(pattern_counts["Call/Phone"]), "Call records"))

    if len(pattern_counts["Technical"]) > 20:
        suggestions.append(("Who has technical issues?", len(pattern_counts["Technical"]), "Tech support"))

    for q, count, source in suggestions:
        print(f"\n  \"{q}\"")
        print(f"    Data: {count} records from {source}")


if __name__ == "__main__":
    analyze_content_patterns()
