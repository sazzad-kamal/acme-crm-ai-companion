"""Analyze unstructured/text data distribution across all Act! API entities."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.act_fetch import _get


def _is_meaningful_text(text: str) -> bool:
    """Check if text is meaningful (not just boilerplate/empty)."""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 5:
        return False
    # Filter out common boilerplate
    boilerplate = ["n/a", "none", "null", "-", ".", "...", "na", "tbd"]
    return text.lower() not in boilerplate


def _analyze_text_field(records: list[dict], field_name: str, entity: str) -> dict:
    """Analyze a text field across records."""
    values = []
    for r in records:
        val = r.get(field_name)
        if _is_meaningful_text(val):
            values.append(val)

    if not values:
        return None

    lengths = [len(v) for v in values]
    return {
        "entity": entity,
        "field": field_name,
        "count": len(values),
        "total_records": len(records),
        "pct_populated": round(100 * len(values) / len(records), 1) if records else 0,
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "max_length": max(lengths),
        "samples": values[:3],  # First 3 samples
    }


def _analyze_custom_fields(records: list[dict], entity: str) -> list[dict]:
    """Analyze custom fields for text content."""
    results = []

    # Collect all custom field keys
    all_keys = set()
    for r in records:
        cf = r.get("customFields") or {}
        all_keys.update(cf.keys())

    # Check each custom field for text content
    for key in sorted(all_keys):
        values = []
        for r in records:
            cf = r.get("customFields") or {}
            val = cf.get(key)
            if _is_meaningful_text(val):
                values.append(val)

        if values:
            lengths = [len(v) for v in values]
            results.append({
                "entity": entity,
                "field": f"customFields.{key}",
                "count": len(values),
                "total_records": len(records),
                "pct_populated": round(100 * len(values) / len(records), 1) if records else 0,
                "avg_length": round(sum(lengths) / len(lengths), 1),
                "max_length": max(lengths),
                "samples": values[:2],
            })

    return results


def analyze_all_entities() -> None:
    """Analyze text fields across all Act! API entities."""
    print("=" * 70)
    print("UNSTRUCTURED DATA ANALYSIS - Act! API (KQC Database)")
    print("=" * 70)

    all_results = []

    # 1. Contacts
    print("\n[1/7] Fetching contacts...")
    try:
        contacts = _get("/api/contacts", {"$top": 500})
        print(f"      Got {len(contacts)} contacts")
        # No direct text fields on contacts, but check custom fields
        cf_results = _analyze_custom_fields(contacts, "contacts")
        all_results.extend(cf_results)
        if cf_results:
            print(f"      Found {len(cf_results)} custom fields with text")
    except Exception as e:
        print(f"      ERROR: {e}")

    # 2. Opportunities
    print("\n[2/7] Fetching opportunities...")
    try:
        opps = _get("/api/opportunities", {"$top": 500})
        print(f"      Got {len(opps)} opportunities")
        # Check standard text fields
        for field in ["name", "product", "reason", "competitor"]:
            result = _analyze_text_field(opps, field, "opportunities")
            if result:
                all_results.append(result)
        # Check custom fields
        cf_results = _analyze_custom_fields(opps, "opportunities")
        all_results.extend(cf_results)
        if cf_results:
            print(f"      Found {len(cf_results)} custom fields with text")
    except Exception as e:
        print(f"      ERROR: {e}")

    # 3. History
    print("\n[3/7] Fetching history...")
    try:
        history = _get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})
        print(f"      Got {len(history)} history items")
        for field in ["details", "regarding"]:
            result = _analyze_text_field(history, field, "history")
            if result:
                all_results.append(result)
    except Exception as e:
        print(f"      ERROR: {e}")

    # 4. Companies
    print("\n[4/7] Fetching companies...")
    try:
        companies = _get("/api/companies", {"$top": 500})
        print(f"      Got {len(companies)} companies")
        for field in ["description", "industry", "territory", "region"]:
            result = _analyze_text_field(companies, field, "companies")
            if result:
                all_results.append(result)
        cf_results = _analyze_custom_fields(companies, "companies")
        all_results.extend(cf_results)
        if cf_results:
            print(f"      Found {len(cf_results)} custom fields with text")
    except Exception as e:
        print(f"      ERROR: {e}")

    # 5. Groups
    print("\n[5/7] Fetching groups...")
    try:
        groups = _get("/api/groups", {"$top": 200})
        print(f"      Got {len(groups)} groups")
        for field in ["description", "name"]:
            result = _analyze_text_field(groups, field, "groups")
            if result:
                all_results.append(result)
        cf_results = _analyze_custom_fields(groups, "groups")
        all_results.extend(cf_results)
    except Exception as e:
        print(f"      ERROR: {e}")

    # 6. Notes (separate endpoint)
    print("\n[6/7] Fetching notes...")
    try:
        notes = _get("/api/notes", {"$top": 1000})
        print(f"      Got {len(notes)} notes")
        result = _analyze_text_field(notes, "noteText", "notes")
        if result:
            all_results.append(result)
    except Exception as e:
        print(f"      ERROR: {e}")

    # 7. Calendar/Activities
    print("\n[7/7] Fetching calendar...")
    try:
        import time
        today = time.strftime("%Y-%m-%d")
        # Fetch last 30 days
        start = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30*24*60*60))
        calendar = _get("/api/calendar", {"startDate": start, "endDate": today})
        # Extract items from calendar
        items = []
        for day in calendar:
            items.extend(day.get("items") or [])
        print(f"      Got {len(items)} calendar items")
        for field in ["details", "regarding", "subject"]:
            result = _analyze_text_field(items, field, "calendar")
            if result:
                all_results.append(result)
    except Exception as e:
        print(f"      ERROR: {e}")

    # Try activities endpoint (may timeout)
    print("\n[BONUS] Trying activities endpoint...")
    try:
        activities = _get("/api/activities", {"$top": 100})
        print(f"      Got {len(activities)} activities")
        for field in ["details", "regarding", "subject"]:
            result = _analyze_text_field(activities, field, "activities")
            if result:
                all_results.append(result)
    except Exception as e:
        print(f"      Skipped (expected): {type(e).__name__}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Text Fields Found")
    print("=" * 70)

    if not all_results:
        print("\nNo meaningful text fields found!")
        return

    # Sort by count descending
    all_results.sort(key=lambda x: -x["count"])

    # Group by entity
    by_entity = defaultdict(list)
    for r in all_results:
        by_entity[r["entity"]].append(r)

    total_text_records = 0
    for entity in ["history", "notes", "opportunities", "contacts", "companies", "groups", "calendar", "activities"]:
        if entity not in by_entity:
            continue
        print(f"\n### {entity.upper()}")
        for r in by_entity[entity]:
            print(f"  {r['field']}: {r['count']}/{r['total_records']} ({r['pct_populated']}%) avg {r['avg_length']} chars")
            total_text_records += r["count"]
            # Show samples (truncated)
            for i, s in enumerate(r["samples"][:2]):
                sample = s[:100] + "..." if len(s) > 100 else s
                sample = sample.replace("\n", " ").replace("\r", "")
                print(f"      Sample {i+1}: \"{sample}\"")

    print("\n" + "=" * 70)
    print(f"TOTAL: {len(all_results)} text fields with {total_text_records} populated records")
    print("=" * 70)

    # Actionability assessment
    print("\n### ACTIONABILITY ASSESSMENT")
    high_value = [r for r in all_results if r["count"] >= 50 and r["avg_length"] >= 20]
    if high_value:
        print("\nHigh-value fields (50+ records, 20+ avg chars):")
        for r in high_value:
            print(f"  - {r['entity']}.{r['field']}: {r['count']} records, {r['avg_length']} avg chars")
    else:
        print("\nNo high-value text fields found (need 50+ records with 20+ avg chars)")


if __name__ == "__main__":
    analyze_all_entities()
