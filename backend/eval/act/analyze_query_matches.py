"""Validate Gold Standard questions return matches with linked notes/history."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.act_fetch import _filter_test_records, _get


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:500]  # Truncate for display


def analyze_question_matches() -> None:
    """Analyze each Gold Standard question for matches and linked context."""
    print("=" * 70)
    print("GOLD STANDARD QUESTIONS - Match & Context Analysis")
    print("=" * 70)

    today = time.strftime("%Y-%m-%d")

    # Fetch all data once
    print("\nFetching data...")
    opps = _get("/api/opportunities", {"$top": 500})
    opps = _filter_test_records(opps)
    open_opps = [o for o in opps if (o.get("statusName") or "").lower() not in ["closed", "won", "lost"]]
    print(f"  Opportunities: {len(opps)} total, {len(open_opps)} open")

    contacts = _get("/api/contacts", {"$top": 500})
    print(f"  Contacts: {len(contacts)}")

    companies = _get("/api/companies", {"$top": 500})
    companies = _filter_test_records(companies)
    print(f"  Companies: {len(companies)}")

    history = _get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})
    print(f"  History: {len(history)}")

    notes = _get("/api/notes", {"$top": 1000})
    print(f"  Notes: {len(notes)}")

    # Build lookup maps
    # Notes by opportunity ID
    notes_by_opp: dict[str, list] = {}
    notes_by_contact: dict[str, list] = {}
    notes_by_company: dict[str, list] = {}
    for n in notes:
        for o in n.get("opportunities") or []:
            oid = o.get("id") if isinstance(o, dict) else o
            if oid:
                notes_by_opp.setdefault(str(oid), []).append(n)
        for c in n.get("contacts") or []:
            cid = c.get("id") if isinstance(c, dict) else c
            if cid:
                notes_by_contact.setdefault(str(cid), []).append(n)
        for co in n.get("companies") or []:
            coid = co.get("id") if isinstance(co, dict) else co
            if coid:
                notes_by_company.setdefault(str(coid), []).append(n)

    # History by opportunity/contact/company
    history_by_opp: dict[str, list] = {}
    history_by_contact: dict[str, list] = {}
    history_by_company: dict[str, list] = {}
    for h in history:
        for o in h.get("opportunities") or []:
            oid = o.get("id") if isinstance(o, dict) else o
            if oid:
                history_by_opp.setdefault(str(oid), []).append(h)
        for c in h.get("contacts") or []:
            cid = c.get("id") if isinstance(c, dict) else c
            if cid:
                history_by_contact.setdefault(str(cid), []).append(h)
        for co in h.get("companies") or []:
            coid = co.get("id") if isinstance(co, dict) else co
            if coid:
                history_by_company.setdefault(str(coid), []).append(h)

    contact_map = {c.get("id"): c for c in contacts if c.get("id")}

    # ============================================================
    # QUESTION 1: At-risk deals
    # ============================================================
    print("\n" + "=" * 70)
    print("Q1: AT-RISK DEALS")
    print("=" * 70)

    at_risk = []
    for opp in open_opps:
        dis = opp.get("daysInStage") or 0
        close = str(opp.get("estimatedCloseDate") or "")[:10]
        is_overdue = close and close < today
        is_stalled = dis > 14

        if is_overdue or is_stalled:
            at_risk.append({
                "opp": opp,
                "reason": "Overdue" if is_overdue else "Stalled",
                "days_in_stage": dis,
            })

    print(f"\nMatches: {len(at_risk)} at-risk deals")

    # Check context availability
    with_notes = sum(1 for ar in at_risk if str(ar["opp"].get("id")) in notes_by_opp)
    with_history = sum(1 for ar in at_risk if str(ar["opp"].get("id")) in history_by_opp)
    print(f"With notes: {with_notes}/{len(at_risk)} ({100*with_notes/len(at_risk):.0f}%)" if at_risk else "")
    print(f"With history: {with_history}/{len(at_risk)} ({100*with_history/len(at_risk):.0f}%)" if at_risk else "")

    # Show samples with context
    print("\nSample matches with context:")
    shown = 0
    for ar in at_risk[:10]:
        opp = ar["opp"]
        oid = str(opp.get("id"))
        opp_notes = notes_by_opp.get(oid, [])
        opp_history = history_by_opp.get(oid, [])

        if opp_notes or opp_history:
            shown += 1
            print(f"\n  [{shown}] {opp.get('name')} - ${opp.get('productTotal', 0):,.0f} ({ar['reason']}, {ar['days_in_stage']} days)")

            # Get contact name
            for oc in opp.get("contacts") or []:
                cid = oc.get("id") if isinstance(oc, dict) else None
                if cid and cid in contact_map:
                    print(f"      Contact: {contact_map[cid].get('fullName')}")
                    break

            # Show note sample
            if opp_notes:
                note_text = opp_notes[0].get("noteText", "")[:200]
                print(f"      Note: \"{note_text}...\"")

            # Show history sample
            if opp_history:
                h = opp_history[0]
                details = strip_html(h.get("details", ""))[:150]
                print(f"      History: [{h.get('regarding')}] {details}...")

            if shown >= 3:
                break

    # ============================================================
    # QUESTION 2: Forecast health
    # ============================================================
    print("\n" + "=" * 70)
    print("Q2: FORECAST HEALTH")
    print("=" * 70)

    d30 = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 30*24*60*60))
    d60 = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 60*24*60*60))
    d90 = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 90*24*60*60))

    forecast = {"30d": [], "60d": [], "90d": [], "overdue": []}
    for opp in open_opps:
        close = str(opp.get("estimatedCloseDate") or "")[:10]
        if not close:
            continue
        if close < today:
            forecast["overdue"].append(opp)
        elif close <= d30:
            forecast["30d"].append(opp)
        elif close <= d60:
            forecast["60d"].append(opp)
        elif close <= d90:
            forecast["90d"].append(opp)

    print("\nMatches:")
    print(f"  Closing in 30 days: {len(forecast['30d'])} deals")
    print(f"  Closing in 60 days: {len(forecast['60d'])} deals")
    print(f"  Closing in 90 days: {len(forecast['90d'])} deals")
    print(f"  Overdue: {len(forecast['overdue'])} deals")

    _total_forecast = len(forecast['30d']) + len(forecast['60d']) + len(forecast['90d']) + len(forecast['overdue'])  # noqa: F841
    with_notes = sum(1 for opp in forecast['30d'] + forecast['overdue'] if str(opp.get("id")) in notes_by_opp)
    print(f"\n30d + overdue with notes: {with_notes}/{len(forecast['30d']) + len(forecast['overdue'])}")

    # ============================================================
    # QUESTION 3: Account momentum
    # ============================================================
    print("\n" + "=" * 70)
    print("Q3: ACCOUNT MOMENTUM")
    print("=" * 70)

    # Build company pipeline
    company_pipeline: dict[str, float] = {}
    company_opps: dict[str, list] = {}
    for opp in open_opps:
        # Link via contacts -> companyID
        for oc in opp.get("contacts") or []:
            cid = oc.get("id") if isinstance(oc, dict) else None
            if cid and cid in contact_map:
                coid = contact_map[cid].get("companyID")
                if coid:
                    company_pipeline[coid] = company_pipeline.get(coid, 0) + (opp.get("productTotal") or 0)
                    company_opps.setdefault(coid, []).append(opp)

    companies_with_pipeline = [c for c in companies if c.get("id") in company_pipeline]
    print(f"\nMatches: {len(companies_with_pipeline)} companies with pipeline")

    # Check context
    with_notes = sum(1 for c in companies_with_pipeline if str(c.get("id")) in notes_by_company)
    with_history = sum(1 for c in companies_with_pipeline if str(c.get("id")) in history_by_company)
    print(f"With company notes: {with_notes}/{len(companies_with_pipeline)}")
    print(f"With company history: {with_history}/{len(companies_with_pipeline)}")

    # Also check via opportunities
    opp_ids = set()
    for coid in company_pipeline:
        for opp in company_opps.get(coid, []):
            opp_ids.add(str(opp.get("id")))
    with_opp_notes = sum(1 for oid in opp_ids if oid in notes_by_opp)
    print(f"Via opportunity notes: {with_opp_notes}/{len(opp_ids)}")

    # ============================================================
    # QUESTION 4: Relationship gaps
    # ============================================================
    print("\n" + "=" * 70)
    print("Q4: RELATIONSHIP GAPS")
    print("=" * 70)

    # Single-threaded deals (only 1 contact)
    single_threaded = [opp for opp in open_opps if len(opp.get("contacts") or []) <= 1]
    print(f"\nMatches: {len(single_threaded)} single-threaded deals")

    with_notes = sum(1 for opp in single_threaded if str(opp.get("id")) in notes_by_opp)
    print(f"With notes: {with_notes}/{len(single_threaded)}")

    # ============================================================
    # QUESTION 5: Daily briefing
    # ============================================================
    print("\n" + "=" * 70)
    print("Q5: DAILY BRIEFING")
    print("=" * 70)

    # Today's activities + recent history
    recent_history = history[:20]  # Most recent
    print(f"\nRecent history items: {len(recent_history)}")

    # Check for context
    with_details = sum(1 for h in recent_history if h.get("details"))
    print(f"With details: {with_details}/{len(recent_history)}")

    # Show sample
    if recent_history:
        h = recent_history[0]
        print(f"\nSample: [{h.get('regarding')}]")
        print(f"  {strip_html(h.get('details', ''))[:200]}...")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Context Coverage")
    print("=" * 70)

    print(f"""
Question                  | Matches | With Notes | With History
--------------------------|---------|------------|-------------
At-risk deals             | {len(at_risk):>7} | {with_notes:>10} | {with_history:>12}
Forecast (30d + overdue)  | {len(forecast['30d']) + len(forecast['overdue']):>7} | {sum(1 for opp in forecast['30d'] + forecast['overdue'] if str(opp.get('id')) in notes_by_opp):>10} | -
Account momentum          | {len(companies_with_pipeline):>7} | {with_opp_notes:>10} | -
Relationship gaps         | {len(single_threaded):>7} | {sum(1 for opp in single_threaded if str(opp.get('id')) in notes_by_opp):>10} | -
Daily briefing            | {len(recent_history):>7} | -          | {with_details:>12}
""")

    print("\nCONCLUSION:")
    total_opp_notes = len(notes_by_opp)
    total_opp_history = len(history_by_opp)
    print(f"  - {total_opp_notes} opportunities have linked notes")
    print(f"  - {total_opp_history} opportunities have linked history")
    print("  - Context available for most deal-based questions")


if __name__ == "__main__":
    analyze_question_matches()
