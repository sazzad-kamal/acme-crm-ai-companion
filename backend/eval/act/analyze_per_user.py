"""Analyze history/matches per user for the 5 questions (using history.details)."""

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
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&nbsp;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


def analyze_per_user() -> None:
    """Analyze data coverage per user for all 5 questions (using history.details)."""
    print("Fetching data...")
    contacts = _get("/api/contacts", {"$top": 1000})
    opps = _filter_test_records(_get("/api/opportunities", {"$top": 500}))
    history = _get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})
    calendar = _get("/api/calendar", {"startDate": time.strftime("%Y-%m-%d"), "$top": 30})

    cutoff_30d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30*24*60*60))
    cutoff_7d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 7*24*60*60))

    # Build history by contact (only those with details)
    history_by_contact: dict[str, list] = {}
    for h in history:
        if not h.get("details"):
            continue
        for c in h.get("contacts") or []:
            cid = c.get("id") if isinstance(c, dict) else c
            if cid:
                history_by_contact.setdefault(str(cid), []).append(h)

    # Get unique users
    users: set[str] = set()
    for c in contacts:
        if c.get("recordManager"):
            users.add(c.get("recordManager"))
    for o in opps:
        if o.get("manager"):
            users.add(o.get("manager"))

    # Commitment patterns for Q2 (in history details)
    commitment_patterns = [
        r"will send", r"will call", r"will follow", r"follow.?up",
        r"get back to", r"send.*quote", r"send.*invoice", r"send.*proposal",
        r"promised", r"need to", r"i'?ll send", r"i'?ll call"
    ]

    print(f"\nUsers found: {len(users)}")
    print(f"Contacts with history: {len(history_by_contact)}")
    print()
    print("=" * 100)
    print(f"{'User':<20} | {'Q1:Call':<8} | {'Q2:Promise':<10} | {'Q3:LostTouch':<12} | {'Q4:Meetings':<10} | {'Q5:FollowUp':<10}")
    print(f"{'':20} | {'w/Hist':<8} | {'Commits':<10} | {'30d+Hist':<12} | {'Upcoming':<10} | {'Recent':<10}")
    print("=" * 100)

    for user in sorted(users):
        # User's contacts
        user_contacts = [c for c in contacts if c.get("recordManager") == user]
        user_contact_ids = {str(c.get("id")) for c in user_contacts if c.get("id")}

        # Q1: Who should I call today? (contacts with open deals + history)
        user_opps = [o for o in opps if o.get("manager") == user or o.get("recordManager") == user]
        open_opps = [o for o in user_opps if (o.get("statusName") or "").lower() not in ["closed", "won", "lost"]]
        q1_deals_with_history = 0
        for opp in open_opps:
            for oc in opp.get("contacts") or []:
                cid = oc.get("id") if isinstance(oc, dict) else oc
                if cid and str(cid) in history_by_contact:
                    q1_deals_with_history += 1
                    break

        # Q2: What did I promise? (scan history details for commitments)
        q2_commitments = 0
        for cid in user_contact_ids:
            for h in history_by_contact.get(cid, []):
                text = strip_html(h.get("details", "")).lower()
                if any(re.search(p, text) for p in commitment_patterns):
                    q2_commitments += 1
                    break

        # Q3: Who am I losing touch with? (no recent activity but has history context)
        q3_losing_touch = 0
        for c in user_contacts:
            cid = str(c.get("id"))
            last_activity = c.get("lastReach") or c.get("lastAttempt") or ""
            last_date = str(last_activity)[:10] if last_activity else ""
            if (not last_date or last_date < cutoff_30d) and cid in history_by_contact:
                q3_losing_touch += 1

        # Q4: Meetings to prepare for (calendar items with contacts that have history)
        q4_meetings = 0
        for day in calendar:
            for item in day.get("items") or []:
                for ic in item.get("contacts") or []:
                    icid = ic.get("id") if isinstance(ic, dict) else ic
                    if icid and str(icid) in history_by_contact and str(icid) in user_contact_ids:
                        q4_meetings += 1
                        break

        # Q5: Who needs follow-up? (recently contacted with history for context)
        q5_followup = 0
        for cid in user_contact_ids:
            hist_list = history_by_contact.get(cid, [])
            if hist_list:
                latest = hist_list[0]  # Already sorted desc
                h_date = str(latest.get("startTime") or "")[:10]
                if h_date >= cutoff_7d:  # Contacted in last 7 days
                    q5_followup += 1

        # Only print users with some data
        if q1_deals_with_history > 0 or q2_commitments > 0 or q3_losing_touch > 0:
            print(f"{user:<20} | {q1_deals_with_history:>8} | {q2_commitments:>10} | {q3_losing_touch:>12} | {q4_meetings:>10} | {q5_followup:>10}")

    print("=" * 100)
    print("\nLegend (using history.details):")
    print("  Q1:Call    = Open deals with contact history (for context)")
    print("  Q2:Promise = Contacts with commitment language in history")
    print("  Q3:LostTouch = Contacts inactive 30+ days with history context")
    print("  Q4:Meetings = Upcoming meetings with contacts that have history")
    print("  Q5:FollowUp = Contacted in last 7 days with history context")


if __name__ == "__main__":
    analyze_per_user()
