"""Debug: Try to fetch contact by ID from history."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.act_fetch import _get


def debug_contact_fetch() -> None:
    """Try to fetch contact by ID."""
    print("Fetching history to get a contact ID...")
    history = _get("/api/history", {"$top": 10, "$orderby": "startTime desc"})

    # Get first contact ID from history
    contact_id = None
    display_name = None
    for h in history:
        contacts = h.get("contacts") or []
        if contacts:
            contact_id = contacts[0].get("id")
            display_name = contacts[0].get("displayName")
            break

    if not contact_id:
        print("No contact found in history!")
        return

    print(f"\nContact from history: {display_name} (ID: {contact_id})")

    # Try to fetch this contact directly
    print("\nFetching contact by ID...")
    try:
        result = _get(f"/api/contacts/{contact_id}", {})
        print(f"Raw result type: {type(result)}")
        contact = result[0] if isinstance(result, list) and result else result
        print("SUCCESS! Got contact:")
        print(f"  fullName: {contact.get('fullName')}")
        print(f"  emailAddress: {contact.get('emailAddress')}")
        print(f"  company: {contact.get('company')}")
        print(f"  businessPhone: {contact.get('businessPhone')}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

    # Also check if we can find by name in contacts list
    print("\nSearching in contacts list by name...")
    contacts = _get("/api/contacts", {"$top": 1000})
    for c in contacts:
        if c.get("fullName") == display_name:
            print("FOUND by name:")
            print(f"  ID: {c.get('id')}")
            print(f"  emailAddress: {c.get('emailAddress')}")
            break
    else:
        print("NOT FOUND by name in top 1000 contacts")


if __name__ == "__main__":
    debug_contact_fetch()
