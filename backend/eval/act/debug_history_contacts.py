"""Debug: Check what contact info is actually in history records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.act_fetch import _get


def debug_history() -> None:
    """Check history contact structure."""
    print("Fetching 10 history records...")
    history = _get("/api/history", {"$top": 10, "$orderby": "startTime desc"})

    for i, h in enumerate(history[:3]):
        print(f"\n{'='*60}")
        print(f"HISTORY RECORD {i+1}")
        print(f"{'='*60}")

        # Show all top-level fields
        print("\nTOP-LEVEL FIELDS:")
        for k, v in h.items():
            if k == "details":
                print(f"  {k}: (length {len(v or '')})")
            elif k == "contacts":
                print(f"  {k}: {json.dumps(v, indent=4)}")
            elif isinstance(v, dict):
                print(f"  {k}: {json.dumps(v)}")
            elif isinstance(v, list):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    debug_history()
