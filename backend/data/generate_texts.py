#!/usr/bin/env python3
"""Generate texts.jsonl from source CSVs."""

import csv
import json
import sys
from pathlib import Path

CSV_DIR = Path(__file__).parent / "csv"

# (filename, id_field, text_field, record_type)
SOURCES = [
    ("companies.csv", "company_id", "notes", "company"),
    ("contacts.csv", "contact_id", "notes", "contact"),
    ("opportunities.csv", "opportunity_id", "notes", "opportunity"),
    ("history.csv", "history_id", "notes", "history"),
    ("activities.csv", "activity_id", "notes", "activity"),
]

ENTITY_KEYS = ["company_id", "contact_id", "opportunity_id"]

# CSV column name differs from output field
FIELD_REMAP = {("opportunity", "contact_id"): "primary_contact_id"}


def process_csv_source(
    filename: str, id_field: str, text_field: str, record_type: str, csv_dir: Path
) -> list[dict]:
    """Process a single CSV source and return records."""
    csv_path = csv_dir / filename
    if not csv_path.exists():
        return []

    records = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get(text_field, "").strip()
            if not text:
                continue

            record = {
                "id": f"{record_type}::{row.get(id_field, '')}",
                "type": record_type,
                "text": text,
            }
            for key in ENTITY_KEYS:
                col = FIELD_REMAP.get((record_type, key), key)
                record[key] = row.get(col, "")

            records.append(record)

    return records


def generate_texts(csv_dir: Path) -> int:
    """Generate texts.jsonl from source CSVs."""
    output_path = csv_dir.parent / "texts.jsonl"
    all_records = []

    print("Processing sources:")
    for filename, id_field, text_field, record_type in SOURCES:
        records = process_csv_source(filename, id_field, text_field, record_type, csv_dir)
        if records:
            print(f"  {record_type.capitalize()}: {len(records)} records")
            all_records.extend(records)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nGenerated {output_path.name} with {len(all_records)} total records")
    return len(all_records)


def main() -> int:
    if not CSV_DIR.exists():
        print(f"Error: CSV directory not found: {CSV_DIR}", file=sys.stderr)
        return 1

    print("Generating texts.jsonl...")
    generate_texts(CSV_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
