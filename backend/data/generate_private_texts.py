#!/usr/bin/env python3
"""
Generate private_texts.jsonl from source CSVs.

Sources:
- companies.csv (description field)
- contacts.csv (notes field)
- opportunities.csv (notes field)
- history.csv (description field)
- activities.csv (description field)
- attachments.csv (summary field)

Usage:
    python generate_private_texts.py [--merge-opportunities] [--input-dir DIR]

Options:
    --merge-opportunities  Merge opportunity_descriptions.csv into opportunities.csv first
    --input-dir DIR        Input directory for CSVs (default: ./csv)
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CSV_DIR = Path(__file__).parent / "csv"


@dataclass
class CsvSource:
    """Configuration for a CSV source file."""

    filename: str
    id_field: str
    text_field: str
    record_type: str
    title_fn: Callable[[dict[str, str]], str]
    metadata_fn: Callable[[dict[str, str]], dict[str, str]]
    # Optional: override which ID fields to extract
    company_id_field: str = "company_id"
    contact_id_field: str = "contact_id"
    opportunity_id_field: str = "opportunity_id"


# Define all CSV sources in one place
SOURCES: list[CsvSource] = [
    CsvSource(
        filename="companies.csv",
        id_field="company_id",
        text_field="description",
        record_type="company",
        title_fn=lambda r: r.get("name", ""),
        metadata_fn=lambda r: {
            "status": r.get("status", ""),
            "plan": r.get("plan", ""),
            "industry": r.get("industry", ""),
        },
        contact_id_field="",  # Companies don't have contact_id
        opportunity_id_field="",  # Companies don't have opportunity_id
    ),
    CsvSource(
        filename="contacts.csv",
        id_field="contact_id",
        text_field="notes",
        record_type="contact",
        title_fn=lambda r: f"{r.get('first_name', '')} {r.get('last_name', '')}".strip(),
        metadata_fn=lambda r: {
            "job_title": r.get("job_title", ""),
            "role": r.get("role", ""),
            "lifecycle_stage": r.get("lifecycle_stage", ""),
        },
        opportunity_id_field="",  # Contacts don't have opportunity_id directly
    ),
    CsvSource(
        filename="opportunities.csv",
        id_field="opportunity_id",
        text_field="notes",
        record_type="opportunity",
        title_fn=lambda r: r.get("name", ""),
        metadata_fn=lambda r: {
            "stage": r.get("stage", ""),
            "value": r.get("value", ""),
            "created_at": r.get("created_at", ""),
        },
        contact_id_field="primary_contact_id",
    ),
    CsvSource(
        filename="history.csv",
        id_field="history_id",
        text_field="description",
        record_type="history",
        title_fn=lambda r: r.get("subject", ""),
        metadata_fn=lambda r: {
            "history_type": r.get("type", ""),
            "occurred_at": r.get("occurred_at", ""),
            "owner": r.get("owner", ""),
            "source": r.get("source", ""),
        },
    ),
    CsvSource(
        filename="activities.csv",
        id_field="activity_id",
        text_field="description",
        record_type="activity",
        title_fn=lambda r: r.get("subject", ""),
        metadata_fn=lambda r: {
            "activity_type": r.get("type", ""),
            "due_datetime": r.get("due_datetime", ""),
            "owner": r.get("owner", ""),
            "status": r.get("status", ""),
            "priority": r.get("priority", ""),
        },
    ),
    CsvSource(
        filename="attachments.csv",
        id_field="attachment_id",
        text_field="summary",
        record_type="attachment",
        title_fn=lambda r: r.get("title", ""),
        metadata_fn=lambda r: {
            "file_type": r.get("file_type", ""),
            "created_at": r.get("created_at", ""),
        },
    ),
]


def merge_opportunity_descriptions(csv_dir: Path) -> None:
    """Merge opportunity_descriptions.csv text into opportunities.csv as notes column.

    This operation is idempotent - running multiple times won't corrupt data.
    """
    opps_path = csv_dir / "opportunities.csv"
    descs_path = csv_dir / "opportunity_descriptions.csv"

    if not descs_path.exists():
        print("No opportunity_descriptions.csv found, skipping merge")
        return

    if not opps_path.exists():
        print("No opportunities.csv found, skipping merge")
        return

    # Read opportunity descriptions
    desc_map: dict[str, str] = {}
    with open(descs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opp_id = row.get("opportunity_id", "")
            text = row.get("text", "")
            if opp_id and text:
                desc_map[opp_id] = text

    print(f"Loaded {len(desc_map)} opportunity descriptions")

    # Read opportunities and add/update notes column
    rows = []
    fieldnames: list[str] = []
    with open(opps_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "notes" not in fieldnames:
            fieldnames.append("notes")
        for row in reader:
            opp_id = row.get("opportunity_id", "")
            # Only update if we have a description (idempotent)
            if opp_id in desc_map:
                row["notes"] = desc_map[opp_id]
            elif "notes" not in row:
                row["notes"] = ""
            rows.append(row)

    # Write updated opportunities.csv
    with open(opps_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated opportunities.csv with notes column ({len(rows)} rows)")


def process_csv_source(source: CsvSource, csv_dir: Path, output_file: Any) -> int:
    """Process a single CSV source and write records to output file.

    Returns the number of records written.
    """
    csv_path = csv_dir / source.filename
    if not csv_path.exists():
        return 0

    count = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                text = row.get(source.text_field, "").strip()
                if not text:
                    continue

                record = {
                    "id": f"{source.record_type}::{row.get(source.id_field, '')}",
                    "company_id": row.get(source.company_id_field, "") if source.company_id_field else "",
                    "contact_id": row.get(source.contact_id_field, "") if source.contact_id_field else "",
                    "opportunity_id": row.get(source.opportunity_id_field, "") if source.opportunity_id_field else "",
                    "type": source.record_type,
                    "title": source.title_fn(row),
                    "text": text,
                    "metadata": source.metadata_fn(row),
                }

                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"  Warning: Error processing row in {source.filename}: {e}", file=sys.stderr)

    return count


def generate_private_texts(csv_dir: Path) -> int:
    """Generate private_texts.jsonl from source CSVs.

    Uses streaming writes to handle large datasets efficiently.
    """
    output_path = csv_dir / "private_texts.jsonl"
    total_count = 0

    print("Processing sources:")
    with open(output_path, "w", encoding="utf-8") as output_file:
        for source in SOURCES:
            count = process_csv_source(source, csv_dir, output_file)
            if count > 0:
                print(f"  {source.record_type.capitalize()}: {count} records")
            total_count += count

    print(f"\nGenerated {output_path.name} with {total_count} total records")
    return total_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate private_texts.jsonl from source CSVs")
    parser.add_argument(
        "--merge-opportunities",
        action="store_true",
        help="Merge opportunity_descriptions.csv into opportunities.csv first",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=CSV_DIR,
        help=f"Input directory for CSVs (default: {CSV_DIR})",
    )
    args = parser.parse_args()

    csv_dir = args.input_dir
    if not csv_dir.exists():
        print(f"Error: Input directory does not exist: {csv_dir}", file=sys.stderr)
        return 1

    if args.merge_opportunities:
        print("Step 1: Merging opportunity descriptions...")
        merge_opportunity_descriptions(csv_dir)
        print()

    print("Generating private_texts.jsonl...")
    generate_private_texts(csv_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
