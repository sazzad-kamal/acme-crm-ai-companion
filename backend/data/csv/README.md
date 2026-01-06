# CRM Synthetic Data Bundle

Drop this folder into your repo as `data/crm/`.

## Files

### Core Tables
- `companies.csv`: Company records
- `contacts.csv`: Contact records linked to companies
- `opportunities.csv`: Sales opportunities (includes `notes` column)
- `activities.csv`: Scheduled tasks and activities
- `history.csv`: Completed interactions (calls, emails, meetings, notes)
- `groups.csv`: Account group definitions
- `group_members.csv`: Links companies/contacts to groups
- `attachments.csv`: Document metadata and summaries

### Generated File
- `private_texts.jsonl`: Combined text content for RAG search (generated from opportunities, history, activities, attachments, groups)

## Regenerating private_texts.jsonl

Run from the backend/data directory:
```bash
python generate_private_texts.py
```
