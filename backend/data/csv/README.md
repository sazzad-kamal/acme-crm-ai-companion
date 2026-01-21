# CRM Synthetic Data (Source CSVs)

## Files

- `companies.csv`: Company records (includes `notes` column for RAG)
- `contacts.csv`: Contact records linked to companies (includes `notes` column for RAG)
- `opportunities.csv`: Sales opportunities (includes `notes` column for RAG)
- `activities.csv`: Scheduled tasks and activities (includes `notes` column for RAG)
- `history.csv`: Completed interactions (calls, emails, meetings, notes)

## Regenerating texts.jsonl

Run from the backend/data directory:
```bash
python generate_texts.py
```

Output: `backend/data/texts.jsonl`
