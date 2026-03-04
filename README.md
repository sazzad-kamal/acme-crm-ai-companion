# CRM Chat Assistant

**Ask questions about your CRM in plain English. Get answers grounded in data.**

A multi-agent AI system that routes queries to specialized agents, enforces output contracts, and never hallucinates.

![Tests](https://img.shields.io/badge/tests-1,149_passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-purple)

---

## How It Works

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             SUPERVISOR                                      │
│                    Heuristics-first classification                          │
│                    (LLM fallback only when needed)                          │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
        ┌─────────┬─────────┬─────┴─────┬─────────┬─────────┐
        ▼         ▼         ▼           ▼         ▼         ▼
    ┌───────┐ ┌───────┐ ┌───────┐ ┌─────────┐ ┌───────┐ ┌───────┐
    │ FETCH │ │COMPARE│ │ TREND │ │ PLANNER │ │EXPORT │ │HEALTH │
    │       │ │       │ │       │ │         │ │       │ │       │
    │  SQL  │ │ A vs B│ │ Time  │ │ Multi-  │ │ CSV/  │ │Account│
    │ Query │ │       │ │ Series│ │ Agent   │ │ PDF   │ │ Score │
    └───┬───┘ └───┬───┘ └───┬───┘ └────┬────┘ └───┬───┘ └───┬───┘
        │         │         │          │          │         │
        └─────────┴─────────┴────┬─────┴──────────┴─────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              ANSWER                                         │
│                                                                             │
│   • Synthesize with evidence tags [E1], [E2]                               │
│   • Loop back if more data needed (max 2)                                   │
│   • Validate → Repair → Fallback (never crashes)                           │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
               ┌─────────┐                ┌──────────┐
               │ ACTION  │                │ FOLLOWUP │
               │         │                │          │
               │ "Next:  │                │ 3 smart  │
               │ call X" │                │ questions│
               └─────────┘                └──────────┘
```

---

## What Makes This Production-Grade

### 1. Multi-Agent Orchestration

6 specialized agents, each optimized for its task:

| Query | Agent | What Happens |
|-------|-------|--------------|
| "Show Q1 deals" | Fetch | SQL generation → DuckDB |
| "Q1 vs Q2 revenue" | Compare | Parallel queries → Delta analysis |
| "Revenue trend" | Trend | Time-series → Growth calculation |
| "Deals and compare regions" | **Planner** | Decompose → Route to multiple agents → Aggregate |
| "Export to CSV" | Export | Query → File generation |
| "Acme health score" | Health | Multi-factor scoring |

**Planner handles complex queries** by decomposing them and routing sub-queries to the right agents:

```
"Show all deals and compare Q1 vs Q2"
              │
              ▼
          PLANNER
              │
    ┌─────────┴─────────┐
    ▼                   ▼
  FETCH              COMPARE
"Show deals"        "Q1 vs Q2"
    │                   │
    └─────────┬─────────┘
              ▼
          AGGREGATE → ANSWER
```

### 2. Heuristics-First Classification

**90% of queries classified without LLM** — pattern matching handles obvious cases:

```
"export deals"     →  EXPORT   (keyword match)
"Q1 vs Q2"         →  COMPARE  (keyword match)
"trend over time"  →  TREND    (keyword match)
"yes"              →  CLARIFY  (too short)
"hmm not sure"     →  LLM      (ambiguous → GPT-4o-mini)
```

### 3. Contract-Enforced Outputs

Every LLM output goes through **Validate → Repair → Fallback**:

```
LLM Output → Validate → [invalid] → Repair (re-prompt) → [still bad] → Fallback
                ↓
            [valid] → Return
```

| Output | Contract | Fallback |
|--------|----------|----------|
| Answer | Must have `[E1]` evidence tags | "I don't have that data" |
| Action | Numbered list, ≤28 words each | Empty list |
| Followup | Exactly 3 questions, ≤10 words | Static suggestions |

**The system never crashes on bad LLM output.**

### 4. Evidence-Grounded Responses

Every claim cites its source:

```
Answer: The deal is in Negotiation [E1] valued at $50,000 [E2].

Evidence:
- E1: opportunities.stage = "Negotiation"
- E2: opportunities.value = 50000
```

Optional **grounding verifier** catches hallucinations before they ship.

### 5. Data Refinement Loops

Answer can request more data (max 2 iterations):

```
Fetch → Answer: "I have deals but need contacts"
                   │
                   └─→ needs_more_data = true
                          │
                          ▼
                   Fetch (refined) → Answer: "Here's the complete picture..."
```

### 6. SQL Safety Guard

All LLM-generated SQL validated via `sqlglot`:

- **Blocked**: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`
- **Auto-added**: `LIMIT 1000` (prevent memory exhaustion)
- **Blocked**: `read_csv()` (no file access)

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| **Orchestration** | LangGraph | Stateful multi-agent workflows |
| **SQL Generation** | Claude | Better structured output than GPT |
| **Synthesis** | GPT-4 | Natural language strength |
| **Database** | DuckDB | Fast SQL over CSV |
| **Backend** | FastAPI + Pydantic v2 | Type-safe, async |
| **Frontend** | React 18 + TypeScript | Modern, type-safe |
| **Streaming** | SSE | Real-time token delivery |

---

## Quality

| Metric | Value |
|--------|-------|
| **Tests** | 1,149 (420 backend + 562 frontend + 167 E2E) |
| **Faithfulness SLO** | ≥ 0.9 (RAGAS) |
| **p50 Latency** | ≤ 3s |
| **Eval Framework** | RAGAS with regression gates |

---

## Quick Start

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

---

## API

```http
POST /api/chat/stream
{"question": "What deals closed this quarter?"}
```

**SSE Events**: `fetch_start` → `answer_chunk` (streaming) → `action` → `followup` → `done`

---

## Project Structure

```
backend/
├── agent/
│   ├── graph.py           # LangGraph workflow
│   ├── supervisor/        # Intent classification (heuristics + LLM)
│   ├── fetch/             # SQL planning (Claude) + execution (DuckDB)
│   ├── compare/           # A vs B analysis
│   ├── trend/             # Time-series patterns
│   ├── planner/           # Multi-agent orchestration
│   ├── answer/            # Response synthesis with evidence
│   └── validate/          # Contract enforcement (repair chain)
└── eval/                  # RAGAS evaluation framework
```

---

## License

MIT
