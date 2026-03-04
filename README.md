# CRM Chat Assistant

> **Natural language queries over CRM data using multi-agent orchestration**

A production-grade AI assistant that answers business questions about your CRM using a **LangGraph multi-agent pipeline** with supervisor routing, specialized data agents, and evidence-grounded responses.

```
"What deals closed this quarter?"     →  Fetch Agent   →  SQL + Data
"Compare Q1 vs Q2 revenue"            →  Compare Agent →  Side-by-side analysis
"Show revenue trend by month"         →  Trend Agent   →  Time-series patterns
"Show deals and compare by region"    →  Planner Agent →  Multi-step orchestration
"Export pipeline to CSV"              →  Export Agent  →  File generation
"What's Acme's health score?"         →  Health Agent  →  Account scoring
```

---

## Architecture

```
                                ┌─────────────────────────────────────────┐
                                │              SUPERVISOR                  │
                                │                                          │
                                │   Heuristics-first intent classification │
                                │   (LLM fallback for ambiguous cases)     │
                                └───────────────────┬─────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │           │           │       │       │           │           │
                    ▼           ▼           ▼       ▼       ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
              │  FETCH  │ │ COMPARE │ │  TREND  │ │...│ │ PLANNER │ │ EXPORT  │ │ HEALTH  │
              │         │ │         │ │         │ │   │ │         │ │         │ │         │
              │ Simple  │ │ A vs B  │ │ Time-   │ │   │ │ Decompose│ │ CSV/PDF │ │ Account │
              │ SQL     │ │ Analysis│ │ Series  │ │   │ │ & Route │ │ JSON    │ │ Scoring │
              └────┬────┘ └────┬────┘ └────┬────┘ └───┘ └────┬────┘ └────┬────┘ └────┬────┘
                   │           │           │                  │           │           │
                   └───────────┴───────────┴──────────────────┴───────────┴───────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ANSWER NODE                                            │
│                                                                                           │
│   Synthesize response with evidence tags [E1], [E2]                                       │
│                         │                                                                 │
│                         ├── needs_more_data? ───► Loop back to Fetch (max 2 iterations)  │
│                         │                                                                 │
│                         ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│   │     Validate ──► Repair ──► Fallback   (contract enforcement)                   │    │
│   └─────────────────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ▼                               ▼
                              ┌───────────┐                   ┌───────────┐
                              │  ACTION   │                   │ FOLLOWUP  │
                              │           │                   │           │
                              │ Suggest   │                   │ Generate  │
                              │ next steps│                   │ 3 questions│
                              └───────────┘                   └───────────┘
```

### Intent Classification (8 intents → 6 specialized agents)

| Intent | Route | Description | Example |
|--------|-------|-------------|---------|
| `data_query` | **Fetch** | Simple SQL queries | "Show all deals over $50K" |
| `compare` | **Compare** | Side-by-side analysis | "Q1 vs Q2 revenue by region" |
| `trend` | **Trend** | Time-series patterns | "Monthly revenue trend" |
| `complex` | **Planner** | Multi-step queries | "Show deals and compare by rep" |
| `export` | **Export** | File generation | "Export pipeline to CSV" |
| `health` | **Health** | Account scoring | "What's Acme's health score?" |
| `clarify` | **Answer** (direct) | Vague questions | "yes", "that one" |
| `help` | **Answer** (direct) | Usage questions | "What can you do?" |

---

## Design Principles

### 1. Heuristics First, LLM Where It Matters

**90% of queries match keyword patterns — no LLM needed for classification.**

```python
# Intent classification priority:
1. Short/vague → CLARIFY (no LLM)
2. "export", "csv", "download" → EXPORT (no LLM)
3. "vs", "compare" → COMPARE (no LLM)
4. "trend", "over time" → TREND (no LLM)
5. Multi-part with "and" → COMPLEX (no LLM)
6. Health keywords → HEALTH (no LLM)
7. Data indicators → DATA_QUERY (no LLM)
8. Ambiguous → LLM fallback (GPT-4o-mini)
```

### 2. Data Refinement Loops

Answer node can request additional data when incomplete:

```
User: "Show top deals and their contacts"
  │
  ▼
Fetch → Answer: "I have deals but need contact data"
         │
         └── needs_more_data=True
               │
               ▼
         Fetch (refined) → Answer: "Here are the deals with contacts..."
```

**Max 2 refinement iterations** to prevent infinite loops.

### 3. Validate → Repair → Fallback

**Every LLM output** goes through contract enforcement:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Validate   │────►│   Repair    │────►│  Fallback   │
│             │     │             │     │             │
│ Schema check│     │ Re-prompt   │     │ Safe default│
│ Format rules│     │ with errors │     │ Never crash │
└─────────────┘     └─────────────┘     └─────────────┘
```

| Output | Validation | Repair Strategy |
|--------|------------|-----------------|
| **Answer** | Evidence tags `[E1]`, sections | Re-prompt with format |
| **Action** | Numbered list, owner prefix, ≤28 words | Re-prompt with examples |
| **Followup** | Exactly 3 questions, ≤10 words each | Re-prompt with constraints |

### 4. Evidence-Grounded Responses

Every claim must cite data:

```
Answer: The deal is in Negotiation stage [E1] valued at $50,000 [E2].

Evidence:
- E1: opportunities table, row 1, stage="Negotiation"
- E2: opportunities table, row 1, value=50000
```

Optional **grounding verifier** catches ungrounded claims before response.

### 5. SQL Safety Guard

All LLM-generated SQL validated before execution:

```python
# Blocked operations
INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, GRANT

# Safety measures
- Auto-adds LIMIT 1000 (prevent memory exhaustion)
- Validates via sqlglot AST parsing
- Blocks read_csv() (file access)
```

---

## Multi-Provider LLM Strategy

| Task | Model | Why |
|------|-------|-----|
| Intent classification | GPT-4o-mini | Fast, cheap, sufficient |
| SQL planning | **Claude** | Better structured output |
| Answer synthesis | GPT-4 | Natural language strength |
| Repair prompts | Same as original | Preserve context |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Orchestration** | LangGraph (stateful multi-agent) |
| **LLMs** | OpenAI GPT-4, Anthropic Claude |
| **Backend** | FastAPI, Pydantic v2, DuckDB |
| **Frontend** | React 18, TypeScript, Vite |
| **Streaming** | Server-Sent Events (SSE) |
| **Testing** | pytest, Vitest, Playwright |
| **Evaluation** | RAGAS (faithfulness, relevancy) |

---

## Testing & Evaluation

### Test Coverage

| Layer | Tests |
|-------|-------|
| Backend (pytest) | 420 |
| Frontend (Vitest) | 562 |
| E2E (Playwright) | 167 |
| **Total** | **1,149** |

### RAGAS Evaluation

```bash
python -m backend.eval.integration  # Full conversation eval
```

| Metric | SLO |
|--------|-----|
| Pass Rate | ≥ 95% |
| Faithfulness | ≥ 0.9 |
| Answer Relevancy | ≥ 0.85 |
| p50 Latency | ≤ 3000ms |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key (required)
- Anthropic API key (optional, improves SQL)

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

---

## API

### Chat (Streaming)

```http
POST /api/chat/stream
Content-Type: application/json

{"question": "What deals closed this quarter?"}
```

**SSE Events:**
```
event: fetch_start     →  Agent activated
event: answer_chunk    →  Response tokens (streaming)
event: action          →  Suggested next steps
event: followup        →  Related questions
event: done            →  Complete
```

---

## Project Structure

```
backend/
├── agent/
│   ├── graph.py              # LangGraph workflow
│   ├── supervisor/           # Intent classification + routing
│   │   └── classifier.py     # Heuristics + LLM fallback
│   ├── fetch/                # SQL planning + execution
│   │   └── sql/guard.py      # SQL safety validation
│   ├── compare/              # A vs B analysis
│   ├── trend/                # Time-series queries
│   ├── planner/              # Multi-step orchestration
│   ├── export/               # CSV/PDF/JSON generation
│   ├── health/               # Account scoring
│   ├── answer/               # Response synthesis
│   ├── action/               # Next step suggestions
│   ├── followup/             # Question generation
│   └── validate/             # Contract enforcement
│       ├── answer.py         # Answer validation
│       ├── repair.py         # Repair chain
│       └── grounding.py      # Grounding verifier
├── eval/                     # RAGAS evaluation framework
└── api/                      # FastAPI routes
```

---

## License

MIT
