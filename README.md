# CRM Chat Assistant

> **Natural language queries over CRM data using multi-agent orchestration**

A production-grade AI assistant that answers business questions about your CRM using a **LangGraph multi-agent pipeline** with supervisor routing, specialized data agents, and evidence-grounded responses.

```
"What deals closed this quarter?"     →  Fetch Agent   →  SQL + Data
"Compare Q1 vs Q2 revenue"            →  Compare Agent →  Analysis
"Show revenue trend by month"         →  Trend Agent   →  Time-series
"Export top accounts to CSV"          →  Export Agent  →  File generation
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SUPERVISOR                                      │
│                                                                              │
│   User Query ──► Heuristics ──► Intent ──► Route to Specialized Agent       │
│                      │                                                       │
│                      └── LLM Fallback (only if heuristics uncertain)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │           │           │           │           │
            ▼           ▼           ▼           ▼           ▼
       ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
       │  FETCH  │ │ COMPARE │ │  TREND  │ │ PLANNER │ │ EXPORT  │
       │         │ │         │ │         │ │         │ │         │
       │ Simple  │ │ A vs B  │ │ Time-   │ │ Multi-  │ │ CSV/PDF │
       │ queries │ │ analysis│ │ series  │ │ step    │ │ JSON    │
       └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
            │           │           │           │           │
            └───────────┴───────────┴───────────┴───────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESPONSE PIPELINE                                  │
│                                                                              │
│   Data ──► ANSWER (synthesize with [E1] [E2] citations)                     │
│                │                                                             │
│                ├──► ACTION (suggest next steps)                              │
│                │                                                             │
│                └──► FOLLOWUP (generate related questions)                    │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Validate ──► Repair ──► Fallback  (every LLM output)               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Routing

| Agent | Intent | What It Does | Example |
|-------|--------|--------------|---------|
| **Fetch** | `data_query` | SQL generation + execution | "Show all deals over $50K" |
| **Compare** | `compare` | Side-by-side analysis | "Q1 vs Q2 revenue by region" |
| **Trend** | `trend` | Time-series patterns | "Monthly revenue trend" |
| **Planner** | `complex` | Multi-step orchestration | "Show deals and compare by rep" |
| **Export** | `export` | File generation | "Export pipeline to CSV" |
| **Health** | `health` | Account scoring | "What's Acme Corp's health score?" |

---

## Design Philosophy

### Lean AI: Deterministic First, LLM Where It Matters

The system follows a **heuristics-first architecture** — deterministic rules handle measurable, predictable tasks while LLMs focus on interpretation and synthesis.

| Layer | Approach | Why |
|-------|----------|-----|
| **Intent Classification** | Regex + keyword patterns → LLM fallback | 90% of queries match patterns; LLM only for ambiguous cases |
| **SQL Validation** | sqlglot AST parsing (blocks INSERT/UPDATE/DELETE) | Deterministic safety > LLM judgment |
| **Answer Grounding** | Evidence tags `[E1]`, `[E2]` required | Forces claims to link to data |
| **Output Validation** | Schema validation → repair prompt → fallback | Never crash on bad LLM output |

### Multi-Provider LLM Strategy

| Task | Model | Reasoning |
|------|-------|-----------|
| Intent classification | GPT-4o-mini | Fast, cheap, sufficient accuracy |
| SQL planning | Claude | Better at structured output |
| Answer synthesis | GPT-4 | Strong at natural language |
| Repair prompts | Same as original | Context preserved |

### Production Patterns

**Validate → Repair → Fallback** for every LLM output:

```
LLM Output
    │
    ▼
┌──────────────┐
│   Validate   │ ← Schema check (Pydantic)
└──────┬───────┘
       │
       ├── Valid? ────────────────────► Return result
       │
       ▼
┌──────────────┐
│    Repair    │ ← Re-prompt with error context
└──────┬───────┘
       │
       ├── Valid? ────────────────────► Return (repaired)
       │
       ▼
┌──────────────┐
│   Fallback   │ ← Safe default (never crashes)
└──────────────┘
```

**Data Refinement Loops**: Answer node can request additional Fetch iterations (max 2) when data is incomplete — the graph loops back to gather more context.

**Evidence-Grounded Responses**: Every claim must cite data with `[E1]`, `[E2]` markers. The Answer node enforces this format.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Orchestration** | LangGraph (stateful multi-agent workflows) |
| **LLMs** | OpenAI GPT-4, Anthropic Claude |
| **Backend** | FastAPI, Pydantic v2, DuckDB |
| **Frontend** | React 18, TypeScript, Vite |
| **Streaming** | Server-Sent Events (SSE) |
| **Testing** | pytest (420), Vitest (562), Playwright (167) |
| **Evaluation** | RAGAS (faithfulness, relevancy, correctness) |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key (required)
- Anthropic API key (optional, improves SQL planning)

### Backend

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key  # Optional

uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
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
event: answer_chunk    →  Response tokens
event: action          →  Suggested next steps
event: followup        →  Related questions
event: done            →  Complete
```

### Other Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/chat/starter-questions` | Initial prompts |
| `GET /api/health` | System health |

---

## Testing

```bash
# All tests (1,149 total)
./scripts/ci.sh all

# By layer
./scripts/ci.sh backend    # 420 tests
./scripts/ci.sh frontend   # 562 tests
cd frontend && npm run test:e2e  # 167 tests
```

---

## Evaluation

RAGAS-based quality metrics:

```bash
python -m backend.eval.answer    # Answer quality
python -m backend.eval.followup  # Followup relevance
```

| Metric | What It Measures |
|--------|------------------|
| **Faithfulness** | Claims grounded in retrieved data |
| **Relevancy** | Answer addresses the question |
| **Correctness** | Factual accuracy |

---

## Project Structure

```
├── backend/
│   ├── agent/
│   │   ├── graph.py           # LangGraph workflow
│   │   ├── supervisor/        # Intent classification + routing
│   │   ├── fetch/             # SQL planning + execution
│   │   ├── compare/           # A vs B analysis
│   │   ├── trend/             # Time-series queries
│   │   ├── answer/            # Response synthesis
│   │   └── validate/          # Output validation + repair
│   ├── api/                   # FastAPI routes
│   └── eval/                  # RAGAS evaluation
├── frontend/
│   ├── src/components/        # React UI
│   └── e2e/                   # Playwright tests
└── tests/                     # Backend unit tests
```

---

## License

MIT
