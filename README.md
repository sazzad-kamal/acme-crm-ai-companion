# Acme CRM AI Companion

An AI-powered CRM assistant that answers natural language questions about your CRM data using a multi-step LangGraph agent pipeline. Built with FastAPI backend and React frontend.

## Architecture

```mermaid
flowchart TB
    subgraph Frontend["Frontend (React + TypeScript)"]
        UI[Chat Interface]
    end

    subgraph Backend["Backend (FastAPI)"]
        API["/api/chat/stream (SSE)"]

        subgraph Agent["LangGraph Agent Pipeline"]
            direction LR
            Fetch["Fetch Node<br/><i>Claude: SQL Planning</i>"]
            Answer["Answer Node<br/><i>GPT: Synthesis</i>"]
            Action["Action Node<br/><i>GPT: Suggestions</i>"]
            Followup["Followup Node<br/><i>GPT: Questions</i>"]
        end

        DuckDB[(DuckDB<br/>CRM Data)]
    end

    UI -->|"User Question"| API
    API --> Fetch
    Fetch -->|"Generated SQL"| DuckDB
    DuckDB -->|"Query Results"| Fetch
    Fetch --> Answer
    Answer --> Action
    Answer --> Followup
    Action -->|"SSE Stream"| UI
    Followup -->|"SSE Stream"| UI
```

### Agent Pipeline

| Node | Purpose | LLM Provider |
|------|---------|--------------|
| **Fetch** | Converts natural language → SQL, executes query | Claude (structured output) |
| **Answer** | Synthesizes data into human-readable response with evidence tags | GPT |
| **Action** | Suggests next steps based on results | GPT |
| **Followup** | Generates relevant follow-up questions | GPT |

### Key Design Decisions

- **Multi-provider LLM**: Claude for SQL planning (better structured output), GPT for synthesis
- **Evidence-based answers**: Responses include `[E1]`, `[E2]` tags linking claims to data
- **Streaming UX**: SSE streaming for real-time progress and token delivery
- **Grounding-first**: Strict prompts prevent hallucination; facts must come from CRM data only

## Project Structure

```
acme-crm-ai-companion/
├── backend/
│   ├── api/
│   │   ├── chat.py              # Chat streaming endpoint
│   │   ├── data.py              # Data explorer endpoints
│   │   └── health.py            # Health check
│   ├── agent/
│   │   ├── graph.py             # LangGraph workflow definition
│   │   ├── state.py             # Agent state schema
│   │   ├── streaming.py         # SSE event streaming
│   │   ├── fetch/
│   │   │   ├── node.py          # Fetch node implementation
│   │   │   ├── planner.py       # SQL planning chain (Claude)
│   │   │   └── sql/             # DuckDB connection & execution
│   │   ├── answer/
│   │   │   ├── node.py          # Answer node implementation
│   │   │   └── answerer.py      # Answer synthesis chain
│   │   ├── action/
│   │   │   ├── node.py          # Action node implementation
│   │   │   └── suggester.py     # Action suggestion chain
│   │   └── followup/
│   │       ├── node.py          # Followup node implementation
│   │       ├── suggester.py     # Followup generation chain
│   │       └── tree/            # Static followup tree fallback
│   ├── eval/                    # Evaluation framework (RAGAS)
│   ├── data/csv/                # CRM data files
│   └── main.py                  # FastAPI app
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── hooks/               # Custom hooks (useChatStream)
│   │   └── styles/              # CSS
│   └── e2e/                     # Playwright E2E tests
├── tests/                       # Backend unit tests
└── scripts/ci.sh                # Local CI runner
```

## Tech Stack

### Backend
- **Framework**: FastAPI + Uvicorn
- **Agent**: LangGraph with memory checkpointing
- **Database**: DuckDB (in-memory SQL over CSV)
- **LLMs**: OpenAI GPT (answers), Anthropic Claude (SQL planning)
- **Validation**: Pydantic v2

### Frontend
- **Framework**: React 18 + TypeScript 5
- **Build**: Vite 5
- **Testing**: Vitest + Playwright

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key
- Anthropic API key (optional, falls back to OpenAI)

### Backend

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key  # Optional

# Run
python -m uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## API Reference

### Stream Chat Response

```http
POST /api/chat/stream
Content-Type: application/json

{
  "question": "What deals are in the pipeline?",
  "session_id": "optional-session-id"
}
```

Returns Server-Sent Events (SSE):
```
event: fetch_start
data: {"node": "fetch"}

event: answer_chunk
data: {"content": "Based on the data..."}

event: action
data: {"suggestions": ["Export to CSV", "Schedule follow-up"]}

event: followup
data: {"questions": ["Which reps own these deals?", "..."]}

event: done
data: {}
```

### Starter Questions

```http
GET /api/chat/starter-questions
```

### Health Check

```http
GET /api/health
```

## Testing

```bash
# Run all CI checks
./scripts/ci.sh all

# Backend only (420 tests)
./scripts/ci.sh backend

# Frontend only (562 tests)
./scripts/ci.sh frontend

# E2E tests (167 tests)
cd frontend && npm run test:e2e
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key for SQL planning | No |
| `MOCK_LLM` | Enable mock LLM for testing | No |
| `ACME_LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) | No |

## Evaluation

The project includes a RAGAS-based evaluation framework:

```bash
# Run answer quality evaluation
python -m backend.eval.answer

# Run followup evaluation
python -m backend.eval.followup
```

Metrics tracked:
- **Faithfulness**: Are claims grounded in retrieved data?
- **Answer Relevancy**: Does the answer address the question?
- **Correctness**: Is the answer factually accurate?

## License

This project is for demonstration purposes.
