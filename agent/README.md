# LangChain Agent — RAG API & Document Ingestion System

A production-ready Retrieval-Augmented Generation (RAG) system with a FastAPI chat API and batch document ingestion pipeline. Uses LangChain, Google Gemini embeddings, and Supabase pgvector for vector similarity search.

## Features

- **Chat API** — Query your documents with AI-generated answers and source citations
- **Batch Ingestion** — Automatic document processing (parse, split, embed, store)
- **Multi-Format Support** — `.txt`, `.md`, `.html`, `.pdf`, `.docx`, `.csv`
- **Pluggable LLM Providers** — OpenRouter (default), Google Gemini, OpenAI
- **Smart Deduplication** — SHA-256 content hashing prevents re-processing
- **Vector Search** — Supabase pgvector with similarity threshold filtering
- **Tool-Calling Agent** — LangChain agent with document search and web search tools
- **Observability** — LangSmith tracing, structured JSON logging, Discord alerts
- **User Feedback** — Like/dislike feedback captured via LangSmith

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Supabase project** (free tier: https://supabase.com)
- **Google API Key** (for embeddings: https://aistudio.google.com/app/apikeys)
- **OpenRouter API Key** (for LLM: https://openrouter.ai/keys)

### 1. Install

```bash
cd agent
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key
SUPABASE_DIRECT_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# LLM Provider (OpenRouter)
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o

# Optional
CRON_INTERVAL_MINUTES=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

### 3. Set Up Database

```bash
python setupdb.py
```

This runs Alembic migrations to create the schema: `documents`, `document_chunks`, `ingestion_logs`, and the `search_documents` RPC function.

### 4. Run

**API Server** (chat interface):
```bash
python server.py
# http://localhost:8000/docs — Swagger UI
```

**Ingestion Pipeline** (document processing):
```bash
python cronjob.py
# Watches knowledge/raw_docs/ and processes new files
# Note: for local dev only — Docker launches server.py + cronjob.py together
# via the container's CMD (see Dockerfile).
```

**Docker** (both services):
```bash
docker build -t langchain-agent .
docker run -p 8000:8000 --env-file .env langchain-agent
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Presentation                         │
│  api/ — FastAPI routes, dependency injection, metrics       │
├─────────────────────────────────────────────────────────────┤
│                          Domain                             │
│  domain/core/ — RAG chain orchestration (retrieve + generate)  │
│  domain/retrieval/ — Vector search with threshold filtering    │
│  domain/ingestion/ — Document ingestion pipeline               │
├─────────────────────────────────────────────────────────────┤
│                       Infrastructure                        │
│  infrastructure/llm/ — LLM providers (OpenRouter, Gemini, OpenAI) │
│  infrastructure/embeddings/ — Google Gemini embeddings            │
│  infrastructure/vector_store/ — Supabase pgvector                 │
│  infrastructure/parsers/ — Multi-format file parsers              │
│  infrastructure/tools/ — LangChain tools (search, web search)     │
│  infrastructure/agent/ — Agent strategies (tool-calling, RAG)     │
│  infrastructure/alerts/ — Discord webhook alerts                  │
│  infrastructure/feedback/ — LangSmith feedback                    │
│  infrastructure/logging/ — Structured JSON logging                │
│  infrastructure/container.py — Composition root (DI)              │
├─────────────────────────────────────────────────────────────┤
│                        Shared                               │
│  config/settings.py — Pydantic Settings (env vars)          │
│  models/ — Pydantic DTOs (request/response)                 │
│  utils/ — Exceptions, retry, rate limiter, correlation IDs  │
│  migrations/ — Alembic database migrations                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Patterns

- **Strategy Pattern** — Every pluggable service (LLM, embeddings, vector store, parsers, alerts, feedback, logging, agents) has an ABC with swappable implementations
- **Composition Root** — `infrastructure/container.py` wires all singletons in one place
- **Clean Architecture** — Domain logic (`domain/`) is independent of infrastructure (`infrastructure/`)
- **Dependency Injection** — API layer receives dependencies via FastAPI `Depends()`

---

## Project Structure

```
agent/
├── server.py                  # FastAPI entry point (uvicorn)
├── cronjob.py                 # Ingestion polling loop
├── setupdb.py                 # Alembic migration runner
├── requirements.txt           # Python dependencies
├── alembic.ini                # Alembic configuration
├── Dockerfile                 # Multi-stage Docker build
├── .env                       # Environment variables (git-ignored)
│
├── config/
│   ├── __init__.py            # load_dotenv() + configure_tracing()
│   └── settings.py            # Pydantic BaseSettings (all env vars)
│
├── api/
│   ├── routes.py              # /v1/chat, /v1/feedback, /v1/metrics, /v1/health
│   ├── dependencies.py        # FastAPI Depends() wrappers
│   └── metrics.py             # In-memory request/error/latency counters
│
├── models/
│   ├── chat.py                # ChatRequest
│   ├── responses.py           # ChatResponse, HealthResponse, MetricsResponse
│   ├── feedback.py            # FeedbackRequest, FeedbackResponse
│   ├── retrieval.py           # RetrievedDocument (internal)
│   └── document.py            # SourceDocument (API-facing)
│
├── domain/
│   ├── core/chain.py          # RAGChain: retrieve → format → generate
│   ├── retrieval/retriever.py # Retriever: embed query → vector search → filter
│   ├── ingestion/pipeline.py  # DocumentIngestionPipeline: parse → split → embed → store
│   └── utils/filters.py       # Similarity threshold filtering
│
├── infrastructure/
│   ├── container.py           # Composition root — wires all singletons
│   ├── llm/                   # LLM providers (OpenRouter, Gemini, OpenAI)
│   ├── embeddings/            # Google Gemini embeddings wrapper
│   ├── vector_store/          # Supabase pgvector operations
│   ├── parsers/               # File parsers (txt, md, html, pdf, docx, csv)
│   ├── tools/                 # LangChain tools (search_documents, web_search)
│   ├── agent/                 # Agent strategies (ToolCallingAgent, RAGChainAgent)
│   ├── alerts/                # Discord webhook alerts (rate-limited, deduped)
│   ├── feedback/              # LangSmith feedback provider
│   └── logging/               # Structured JSON logging (console, file)
│
├── utils/
│   ├── exceptions.py          # RAGException hierarchy + Severity enum
│   ├── correlation.py         # contextvars-based correlation IDs
│   ├── retry.py               # tenacity retry decorator with exponential backoff
│   ├── rate_limiter.py        # Sliding-window rate limiter
│   └── formatting.py          # Shared document formatting utilities
│
├── migrations/
│   ├── env.py                 # Alembic environment
│   └── versions/              # Migration scripts
│
└── knowledge/
    ├── raw_docs/              # Input: drop files here
    ├── processed/             # Successfully ingested
    └── failed/                # Failed ingestion
```

---

## API Endpoints

### POST /v1/chat — Query Documents with RAG

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I enroll in the program?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "response": "To enroll in the program, you need to...",
  "sources": [
    {
      "filename": "enrollment_guide.pdf",
      "similarity_score": 0.95,
      "content_preview": "Enrollment Guidelines: To enroll..."
    }
  ],
  "execution_time_ms": 2340.5,
  "model": "openai/gpt-4o"
}
```

### POST /v1/feedback — Submit User Feedback

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "abc-123",
    "is_positive": true,
    "comment": "Very helpful answer"
  }'
```

### GET /v1/health — Health Check

```bash
curl http://localhost:8000/v1/health
```

**Response:**
```json
{
  "status": "ok",
  "db_connected": true,
  "llm_connected": true,
  "embedding_connected": true
}
```

### GET /v1/metrics — Request Metrics

```bash
curl http://localhost:8000/v1/metrics
```

**Response:**
```json
{
  "total_requests": 1523,
  "total_errors": 12,
  "avg_latency_ms": 1842.3
}
```

---

## Configuration

All settings are loaded from environment variables via `config/settings.py`.

### Required

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Generative AI key (for embeddings) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service role key |
| `SUPABASE_DIRECT_URL` | Direct PostgreSQL connection string |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OPENROUTER_MODEL` | `openai/gpt-4o` | Model identifier |
| `OPENROUTER_TEMPERATURE` | `0.7` | Temperature (0.0-1.0) |
| `OPENROUTER_MAX_TOKENS` | `2000` | Max response tokens |
| `LLM_TIMEOUT_SECONDS` | `60` | Request timeout |

### Embeddings & Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (characters) |
| `EMBEDDING_RETRIES` | `3` | Retry attempts for embedding failures |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity score for retrieval |

### Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `CRON_INTERVAL_MINUTES` | `5` | Polling interval for new documents |
| `KNOWLEDGE_DIR` | `./knowledge/raw_docs` | Input directory |
| `PROCESSED_DIR` | `./knowledge/processed` | Successfully processed files |
| `FAILED_DIR` | `./knowledge/failed` | Failed processing files |

### Logging & Alerts

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOGGER_BACKEND` | `console` | Logger type: `console` or `file` |
| `LOG_FILE` | — | Log file path (if `LOGGER_BACKEND=file`) |
| `DISCORD_WEBHOOK_URL` | — | Discord webhook for error alerts |
| `ALERT_RATE_LIMIT_PER_MINUTE` | `5` | Max alerts per minute |

### Agent Strategy

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_TOOL_AGENT` | `true` | Use tool-calling agent (true) or RAG chain (false) |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated allowed origins |

### LangSmith Tracing

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | `langchain-agent` | LangSmith project name |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |

---

## Document Ingestion

### How It Works

1. **Drop files** into `knowledge/raw_docs/`
2. **`cronjob.py` polls** the directory every `CRON_INTERVAL_MINUTES`
3. **Pipeline processes** each new file:
   - **Parse** — Extract text using the appropriate parser (PDF, DOCX, etc.)
   - **Split** — Break into chunks using `RecursiveCharacterTextSplitter`
   - **Embed** — Generate vectors via Google Gemini embeddings (1536 dimensions)
   - **Store** — Insert document + chunks into Supabase pgvector
4. **Move files** to `knowledge/processed/` or `knowledge/failed/`
5. **Deduplication** — SHA-256 content hashing prevents re-processing

### Supported Formats

| Format | Parser | Library |
|--------|--------|---------|
| `.txt` | `TextFileParser` | Built-in |
| `.md` | `MarkdownParser` | Built-in |
| `.html` | `HTMLParser` | Built-in |
| `.pdf` | `PDFParser` | `pdfplumber` |
| `.docx` | `DOCXParser` | `python-docx` |
| `.csv` | `CSVParser` | Built-in |

---

## Development

### Run Locally

```bash
# Terminal 1: API server
python server.py

# Terminal 2: Ingestion pipeline (local dev only)
python cronjob.py
# In Docker, the container's CMD launches both processes automatically —
# no need to run cronjob.py manually.
```

### Run Tests

```bash
pytest tests/ -v
```

### Database Migrations

```bash
# Run migrations
python setupdb.py

# Reset database (WARNING: deletes all data)
python setupdb.py --reset
```

### Add a New LLM Provider

1. Create `infrastructure/llm/your_provider.py`
2. Implement the `LLMProvider` ABC from `infrastructure/llm/base.py`
3. Add configuration to `config/settings.py`
4. Wire it in `infrastructure/container.py`

### Add a New File Parser

1. Create `infrastructure/parsers/your_parser.py`
2. Implement the `FileParser` ABC from `infrastructure/parsers/parser.py`
3. Register it in `ParserFactory`

---

## Troubleshooting

### "GOOGLE_API_KEY not found"

```bash
# Add to .env
echo "GOOGLE_API_KEY=your_key_here" >> .env
```

### "Supabase connection failed"

```bash
# Verify credentials
python -c "from config import settings; print(settings.supabase_url)"

# Test direct connection
python -c "from supabase import create_client; from config import settings; \
           client = create_client(settings.supabase_url, settings.supabase_key); \
           print('Connected!')"
```

### "Table doesn't exist"

```bash
# Run migrations
python setupdb.py
```

### "Files not processing"

```bash
# Check knowledge directory
ls knowledge/raw_docs/

# Check logs
# If LOGGER_BACKEND=console: check stdout
# If LOGGER_BACKEND=file: tail -f <LOG_FILE>

# Verify ingestion pipeline is running
ps aux | grep cronjob
```

### "No results from chat API"

```bash
# Check that documents have been ingested
ls knowledge/processed/

# Verify vector store has data
# In Supabase Studio → Table Editor → document_chunks
```

---

## Deployment

### Docker

```bash
docker build -t langchain-agent .
docker run -p 8000:8000 --env-file .env langchain-agent
```

The Dockerfile uses a multi-stage build with Python 3.11-slim and includes health checks.

### Environment Variables

Pass all required environment variables via:
- `.env` file (with `--env-file`)
- Docker secrets
- Your orchestration platform's secret management

---

## License

MIT
