# FW Agent — Architecture Guide

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Runtime |
| Framework | FastAPI 0.136 | HTTP API |
| Validation | Pydantic 2.13 | Request/response models |
| LLM Orchestration | LangChain 1.3 + LangGraph | Tool-calling agent |
| Database | Supabase + pgvector | Vector store + relational data |
| Migrations | Alembic | Schema versioning |
| Tracing | LangSmith | LLM observability |
| Frontend | Astro + Tailwind v4 | Dashboard UI |
| Alerts | Discord / Slack webhooks | Error notifications |
| Logging | Console (JSON) / CloudWatch | Structured logging |

---

## Architecture Overview

```
┌──────────────┐     ┌──────────────────────┐     ┌─────────────┐
│   FastAPI    │────→│   ToolCallingAgent   │────→│ LLM Provider│
│  (api/handlers)│     │  (core/tool_calling) │     │  (llm/*)    │
└──────┬───────┘     └──────────┬───────────┘     └──────┬──────┘
       │                       │                         │
       │              ┌────────▼────────┐      ┌─────────▼──────────┐
       │              │ DecisionTracker │      │ MultiProviderLLM   │
       │              │ (observability/ │      │ (core/router.py)   │
       │              │  decisions.py)  │      │  failover + circuit│
       │              └────────┬────────┘      │  breakers          │
       │                       │               └────────────────────┘
       │              ┌────────▼────────┐
       │              │   Supabase DB   │
       │              │  ai_decisions   │
       │              └─────────────────┘
       │
       │              ┌──────────────────┐
       └─────────────→│  MetricsStore    │  (in-memory, resets on restart)
                      │  request_count   │
                      │  error_count     │
                      │  avg_latency_ms  │
                      │  token counters  │
                      └──────────────────┘
```

### Composition Root

All services are wired in `container.py`. Change implementations there:

```python
# Example: swap alert provider
alert_service = MultiAlertProvider([DiscordAlertProvider()])
```

---

## LLM Providers

### Architecture

Multi-provider failover with circuit breakers, defined in `core/router.py`:

```
MultiProviderLLM
  ├── GoogleProvider    (gemini-2.5-flash)
  ├── OpenAIProvider    (gpt-4o-mini)
  └── OpenRouterProvider (google/gemma-4-31b-it:free)
```

### How failover works

1. Providers are tried **in order** (Google → OpenRouter → OpenAI)
2. Each has a **circuit breaker** (closed → open on 3 failures → half-open after 60s)
3. On transient errors, next provider is tried with **exponential backoff** (1s base, 30s max)
4. If all providers fail → `AllProvidersExhaustedError`

### Adding a new provider

1. Create `llm/anthropic.py` extending `LLMProvider`
2. Implement `chat_model` property + `is_configured()` + `invoke()`
3. Add to `_create_llm_providers()` in `container.py`
4. Circuit breaker + failover are automatic

### Circuit breaker states

| State | Meaning | Next action |
|-------|---------|-------------|
| `closed` | Working normally | Requests pass through |
| `open` | Failing, blocked | Requests rejected, timer counts down |
| `half_open` | Probing | One request allowed; success → closed, failure → open |

View current states: `GET /v1/llm/circuits` or the **Health** page in the UI.

---

## Tools

| Tool | File | Description |
|------|------|-------------|
| `search_documents` | `tools/search_documents.py` | Vector search over ingested documents |
| `web_search` | `tools/web_search.py` | DuckDuckGo web search |

Tools are registered in `container.py` → `_create_agent()`. Each tool extends LangChain's `BaseTool`.

---

## Decision Tracking

### What is recorded

Every LLM invocation generates a `DecisionLogEntry` with:

| Field | Description |
|-------|-------------|
| `run_id` | LangSmith trace ID |
| `query_preview` | First 200 chars of the user query |
| `agent_type` | `tool_calling` or `rag_chain` |
| `model_used` | LLM model that actually responded |
| `decision_quality` | `optimal` / `suboptimal` / `poor` |
| `tools_used` | Tool names in execution order |
| `chain_length` | Number of sequential tool calls |
| `chain_tools` | Detailed tool call records (input, output) |
| `latency_ms` | Total execution time |
| `reasoning_summary` | Why the agent chose those tools |
| `tool_selection_rationale` | Raw LLM reasoning text |
| `temperature` / `top_k` | Configuration at time of request |
| `user_feedback` | Like/dislike from the UI |

### Storage

- **In-memory**: `_DecisionStore` (deque, max 10,000 entries)
- **Persistent**: Supabase table `ai_decisions` (immediate write on each record)

### UI

| Page | Route | Description |
|------|-------|-------------|
| List | `/decisions` | Table with Time, Query, Quality, Model, Tools |
| Detail | `/decisions/detail?run_id=xxx` | Full decision metadata + tool chain |

### Decision Quality Classifier

Defined in `core/tool_calling_components.py` → `DecisionQualityClassifier`:

- **optimal**: Tool was appropriate for the query (e.g., search for factual questions)
- **suboptimal**: Tool was used but could have been better
- **poor**: Wrong tool selection or unnecessary chain

---

## Metrics

### What is tracked (in-memory, resets on restart)

| Metric | Source | Description |
|--------|--------|-------------|
| `request_count` | `chat.py` | Total chat requests since startup |
| `error_count` | `chat.py` | Failed requests |
| `avg_latency_ms` | `chat.py` | Average response time |
| `total_input_tokens` | `chat.py` | Total prompt tokens |
| `total_output_tokens` | `chat.py` | Total completion tokens |

Endpoint: `GET /v1/system/status` (includes health + metrics + circuits).

---

## Health Monitoring

### Single consolidated endpoint

```
GET /v1/system/status
```

Returns:
```json
{
  "status": "ok",
  "db_connected": true,
  "llm_connected": true,
  "embedding_connected": true,
  "circuits": {
    "google": "closed",
    "openai": "closed",
    "openrouter": "closed"
  },
  "request_count": 42,
  "error_count": 2,
  "avg_latency_ms": 1234.5,
  "total_input_tokens": 15000,
  "total_output_tokens": 32000
}
```

### Background scheduler

`MonitoringScheduler` (`observability/health/scheduler.py`) runs periodic health checks on a configurable interval. Results feed into the sled mechanism for traffic shedding.

---

## Alerts

### Providers

| Provider | File | Config |
|----------|------|--------|
| Discord | `alerts/discord.py` | `DISCORD_WEBHOOK_URL` |
| Slack | `alerts/slack.py` | `SLACK_WEBHOOK_URL` |

### Architecture

`MultiAlertProvider` (`core/dispatcher.py`) fans out alerts to all configured providers. Each provider handles:

- **Rate limiting**: Max 5 alerts/minute (configurable)
- **Deduplication**: Same fingerprint suppresses repeated alerts
- **Severity filtering**: INFO / WARNING / ERROR / CRITICAL

### Adding a new alert provider

1. Create `alerts/teams.py` extending `AlertProviderBase`
2. Implement `is_configured()` + `send_alert()`
3. Add to `_create_alert_providers()` in `container.py`

### When alerts fire

| Trigger | Severity | Source |
|---------|----------|--------|
| Chat validation error | WARNING | `chat.py` |
| LLM provider exhausted | ERROR | `chat.py` |
| Monitoring check failure | WARNING | `health/scheduler.py` |
| Ingestion failure | WARNING | `cronjob.py` |
| Backup failure | ERROR | `backup.py` |
| Unhandled exception | ERROR | `global_exception.py` |

---

## Logging

Two backends, configured via `LOGGER_BACKEND`:

| Backend | Output | Use case |
|---------|--------|----------|
| `console` (default) | JSON to stdout | Local dev |
| `cloudwatch` | AWS CloudWatch Logs | Production |

Usage: `from loggers import logger`

The logger is built in `loggers/__init__.py` → `_build_logger()`.

---

## Observability (LangSmith)

All agent invocations are traced via `@traceable` decorators and `LangSmithObservabilityProvider` (`observability/langsmith.py`).

Trace tags include:
- `model:{actual_model_used}`
- `agent:{tool_calling|rag_chain}`
- `decision_quality:{optimal|suboptimal|poor}`
- `chain_length:{N}`
- `tools_used:{tool1,tool2}`

---

## Database Schema

### Tables (managed by Alembic)

| Table | Purpose |
|-------|---------|
| `documents` | Ingested document metadata |
| `document_chunks` | Vectorized chunks with embeddings |
| `ai_decisions` | Decision audit trail |
| `ingestion_logs` | Document processing history |

### `ai_decisions` columns

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT PK | LangSmith trace ID |
| `agent_type` | TEXT | Strategy used |
| `query_preview` | TEXT | User query (first 200 chars) |
| `query_hash` | TEXT(50) | SHA-256 hash |
| `tools_used` | JSONB | Tool names |
| `chain_length` | INT | Tool call count |
| `chain_tools` | JSONB | Detailed tool records |
| `decision_quality` | TEXT | optimal/suboptimal/poor |
| `timestamp` | TIMESTAMPTZ | When it happened |
| `model_used` | TEXT | LLM model identifier |
| `top_k` | INT | Documents retrieved |
| `temperature` | FLOAT | LLM temperature |
| `latency_ms` | FLOAT | Execution time |
| `reasoning_summary` | TEXT | AI reasoning |
| `tool_selection_rationale` | TEXT | Raw LLM reasoning |
| `user_feedback` | JSONB | Like/dislike data |

---

## Frontend

Built with **Astro** + **Tailwind CSS v4**.

### Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Chat | Main chat interface |
| `/dashboard` | Health | System status, circuit breakers, metrics |
| `/decisions` | Decisions | Decision audit trail table |
| `/decisions/detail` | Decision Detail | Full decision metadata |

### Layout

- **Sidebar**: Fixed left (240px), navigation links
- **Header**: Top bar with page title
- **Content**: Centered `max-w-4xl`, scrolls naturally
- **Chat**: Fixed overlay, below header, with sidebar offset

---

## Extending the System

### Add a new API endpoint

1. Create `api/handlers/{name}.py` with an `APIRouter`
2. Register in `api/handlers/__init__.py`
3. Include in `api/router.py`

### Add a new tool

1. Create a LangChain `BaseTool` in `tools/`
2. Add to `agent_tools` list in `container.py` → `_create_agent()`

### Add a new LLM provider

1. Create `llm/{name}.py` extending `LLMProvider`
2. Add to `_create_llm_providers()` in `container.py`
3. Circuit breaker + failover are automatic

### Add a new alert channel

1. Create `alerts/{name}.py` extending `AlertProviderBase`
2. Add to `_create_alert_providers()` in `container.py`

### Add a new page

1. Create `ui/src/pages/{name}.astro`
2. Add nav item in `ui/src/components/Sidebar.astro`

---

## Configuration (.env)

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/service key |
| `SUPABASE_DIRECT_URL` | Direct DB connection string |
| `LANGSMITH_API_KEY` | LangSmith tracing |
| `DISCORD_WEBHOOK_URL` | Discord alert webhook |
| `SLACK_WEBHOOK_URL` | Slack alert webhook |
| `LOGGER_BACKEND` | `console` or `cloudwatch` |
| `CRON_INTERVAL_MINUTES` | Ingestion/maintenance interval |
