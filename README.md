# FW Agent — Intelligent LLM System

> Enterprise-grade AI agent with multi-provider LLM failover, intelligent tool selection, comprehensive observability, and production infrastructure.

**Key features**: Tool-calling agent that decides when to search documents vs. web search, automatic failover between Gemini/OpenRouter/OpenAI, decision tracking with user feedback, 3-layer observability (logs + traces + metrics), alerts via Discord/Slack.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Supabase (PostgreSQL + pgvector)
- Google API key (LLM + embeddings)
- Optional: OpenAI/OpenRouter keys, Discord/Slack webhooks

### Installation

```bash
cd agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
python server.py                # Start server (http://localhost:8000)
pytest                          # Run tests
python scripts/cronjob.py       # Start ingestion scheduler
```

---

## Architecture Overview

```
User Query → FastAPI → ToolCallingAgent (decides tools) → Tools → MultiProviderLLM (with circuit breaker)
  ↓
  Observability: LangSmith (traces) + CloudWatch (logs) + Supabase (decisions) + Metrics
  ↓
  Alerts: Discord/Slack (if error) → Response
```

**Full documentation with diagrams**: See [docs/architecture.md](docs/architecture.md)

---

## API Quick Reference

```bash
# Chat (batch or streaming)
curl -X POST http://localhost:8000/v1/chat \
  -d '{"query": "What is the enrollment process?", "top_k": 5}'

# View decisions with analytics
curl "http://localhost:8000/v1/decisions?quality=optimal"

# Record user feedback
curl -X POST http://localhost:8000/v1/feedback \
  -d '{"run_id": "...", "feedback_type": "like"}'

# System health + metrics + circuit status
curl http://localhost:8000/v1/system/status
```

**Full API reference**: [docs/architecture.md#API-Endpoints](docs/architecture.md#API-Endpoints)

---

## Directory Structure

```
agent/
├── api/              # HTTP endpoints (chat, decisions, feedback, health)
├── core/             # ToolCallingAgent, circuit breaker, alerts
├── llm/              # Google, OpenRouter, OpenAI providers
├── embeddings/       # Google Embeddings (1536-dim)
├── vector_store/     # Supabase pgvector integration
├── tools/            # search_documents, web_search, custom tools
├── ingestion/        # Document pipeline (parse → chunk → embed → store)
├── observability/    # LangSmith tracing, decision tracking
├── loggers/          # CloudWatch & console logging
├── alerts/           # Discord & Slack webhooks
├── config/           # Settings (40+ env vars)
├── models/           # Pydantic request/response models
├── scripts/          # Scheduled jobs (cronjob, backup, setupdb)
├── migrations/       # Alembic database schema
├── tests/            # Unit & integration tests (pytest)
├── knowledge/        # Document storage (raw_docs, processed, failed)
├── container.py      # Dependency injection
└── server.py         # FastAPI entry point
```

---

## Development

### Add a New Tool

1. Create `tools/my_tool.py` extending `BaseTool`
2. Implement `_run()` and `_arun()` methods
3. Register in `container.py` → `agent_tools` list
4. Add formatter in `tools/summaries.py` (optional)

See [docs/architecture.md#Extending-the-System](docs/architecture.md#Extending-the-System) for details.

### Add a New LLM Provider

1. Extend `LLMProvider` base class in `llm/my_provider.py`
2. Implement `is_configured()` and `invoke()` methods
3. Register in `container.py` → `_llm_providers` list

Circuit breaker automatically handles the new provider.

### Run Tests

```bash
pytest --cov=agent tests/          # All tests with coverage
pytest tests/unit/core/test_router.py  # Specific test file
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/architecture.md](docs/architecture.md) | Complete system design, all components, decision tracking, extensibility |
| [docs/runbooks/deployment-checklist.md](docs/runbooks/deployment-checklist.md) | EC2 setup, AWS CloudWatch, LangSmith, production checklist |
| [docs/runbooks/weekly-maintenance.md](docs/runbooks/weekly-maintenance.md) | Backups, VACUUM ANALYZE, health monitoring |
| [docs/runbooks/incident-response.md](docs/runbooks/incident-response.md) | Troubleshooting, provider failover, circuit breaker recovery |

---

## Configuration

Key environment variables (see [docs/architecture.md#Configuration](docs/architecture.md#Configuration) for all 40+):

```bash
GOOGLE_API_KEY=...              # Required
OPENROUTER_API_KEY=...          # Optional (fallback)
OPENAI_API_KEY=...              # Optional (last resort)
SUPABASE_URL=...                # Required
SUPABASE_DIRECT_URL=...         # Required
LANGSMITH_API_KEY=...           # Optional (tracing)
LOGGER_BACKEND=cloudwatch|console
DISCORD_WEBHOOK_URL=...         # Optional (alerts)
SLACK_WEBHOOK_URL=...           # Optional (alerts)
CRON_INTERVAL_MINUTES=5         # Ingestion frequency
```

---

## Deployment

### With systemd on EC2

```bash
# Copy to server
sudo cp -r agent/ /opt/langchain-agent/

# Create /etc/systemd/system/langchain-agent.service
[Unit]
Description=FW Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/langchain-agent/agent
ExecStart=/opt/langchain-agent/agent/.venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Enable & start
sudo systemctl daemon-reload
sudo systemctl enable langchain-agent
sudo systemctl start langchain-agent
```

### With Docker

```bash
cd agent
docker build -f Dockerfile -t langchain-agent:latest .
docker run -p 8000:8000 --env-file .env langchain-agent:latest
```

See [docs/runbooks/deployment-checklist.md](docs/runbooks/deployment-checklist.md) for production setup.

---

## Troubleshooting

**LLM provider failing?**
- Check: `GET /v1/llm/circuits` (circuit breaker status)
- View: CloudWatch logs at `/aws/ec2/langchain-agent`
- Verify: API keys in `.env`

**Document ingestion not working?**
- Check: `cronjob.py` logs
- Verify: `KNOWLEDGE_DIR` folder exists
- Status: `GET /v1/system/status` (Supabase connection)

**High latency?**
- Check: `GET /v1/metrics` (average latency)
- View: LangSmith dashboard (slowest operations)
- Monitor: EC2 instance resources

See [docs/runbooks/incident-response.md](docs/runbooks/incident-response.md) for detailed troubleshooting.

---

## Contributing

1. `git checkout -b feature/my-feature`
2. Write tests: `pytest tests/` must pass
3. Follow conventions: Type hints, docstrings, SOLID principles
4. Commit: Use conventional commits (`feat:`, `fix:`, `docs:`, etc.)
5. PR: Link to issue, describe changes

---

## Links

- **Detailed Docs**: [docs/architecture.md](docs/architecture.md)
- **API Docs**: http://localhost:8000/docs (when server running)
- **LangSmith**: https://smith.langchain.com
- **Supabase**: https://app.supabase.com

---

**Status**: Production-Ready ✅ | **Last Updated**: June 5, 2026
