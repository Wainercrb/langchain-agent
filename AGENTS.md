# Agent Instructions — langchain-agent

## Project Stack

- **Language**: Python 3.11
- **Framework**: FastAPI 0.136, Uvicorn 0.24
- **Data Validation**: Pydantic 2.13, Pydantic-Settings 2.14
- **Database**: Supabase + pgvector (SQLAlchemy 2.0, Alembic migrations)
- **LLM Routing**: OpenRouter (primary), Google Gemini, OpenAI
- **LangChain**: langchain 1.3, langchain-core 1.4, langchain-google-genai 4.2, langchain-openai 1.0
- **Tracing**: LangSmith (`@traceable` decorators)
- **Retry**: Tenacity (exponential backoff)
- **Search**: DuckDuckGo (ddgs)
- **Documents**: pdfplumber, python-docx
- **Testing**: pytest 9.0, FastAPI TestClient
- **Frontend**: Astro (dashboard at `ui/src/pages/dashboard.astro`)

## Architecture

- **Strategy Pattern**: `ToolCallingAgent` (intelligent tool selection) vs `RAGChainAgent` (legacy retrieve-then-generate)
- **Composition Root**: `infrastructure/container.py` — single place to wire all services
- **Package Structure**: `api/`, `config/`, `domain/`, `infrastructure/`, `utils/`, `models/`
- **Entry Point**: `agent/server.py`

## Conventions

- Google-style docstrings (Args/Returns/Raises)
- Type hints throughout
- Pydantic BaseModel with Field/validators
- FastAPI dependency injection
- pytest fixtures, descriptive test names
- SOLID principles enforced
- Keep code simple and short — one responsibility per file

## Production Features

- Rate limiting middleware (`/v1/chat`, `/v1/rag`)
- Health checks (`/v1/health`)
- Metrics (`/v1/metrics`)
- Discord alerts for errors
- Ingestion failure alerting
- Runbooks: `docs/runbooks/`

## Testing

- Strict TDD mode active
- Run: `cd agent && pytest`
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
