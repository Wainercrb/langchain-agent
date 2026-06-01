# Deployment Checklist

## Pre-Deployment Verification

Complete all items before proceeding with deployment.

### Code & Tests

- [ ] All tests pass: `cd agent && pytest`
- [ ] Lint clean: `cd agent && ruff check .` (if ruff is configured)
- [ ] No uncommitted changes: `git status` shows clean working tree
- [ ] Migration reviewed: any new migration in `migrations/versions/` is correct

### Environment

- [ ] All required env vars configured in `.env` or deployment config:
  - `GOOGLE_API_KEY`
  - `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_DIRECT_URL`
  - `OPENROUTER_API_KEY` (or alternative LLM provider)
  - `DISCORD_WEBHOOK_URL` (for alerting)
- [ ] New env vars for this release:
  - `RATE_LIMIT_ENABLED` (default: `true`)
  - `RATE_LIMIT_REQUESTS_PER_MINUTE` (default: `100`)

### Database

- [ ] Migration ready: `cd agent && alembic check` shows no conflicts
- [ ] Migration tested locally: `alembic upgrade head` succeeds
- [ ] Rollback plan: `alembic downgrade -1` tested and safe

### Infrastructure

- [ ] Discord webhook URL is valid and accessible
- [ ] Supabase project is running and accessible
- [ ] Vector store (pgvector) extension is enabled

---

## Deployment Steps

1. **Pull latest code**: `git pull origin main`
2. **Install dependencies**: `cd agent && pip install -r requirements.txt`
3. **Run migrations**: `cd agent && alembic upgrade head`
4. **Verify env vars**: `cd agent && python -c "from config import settings; print('OK')"`
5. **Start services**:
   - API server: `cd agent && python server.py`
   - Cronjob (if separate): `cd agent && python cronjob.py`
   - UI: `cd ui && pnpm dev`

---

## Post-Deployment Validation

- [ ] Health endpoint returns 200: `curl http://localhost:8000/v1/health`
- [ ] Metrics endpoint responds: `curl http://localhost:8000/v1/metrics`
- [ ] Chat endpoint works: `curl -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" -d '{"query": "test"}'`
- [ ] Dashboard loads: `http://localhost:4321/dashboard`
- [ ] Discord connectivity verified (check logs for any webhook errors)
- [ ] Rate limiting active: send rapid requests to `/v1/chat`, verify 429 after threshold
- [ ] LangSmith audit trail: verify traces appear at configured LangSmith project dashboard

---

## Rollback Procedure

If deployment fails:

1. Stop all services
2. Rollback migration: `alembic downgrade -1`
3. Checkout previous version: `git checkout <previous-commit>`
4. Restart services with previous code
