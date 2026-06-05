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
- [ ] `CORS_ORIGINS` set to your public domain (only needed if accessing API directly without nginx):
  ```
  CORS_ORIGINS='["http://your-ec2-ip"]'
  ```

### Database

- [ ] Migration ready: `cd agent && alembic check` shows no conflicts
- [ ] Migration tested locally: `alembic upgrade head` succeeds
- [ ] Rollback plan: `alembic downgrade -1` tested and safe

### Infrastructure

- [ ] nginx installed and config at `deploy/nginx.conf` is adapted to your paths
- [ ] Discord webhook URL is valid and accessible
- [ ] Supabase project is running and accessible
- [ ] Vector store (pgvector) extension is enabled

---

## Deployment Steps (nginx + FastAPI + Astro)

```
                            :80
  Browser ─────────────────→ nginx
                              │
                              ├── / → ui/dist/         (static files)
                              │
                              └── /v1/* → 127.0.0.1:8000  (FastAPI)
```

1. **Pull latest code**: `git pull origin main`
2. **Install Python deps**: `cd agent && pip install -r requirements.txt`
3. **Build frontend**: `cd ui && pnpm install && pnpm build`
4. **Run migrations**: `cd agent && alembic upgrade head`
5. **Verify env vars**: `cd agent && python -c "from config import settings; print('OK')"`
6. **Copy nginx config** and adapt paths:
   ```bash
   sudo cp deploy/nginx.conf /etc/nginx/sites-available/agent
   sudo ln -s /etc/nginx/sites-available/agent /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   ```
7. **Start API** (as systemd service or screen/tmux):
   ```bash
   cd agent && uvicorn server:app --host 127.0.0.1 --port 8000
   ```
8. **Start cronjob** (if document ingestion is needed):
   ```bash
   cd agent && python cronjob.py
   ```

---

## Post-Deployment Validation

- [ ] Frontend loads: `curl http://your-ec2-ip/`
- [ ] System status responds: `curl http://your-ec2-ip/v1/system/status`
- [ ] Chat endpoint works: `curl -X POST http://your-ec2-ip/v1/chat -H "Content-Type: application/json" -d '{"query": "test"}'`
- [ ] Dashboard loads in browser: `http://your-ec2-ip/dashboard`
- [ ] Decisions page loads: `http://your-ec2-ip/decisions`
- [ ] Chat streaming works in the UI
- [ ] Discord connectivity verified (check logs for any webhook errors)

---

## Rollback Procedure

If deployment fails:

1. Stop all services
2. Rollback migration: `alembic downgrade -1`
3. Checkout previous version: `git checkout <previous-commit>`
4. Rebuild frontend: `cd ui && pnpm build`
5. Restart services with previous code

---

## Astro production mode

The frontend is built with **Astro in static mode** (default). This means:

- `pnpm build` generates plain HTML/CSS/JS in `ui/dist/`
- No Node.js server is needed for the frontend — just serve the files with nginx
- API calls use relative URLs (same origin via nginx proxy)
- Each route (`/`, `/dashboard`, `/decisions`) is a separate `.html` file

The nginx config in `deploy/nginx.conf` handles the routing correctly:
- Static files served directly from `ui/dist/`
- `/v1/*` proxied to FastAPI on `127.0.0.1:8000`
