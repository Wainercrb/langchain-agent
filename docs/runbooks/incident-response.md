# Incident Response Runbook

## Severity Levels

| Severity | Response SLA | Description | Example |
|----------|-------------|-------------|---------|
| P1 - Critical | 15 min | Complete service outage, data loss risk | Database unreachable, all requests failing |
| P2 - High | 1 hour | Major feature broken, significant user impact | LLM provider down, RAG pipeline broken |
| P3 - Medium | 4 hours | Partial degradation, some users affected | High error rate on specific endpoint |
| P4 - Low | 24 hours | Minor issue, cosmetic or non-urgent | Dashboard styling issue, non-critical log noise |

---

## Escalation Paths

1. **On-call engineer** → investigate and attempt resolution
2. **If unresolved after SLA** → escalate to team lead
3. **If P1 unresolved after 1 hour** → escalate to engineering manager

---

## Recovery Procedures

### Ingestion Pipeline Failure

**Symptoms**: Discord alert "Ingestion pipeline failures detected", documents in `knowledge/failed/`

1. Check failed documents: `ls knowledge/failed/`
2. Review error in logs: look for `Failed:` entries
3. Common causes:
   - **Parser error**: unsupported file format → move to `knowledge/raw_docs/` with correct extension
   - **Embedding failure**: API quota exceeded → wait for quota reset or switch provider
   - **Database error**: Supabase connection issue → check `SUPABASE_DIRECT_URL` connectivity
4. Fix root cause
5. Move files back to `knowledge/raw_docs/` for re-processing
6. Verify: check `knowledge/processed/` for successful ingestion

### LLM Provider Outage

**Symptoms**: 500 errors on `/v1/chat`, logs show `LLM_PROVIDER_ERROR`

1. Check provider status page (OpenRouter, Google, OpenAI)
2. If provider is down:
   - Switch to fallback provider in `infrastructure/container.py`
   - Uncomment alternative provider, comment out current
   - Restart server: `python server.py`
3. Verify: `curl -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" -d '{"query": "test"}'`

### Database Connectivity Issues

**Symptoms**: Health endpoint shows `db_connected: false`, 500 errors

1. Test direct connection: `psql $SUPABASE_DIRECT_URL -c "SELECT 1"`
2. Check Supabase project status dashboard
3. If connection pool exhausted:
   - Restart Supabase connection pooler (via Supabase dashboard)
   - Or restart the application to release stale connections
4. Verify: `curl http://localhost:8000/v1/health`

### Rate Limiting Issues

**Symptoms**: 429 responses on `/v1/chat` or `/v1/rag`

1. Check if rate limit is too low: `RATE_LIMIT_REQUESTS_PER_MINUTE` in `.env`
2. If legitimate traffic is being limited:
   - Increase `RATE_LIMIT_REQUESTS_PER_MINUTE`
   - Restart server
3. If under attack:
   - Keep rate limit low
   - Consider adding IP-based blocking at reverse proxy level

### Discord Webhook Failures

**Symptoms**: Logs show "Discord webhook failed"

1. Verify webhook URL is still valid in Discord server settings
2. Check if webhook rate limit is hit (Discord allows 5 requests per 5 seconds)
3. Update `DISCORD_WEBHOOK_URL` in `.env` if webhook was rotated
4. Test: send a test alert manually

---

## Post-Incident

After resolution:

1. Document the incident (what happened, root cause, resolution)
2. Update this runbook if new recovery steps were discovered
3. Consider adding monitoring/alerting to detect this class of issue earlier
4. Review: could this have been prevented?
