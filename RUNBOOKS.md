# Production Runbooks

Step-by-step procedures for responding to alerts and incidents. Each runbook assumes you received a Discord alert and need to diagnose and resolve the issue.

---

## How to Use This Document

1. **Identify the alert** from Discord (severity, message, error type)
2. **Find the matching runbook** below
3. **Follow the diagnostic steps** in order
4. **Apply the resolution** that matches your findings
5. **Verify** the issue is resolved
6. **Document** the incident in your incident log

---

## Runbook 1: Circuit Breaker Open

### Alert Signature
```
[ERROR] Circuit breaker 'openrouter': CLOSED → OPEN (5 consecutive failures)
```

### Severity: WARNING (if failover working) / CRITICAL (if all providers down)

### Diagnostic Steps

1. **Check provider status**
   ```bash
   curl -s http://localhost:8000/v1/providers | jq
   ```
   Expected output:
   ```json
   {
     "openrouter": {"state": "open", "failure_count": 5, "active": false},
     "gemini": {"state": "closed", "failure_count": 0, "active": true},
     "openai": {"state": "closed", "failure_count": 0, "active": false}
   }
   ```

2. **Check if failover is working**
   ```bash
   curl -X POST http://localhost:8000/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "top_k": 1}'
   ```
   - ✅ If response comes back → failover is working, severity is WARNING
   - ❌ If 503 error → all providers down, severity is CRITICAL

3. **Check OpenRouter status page**
   - Visit: https://status.openrouter.ai/
   - Look for ongoing incidents or maintenance

4. **Check application logs**
   ```bash
   docker logs langchain-agent --since 10m 2>&1 | grep -i "circuit\|failover\|openrouter"
   ```
   Look for:
   - "Failover: switched from openrouter to gemini" → failover working
   - "Failover: all providers failed" → critical issue

### Resolution

#### If failover is working (WARNING)
1. **Monitor** — system is self-healing
2. **Wait** for circuit breaker to half-open (60 seconds)
3. **Verify recovery**:
   ```bash
   # After 60 seconds, check if circuit closed
   curl -s http://localhost:8000/v1/providers | jq '.openrouter.state'
   ```
   Expected: `"closed"`

4. **If circuit stays open >10 minutes**:
   - Check OpenRouter status page again
   - If OpenRouter is down for extended period, no action needed (failover is handling it)
   - Document incident and move on

#### If all providers down (CRITICAL)
1. **Check all provider status pages**:
   - OpenRouter: https://status.openrouter.ai/
   - Google AI: https://status.cloud.google.com/
   - OpenAI: https://status.openai.com/

2. **Check network connectivity**:
   ```bash
   docker exec langchain-agent curl -I https://api.openai.com
   docker exec langchain-agent curl -I https://generativelanguage.googleapis.com
   ```

3. **Check API keys**:
   ```bash
   docker exec langchain-agent env | grep -E "OPENROUTER_API_KEY|GOOGLE_API_KEY|OPENAI_API_KEY"
   ```
   - If keys are missing or invalid, update `.env` and restart:
     ```bash
     docker-compose restart agent
     ```

4. **If all providers are down due to external outage**:
   - No immediate fix possible
   - Notify users via status page or banner
   - Monitor provider status pages
   - System will auto-recover when providers come back online

### Escalation
- If all providers down >30 minutes → escalate to engineering lead
- If circuit breaker opens repeatedly (3+ times in 1 hour) → investigate root cause

---

## Runbook 2: LLM Provider Timeout

### Alert Signature
```
[ERROR] Chat processing failed: LLM_TRANSIENT_ERROR: timeout after 60s
```

### Severity: WARNING (transient) / ERROR (persistent)

### Diagnostic Steps

1. **Check if it's a one-off or pattern**
   ```bash
   docker logs langchain-agent --since 1h 2>&1 | grep -c "timeout"
   ```
   - 1-2 timeouts → transient, likely network blip
   - 10+ timeouts → persistent issue

2. **Check average latency**
   ```bash
   curl -s http://localhost:8000/v1/metrics | jq '.avg_latency_ms'
   ```
   - <5000ms → normal
   - >10000ms → provider is slow

3. **Check which provider is timing out**
   ```bash
   docker logs langchain-agent --since 1h 2>&1 | grep -E "openrouter|gemini|openai" | tail -20
   ```

4. **Check provider status page** (see Runbook 1)

### Resolution

#### If transient (1-2 timeouts)
1. **No action needed** — retry logic handles it
2. **Monitor** for recurrence

#### If persistent (10+ timeouts)
1. **Increase timeout** (temporary fix):
   ```bash
   # Edit .env
   LLM_TIMEOUT_SECONDS=120
   
   # Restart
   docker-compose restart agent
   ```

2. **Switch to faster provider** (if available):
   ```bash
   # Edit .env to prioritize faster provider
   FAILOVER_PROVIDERS=gemini,openrouter,openai
   
   # Restart
   docker-compose restart agent
   ```

3. **If provider is consistently slow**:
   - Check if you're using a slower model (e.g., GPT-4 vs GPT-4o-mini)
   - Consider switching to a faster model for non-critical queries
   - Document in incident log for capacity planning

### Escalation
- If timeouts persist >1 hour → escalate to engineering lead
- If latency >30s consistently → investigate model/provider choice

---

## Runbook 3: Database Connection Failure

### Alert Signature
```
[ERROR] Chat processing failed: DOCUMENT_STORE_ERROR: connection refused
```

### Severity: ERROR

### Diagnostic Steps

1. **Check health endpoint**
   ```bash
   curl -s http://localhost:8000/v1/health | jq
   ```
   Expected:
   ```json
   {
     "status": "error",
     "db_connected": false,
     "llm_connected": true,
     "embedding_connected": true
   }
   ```

2. **Check Supabase status**
   - Visit: https://status.supabase.com/
   - Look for ongoing incidents

3. **Check database connectivity**
   ```bash
   docker exec langchain-agent python -c "
   from supabase import create_client
   from config import settings
   client = create_client(settings.supabase_url, settings.supabase_key)
   print(client.table('documents').select('id').limit(1).execute())
   "
   ```

4. **Check connection pool**
   ```bash
   docker exec langchain-agent python -c "
   from services.container import db_session
   print(f'Active connections: {db_session.connection().connection.info}')
   "
   ```

### Resolution

#### If Supabase is down (external outage)
1. **No immediate fix** — wait for Supabase to recover
2. **Monitor** status page
3. **Notify users** if extended outage
4. **System will auto-recover** when Supabase comes back

#### If connection pool exhausted
1. **Restart container** to reset pool:
   ```bash
   docker-compose restart agent
   ```

2. **If it happens repeatedly**, increase pool size:
   ```python
   # In infrastructure/container.py
   from sqlalchemy import create_engine
   from sqlalchemy.orm import sessionmaker
   
   engine = create_engine(
       settings.supabase_direct_url,
       pool_size=20,  # Increase from default (5)
       max_overflow=10,
   )
   ```

3. **Investigate root cause**:
   - Check for connection leaks (sessions not closed)
   - Check for long-running queries
   - Check for traffic spikes

#### If connection refused (network issue)
1. **Check network**:
   ```bash
   docker exec langchain-agent ping -c 3 db.<project>.supabase.co
   ```

2. **Check firewall/security groups** (if on AWS)

3. **Restart container**:
   ```bash
   docker-compose restart agent
   ```

### Escalation
- If database down >15 minutes → escalate to engineering lead
- If connection pool issues recur → investigate application code for leaks

---

## Runbook 4: High Error Rate

### Alert Signature
```
[CRITICAL] Error rate >5% in last 5 minutes (current: 12%)
```

### Severity: CRITICAL

### Diagnostic Steps

1. **Check error count and rate**
   ```bash
   curl -s http://localhost:8000/v1/metrics | jq
   ```
   Calculate error rate: `error_count / request_count * 100`

2. **Identify error pattern**
   ```bash
   docker logs langchain-agent --since 10m 2>&1 | grep -E "ERROR|CRITICAL" | tail -50
   ```
   Look for:
   - Same error message repeated → single root cause
   - Different errors → multiple issues or cascading failure

3. **Check recent deployments**
   ```bash
   git log --oneline --since="1 hour ago"
   ```
   - If recent deployment → possible regression

4. **Check resource usage**
   ```bash
   docker stats langchain-agent --no-stream
   ```
   - CPU >90% → overload
   - Memory >90% → memory leak or insufficient resources

### Resolution

#### If caused by recent deployment
1. **Rollback immediately**:
   ```bash
   git revert HEAD
   docker-compose up -d --build agent
   ```

2. **Verify rollback**:
   ```bash
   curl -s http://localhost:8000/v1/health | jq
   ```

3. **Investigate** the failed deployment in staging

#### If caused by external service outage
1. **Identify which service** (LLM provider, database, etc.)
2. **Follow the relevant runbook** (Runbook 1, 2, or 3)
3. **Monitor** until service recovers

#### If caused by resource exhaustion
1. **Scale up** (if on AWS ECS):
   ```bash
   aws ecs update-service \
     --cluster langchain-cluster \
     --service langchain-agent \
     --desired-count 4  # Increase from 2
   ```

2. **Restart container** (temporary fix):
   ```bash
   docker-compose restart agent
   ```

3. **Investigate root cause**:
   - Memory leak? Check for unclosed resources
   - CPU spike? Check for inefficient queries or infinite loops
   - Traffic spike? Check access logs for unusual patterns

#### If unknown cause
1. **Collect logs**:
   ```bash
   docker logs langchain-agent --since 1h > /tmp/agent-logs.txt
   ```

2. **Restart container** (may clear transient issue):
   ```bash
   docker-compose restart agent
   ```

3. **Escalate** to engineering lead with logs

### Escalation
- **Immediate** if error rate >20%
- **Within 15 minutes** if error rate >10%
- **Within 1 hour** if error rate >5%

---

## Runbook 5: Failed Document Ingestion Spike

### Alert Signature
```
[ERROR] Document ingestion permanently failed: document.pdf (3 retries exhausted)
```

### Severity: WARNING (single document) / ERROR (multiple documents)

### Diagnostic Steps

1. **Check ingestion health**
   ```bash
   curl -s http://localhost:8000/v1/ingestion/health | jq
   ```
   Expected:
   ```json
   {
     "success_count": 45,
     "failed_count": 5,
     "pending_retry": 2,
     "success_rate": 0.90
   }
   ```

2. **Check failed documents**
   ```bash
   ls -la agent/knowledge/failed/
   ```

3. **Check error messages**
   ```bash
   docker logs langchain-agent --since 1h 2>&1 | grep -i "ingestion.*failed"
   ```
   Look for:
   - "Invalid PDF format" → document corruption
   - "Embedding API error" → external service issue
   - "Database timeout" → vector store issue

4. **Check database retry queue**
   ```bash
   docker exec langchain-agent python -c "
   from services.container import db_session
   from models.ingestion import IngestionFailure
   failures = db_session.query(IngestionFailure).filter(
       IngestionFailure.status == 'failed'
   ).all()
   for f in failures:
       print(f'{f.filename}: {f.error_message}')
   "
   ```

### Resolution

#### If single document failed
1. **Check document format**:
   ```bash
   file agent/knowledge/failed/document.pdf
   ```
   - If not a valid PDF → request corrected document from uploader

2. **Manual retry** (if transient error):
   ```bash
   mv agent/knowledge/failed/document.pdf agent/knowledge/raw_docs/
   ```
   Next cronjob will retry

#### If multiple documents failed (same error)
1. **Identify common pattern**:
   - Same uploader? → check their document preparation process
   - Same file type? → check if parser supports that format
   - Same time window? → check if external service was down

2. **Fix root cause**:
   - If parser issue → update parser or convert documents
   - If external service → wait for recovery, then retry
   - If uploader issue → provide guidance on document format

3. **Bulk retry**:
   ```bash
   mv agent/knowledge/failed/*.pdf agent/knowledge/raw_docs/
   ```

#### If embedding API is down
1. **Check Google AI status**: https://status.cloud.google.com/
2. **Wait for recovery**
3. **Retry failed documents**:
   ```bash
   mv agent/knowledge/failed/*.pdf agent/knowledge/raw_docs/
   ```

### Escalation
- If >10 documents fail in 1 hour → escalate to engineering lead
- If same document fails 3+ times → investigate document format

---

## Runbook 6: Health Check Failing

### Alert Signature
```
Docker healthcheck failed (3/3 retries)
```

### Severity: CRITICAL

### Diagnostic Steps

1. **Check container status**
   ```bash
   docker ps | grep langchain-agent
   ```
   - If "unhealthy" → container is running but health check failing
   - If not listed → container crashed

2. **Check health endpoint manually**
   ```bash
   curl -v http://localhost:8000/v1/health
   ```
   - If timeout → application hung
   - If 500 error → application error
   - If connection refused → application not running

3. **Check container logs**
   ```bash
   docker logs langchain-agent --tail 100
   ```
   Look for:
   - Stack traces → application crash
   - "Out of memory" → resource exhaustion
   - "Address already in use" → port conflict

4. **Check resource usage**
   ```bash
   docker stats langchain-agent --no-stream
   ```

### Resolution

#### If container crashed
1. **Check exit code**:
   ```bash
   docker inspect langchain-agent --format='{{.State.ExitCode}}'
   ```
   - 0 → clean exit (unlikely)
   - 1 → application error
   - 137 → OOM killed
   - 139 → segfault

2. **Restart container**:
   ```bash
   docker-compose up -d agent
   ```

3. **If OOM killed (137)**, increase memory limit:
   ```yaml
   # In docker-compose.yml
   services:
     agent:
       deploy:
         resources:
           limits:
             memory: 2G  # Increase from 1G
   ```

#### If health check timing out
1. **Check if application is hung**:
   ```bash
   docker exec langchain-agent ps aux
   ```

2. **Restart container**:
   ```bash
   docker-compose restart agent
   ```

3. **If it happens repeatedly**, investigate:
   - Deadlocks in application code
   - Database connection pool exhaustion
   - Infinite loops

#### If port conflict
1. **Check what's using port 8000**:
   ```bash
   lsof -i :8000
   ```

2. **Kill conflicting process** or change port in `docker-compose.yml`

### Escalation
- **Immediate** if container won't start
- **Within 15 minutes** if health checks fail repeatedly

---

## Runbook 7: Discord Alert Not Firing

### Alert Signature
(No alert received, but errors are occurring)

### Severity: WARNING

### Diagnostic Steps

1. **Check if alerts are enabled**
   ```bash
   docker exec langchain-agent env | grep DISCORD_WEBHOOK_URL
   ```
   - If empty → alerts not configured

2. **Check alert service logs**
   ```bash
   docker logs langchain-agent --since 1h 2>&1 | grep -i "discord\|alert"
   ```
   Look for:
   - "Discord webhook failed" → webhook URL invalid or Discord down
   - "Alert rate-limited" → too many alerts, being throttled
   - "Alert dedup'd" → same alert sent recently

3. **Test webhook manually**
   ```bash
   curl -X POST "$DISCORD_WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d '{"content": "Test alert"}'
   ```
   - If 204 → webhook working
   - If 404 → webhook URL invalid
   - If timeout → Discord down

### Resolution

#### If webhook URL invalid
1. **Get new webhook URL** from Discord:
   - Server Settings → Integrations → Webhooks → New Webhook
   - Copy webhook URL

2. **Update `.env`**:
   ```bash
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

3. **Restart**:
   ```bash
   docker-compose restart agent
   ```

#### If rate-limited
1. **Increase rate limit** (if needed):
   ```bash
   # Edit .env
   ALERT_RATE_LIMIT_PER_MINUTE=10
   ```

2. **Restart**:
   ```bash
   docker-compose restart agent
   ```

#### If Discord is down
1. **Check Discord status**: https://discordstatus.com/
2. **Wait for recovery**
3. **Alerts will queue and send when Discord recovers**

### Escalation
- If alerts not firing for >1 hour → escalate to engineering lead
- Critical: you're flying blind without alerts

---

## General Troubleshooting Commands

### Check all services
```bash
docker-compose ps
```

### View recent logs
```bash
docker logs langchain-agent --since 1h --tail 100
```

### Check environment variables
```bash
docker exec langchain-agent env | grep -E "API_KEY|URL|ENABLE"
```

### Restart all services
```bash
docker-compose restart
```

### Rebuild and restart
```bash
docker-compose up -d --build
```

### Check disk space
```bash
df -h
docker system df
```

### Clean up Docker
```bash
docker system prune -f
```

---

## Incident Documentation Template

After resolving an incident, document it:

```markdown
## Incident: [Brief description]

**Date**: YYYY-MM-DD HH:MM UTC
**Duration**: X minutes
**Severity**: WARNING / ERROR / CRITICAL
**Alert**: [Copy alert message from Discord]

### Impact
- [What was affected? Users? Data? Performance?]
- [How many users impacted?]

### Root Cause
- [What caused the incident?]
- [Why did it happen?]

### Resolution
- [What steps were taken to resolve?]
- [Which runbook was followed?]

### Prevention
- [What can be done to prevent recurrence?]
- [Any code changes needed?]
- [Any monitoring improvements?]

### Timeline
- HH:MM - Alert received
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Resolution applied
- HH:MM - Incident resolved
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Production Ready
