# Production Hardening Roadmap

## Executive Summary

This roadmap addresses all production readiness gaps identified in the langchain-agent system. Items are prioritized by **impact × likelihood** and organized into phases.

**Current State**: System works but has critical gaps in resilience (no circuit breaker, no failover), observability (metrics lost on restart, no log aggregation), and self-healing.

**Target State**: Production-grade system with automatic failure handling, persistent observability, and autonomous recovery.

**Total Effort**: ~40-60 hours across 4 phases

---

## Priority Matrix

| Priority | Item | Impact | Effort | Risk if Not Fixed |
|----------|------|--------|--------|-------------------|
| **P0** | Circuit Breaker | High | 4h | Cascading failures when LLM provider is down |
| **P0** | LLM Failover | High | 6h | Complete outage if primary LLM is unavailable |
| **P1** | Log Aggregation (CloudWatch) | High | 8h | No visibility into production issues |
| **P1** | Persistent Metrics | Medium | 4h | Metrics lost on restart, can't track trends |
| **P2** | Failed Document Retry | Medium | 6h | Failed ingestions sit forever, no notification |
| **P2** | Local Decision Audit Trail | Medium | 8h | Can't query AI decisions without LangSmith |
| **P3** | Full Self-Healing | Low | 20h+ | Diminishing returns, complex to implement |

---

## Phase 1: Critical Resilience (P0) — 10 hours

**Goal**: Prevent cascading failures and ensure availability when primary LLM is down.

### 1.1 Circuit Breaker Pattern

**Problem**: When OpenRouter is down, every request burns 3 retries × exponential backoff before failing. No "stop trying after N consecutive failures" pattern.

**Solution**: Implement a circuit breaker that:
- Tracks consecutive failures per LLM provider
- Opens the circuit after 5 consecutive failures
- Returns cached responses or fallback immediately when circuit is open
- Half-opens after 60 seconds to test if provider recovered
- Closes circuit after 2 consecutive successes

**Files to Create/Modify**:
- `agent/utils/circuit_breaker.py` (new)
- `agent/infrastructure/llm/openrouter.py` (wrap calls)
- `agent/infrastructure/container.py` (inject circuit breaker)

**Effort**: 4 hours  
**Risk**: Low — well-established pattern, easy to test

**Documentation**: See `CIRCUIT_BREAKER_FIX.md`

---

### 1.2 LLM Provider Failover

**Problem**: `container.py` hardcodes `OpenRouterProvider`. If OpenRouter is down, the entire system is down. No automatic fallback to Gemini or OpenAI despite having both configured.

**Solution**: Implement a failover chain:
- Primary: OpenRouter (current)
- Secondary: Gemini (already configured in settings)
- Tertiary: OpenAI (already configured in settings)
- Automatic failover when primary circuit is open
- Automatic failback when primary recovers

**Files to Create/Modify**:
- `agent/infrastructure/llm/failover_provider.py` (new)
- `agent/infrastructure/container.py` (wire failover chain)
- `agent/config/settings.py` (add failover config)

**Effort**: 6 hours  
**Risk**: Medium — need to test failover/failback scenarios

**Documentation**: See `LLM_FAILOVER_FIX.md`

---

## Phase 2: Observability (P1) — 12 hours

**Goal**: Persistent metrics and centralized logging for production visibility.

### 2.1 Log Aggregation to CloudWatch

**Problem**: Logs go to console or file. No structured JSON shipping to CloudWatch Logs. Can't set CloudWatch Alarms or search logs across restarts.

**Solution**: Ship structured JSON logs to CloudWatch Logs:
- Use `watchtower` library for CloudWatch Logs integration
- Structured JSON format with correlation_id, timestamp, level, message
- Separate log groups for: application, errors, access
- CloudWatch Alarms for error rate > 5% in 5 minutes

**Files to Create/Modify**:
- `agent/infrastructure/logging/cloudwatch.py` (new)
- `agent/infrastructure/logging/__init__.py` (register backend)
- `agent/config/settings.py` (add CloudWatch config)
- `agent/requirements.txt` (add watchtower)
- `docker-compose.yml` (add AWS credentials)

**Effort**: 8 hours  
**Risk**: Medium — need AWS permissions, test log shipping

**Documentation**: See `LOG_AGGREGATION_FIX.md`

---

### 2.2 Persistent Metrics (Redis or DynamoDB)

**Problem**: `SimpleMetrics` is in-memory only. Metrics reset to zero on every container restart. Can't track trends or set alerts on historical data.

**Solution**: Store metrics in Redis (or DynamoDB for serverless):
- Request count, error count, latency percentiles (p50, p95, p99)
- Time-series data with 1-minute granularity
- Retain for 30 days (configurable)
- Expose via `/v1/metrics` endpoint (backward compatible)

**Files to Create/Modify**:
- `agent/api/metrics.py` (replace SimpleMetrics with RedisMetrics)
- `agent/infrastructure/metrics/redis_metrics.py` (new)
- `agent/config/settings.py` (add Redis config)
- `agent/requirements.txt` (add redis)
- `docker-compose.yml` (add Redis service)

**Effort**: 4 hours  
**Risk**: Low — Redis is battle-tested, easy to implement

**Documentation**: See `PERSISTENT_METRICS_FIX.md`

---

## Phase 3: Data Quality (P2) — 14 hours

**Goal**: Ensure failed document ingestions are retried and AI decisions are auditable.

### 3.1 Failed Document Retry & Notification

**Problem**: Failed document ingestions go to `knowledge/failed/` directory. No automated retry, no notification, no dashboard showing ingestion health.

**Solution**: Implement retry queue with notifications:
- Failed documents are queued in database (not just filesystem)
- Automatic retry with exponential backoff (max 3 attempts)
- Discord alert after 3 failed attempts for same document
- `/v1/ingestion/health` endpoint showing success/failure rates
- Admin UI to view failed documents and manually retry

**Files to Create/Modify**:
- `agent/infrastructure/ingestion/retry_queue.py` (new)
- `agent/cronjob.py` (integrate retry queue)
- `agent/api/routes.py` (add ingestion health endpoint)
- `agent/models/ingestion.py` (new)
- Database migration for `ingestion_failures` table

**Effort**: 6 hours  
**Risk**: Medium — need database migration, test retry logic

**Documentation**: See `FAILED_DOCUMENT_RETRY_FIX.md`

---

### 3.2 Local Decision Audit Trail

**Problem**: AI decision logs (which tool was chosen, why) exist only in LangSmith. Can't query "show me all queries where agent chose web_search instead of search_documents" from your own system.

**Solution**: Store decision audit trail in database:
- Log every query with: timestamp, query text, tool chosen, run_id, latency, sources count
- Queryable via SQL (no LangSmith dependency)
- Retain for 90 days (configurable)
- `/v1/audit/decisions` endpoint for admin queries
- Optional: export to S3 for long-term archival

**Files to Create/Modify**:
- `agent/infrastructure/audit/decision_logger.py` (new)
- `agent/infrastructure/agent/tool_calling.py` (log decisions)
- `agent/api/routes.py` (add audit endpoint)
- `agent/models/audit.py` (new)
- Database migration for `decision_audit` table

**Effort**: 8 hours  
**Risk**: Low — additive feature, no breaking changes

**Documentation**: See `DECISION_AUDIT_TRAIL_FIX.md`

---

## Phase 4: Advanced Self-Healing (P3) — 20+ hours

**Goal**: Autonomous recovery from common failure scenarios.

**Note**: This phase has diminishing returns. Implement only if Phases 1-3 are stable and you have specific self-healing requirements.

### 4.1 Connection Pool Auto-Reset

**Problem**: If Supabase connection pool is exhausted, the system alerts humans but never self-heals.

**Solution**: Monitor connection pool health and auto-reset when exhausted:
- Track connection pool utilization
- If >90% for >60 seconds, automatically reset pool
- Log the reset event for post-mortem analysis

**Effort**: 4 hours  
**Risk**: High — connection pool resets can cause request failures

---

### 4.2 Graceful Degradation (Cached Responses)

**Problem**: When LLM is down, system returns 500 errors. No fallback to cached responses.

**Solution**: Cache recent successful responses and serve them when LLM is unavailable:
- Cache top 100 queries with their responses
- When circuit breaker is open, check cache for similar queries
- Return cached response with warning: "This is a cached response from [timestamp]"
- Cache TTL: 24 hours

**Effort**: 8 hours  
**Risk**: Medium — cache invalidation is hard, stale responses can mislead users

---

### 4.3 Automatic Document Re-ingestion

**Problem**: If vector store is temporarily unavailable during ingestion, documents are marked as failed but never retried automatically.

**Solution**: Implement automatic re-ingestion for transient failures:
- Distinguish between transient (network timeout) and permanent (invalid format) failures
- Transient failures: automatic retry with exponential backoff
- Permanent failures: alert admin, don't retry
- Dashboard showing retry attempts and success rates

**Effort**: 6 hours  
**Risk**: Low — retry logic is straightforward

---

### 4.4 Auto-Scaling Based on Load

**Problem**: System doesn't scale based on request volume. During traffic spikes, latency increases or requests fail.

**Solution**: Implement auto-scaling (AWS ECS or Kubernetes):
- Scale based on CPU utilization (>70% for 5 minutes)
- Scale based on request queue depth (>100 pending requests)
- Minimum 2 instances, maximum 10 instances
- Health check-based load balancing

**Effort**: 12+ hours  
**Risk**: High — infrastructure changes, need load testing

---

## Implementation Timeline

### Week 1: Phase 1 (Critical Resilience)
- Day 1-2: Circuit Breaker (4h)
- Day 3-4: LLM Failover (6h)
- Day 5: Testing and staging deployment

### Week 2: Phase 2 (Observability)
- Day 1-3: Log Aggregation (8h)
- Day 4-5: Persistent Metrics (4h)

### Week 3: Phase 3 (Data Quality)
- Day 1-2: Failed Document Retry (6h)
- Day 3-5: Decision Audit Trail (8h)

### Week 4+: Phase 4 (Self-Healing) — Optional
- Implement only if business requirements demand it
- Prioritize based on actual production incidents

---

## Success Metrics

After completing Phases 1-3, the system should achieve:

| Metric | Before | After |
|--------|--------|-------|
| **Availability** | 95% (downtime when LLM is down) | 99.5% (automatic failover) |
| **Mean Time to Recovery (MTTR)** | 30+ minutes (manual intervention) | <5 minutes (automatic) |
| **Error Visibility** | Low (logs lost on restart) | High (CloudWatch, alarms) |
| **Feedback Correlation** | 0% (run_id bug) | 100% (after run_id fix) |
| **Failed Document Recovery** | 0% (manual retry only) | 90% (automatic retry) |
| **Decision Auditability** | LangSmith only | LangSmith + local SQL |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Circuit breaker opens too aggressively | Medium | Medium | Tune thresholds based on production data |
| LLM failover causes inconsistent responses | Low | High | Test failover scenarios, log provider switches |
| CloudWatch costs spike | Low | Medium | Set log retention policies, filter noisy logs |
| Redis becomes single point of failure | Low | High | Use Redis Sentinel or AWS ElastiCache with replication |
| Decision audit trail grows too large | Medium | Low | Implement retention policy (90 days), archive to S3 |

---

## Dependencies

### Phase 1 Dependencies
- None (can start immediately)

### Phase 2 Dependencies
- AWS account with CloudWatch Logs permissions
- Redis instance (local or AWS ElastiCache)

### Phase 3 Dependencies
- Database migration capability
- Discord webhook URL (for failed document alerts)

### Phase 4 Dependencies
- AWS ECS or Kubernetes cluster
- Load testing infrastructure

---

## Cost Estimate

| Item | Monthly Cost (AWS) |
|------|-------------------|
| CloudWatch Logs (10GB/month) | $5 |
| Redis (t3.micro ElastiCache) | $15 |
| DynamoDB (if used instead of Redis) | $1-5 |
| S3 (for log archival) | $0.50 |
| **Total** | **$20-25/month** |

---

## Rollback Strategy

Each phase is independent and can be rolled back without affecting others:

- **Phase 1**: Revert circuit breaker and failover code, system returns to single-provider mode
- **Phase 2**: Revert logging changes, logs return to console/file
- **Phase 3**: Revert audit trail code, no data loss (audit table can be dropped)
- **Phase 4**: Revert self-healing code, system returns to manual recovery mode

---

## Documentation Index

### Phase 1: Critical Resilience
- `CIRCUIT_BREAKER_FIX.md` — Implementation guide for circuit breaker pattern
- `LLM_FAILOVER_FIX.md` — Implementation guide for LLM provider failover

### Phase 2: Observability
- `LOG_AGGREGATION_FIX.md` — Implementation guide for CloudWatch Logs integration
- `PERSISTENT_METRICS_FIX.md` — Implementation guide for Redis-backed metrics

### Phase 3: Data Quality
- `FAILED_DOCUMENT_RETRY_FIX.md` — Implementation guide for retry queue and notifications
- `DECISION_AUDIT_TRAIL_FIX.md` — Implementation guide for local decision audit trail

### Phase 4: Self-Healing (Optional)
- Documentation created on-demand if phase is approved

---

## Next Steps

1. **Review this roadmap** with stakeholders
2. **Approve Phase 1** (critical resilience) — highest impact, lowest risk
3. **Implement run_id fix** (already documented, 2 hours)
4. **Start Phase 1** — Circuit Breaker (4 hours)
5. **Deploy to staging** and test failover scenarios
6. **Deploy to production** and monitor for 1 week
7. **Proceed to Phase 2** if Phase 1 is stable

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Review
