# Production Readiness — Documentation Index

## Start Here

This is the complete production hardening plan for the langchain-agent system. It addresses all gaps identified in the production readiness assessment across three categories:

1. **Maintenance & Monitoring** (65% → 90%)
2. **AI Decision Logs** (75% → 95%)
3. **Automated Error-Handling** (65% → 90%)

---

## Quick Start: What to Fix First

### Immediate (This Week) — 2 hours

| # | Item | Doc | Effort |
|---|------|-----|--------|
| 1 | **Fix run_id bug** | `FIX_RUN_ID_PLAN.md` | 2h |

This is the highest-impact, lowest-risk fix. The feedback loop is completely broken without it.

### Critical (Next Week) — 3 hours

| # | Item | Doc | Effort |
|---|------|-----|--------|
| 2 | **Provider Chain** | `PROVIDER_CHAIN_FIX.md` | 3h |

Simple provider chain with single retry. Replaces complex circuit breaker + failover approach. Achieves all three goals (monitoring, decision logs, error handling) with 70% less code.

### High Priority (Week 3) — 12 hours

| # | Item | Doc | Effort |
|---|------|-----|--------|
| 4 | **Log Aggregation** | `PHASE_2_3_SUMMARIES.md` § P1.1 | 8h |
| 5 | **Persistent Metrics** | `PHASE_2_3_SUMMARIES.md` § P1.2 | 4h |

These give you production visibility — logs survive restarts, metrics persist, CloudWatch alarms work.

### Medium Priority (Week 4) — 22 hours

| # | Item | Doc | Effort |
|---|------|-----|--------|
| 6 | **Failed Document Retry** | `PHASE_2_3_SUMMARIES.md` § P2.1 | 6h |
| 7 | **Decision Audit Trail** | `PHASE_2_3_SUMMARIES.md` § P2.2 | 8h |
| 8 | **"Why" Logging** | `WHY_LOGGING_FIX.md` | 8h |

These improve data quality and make AI decisions queryable without LangSmith.

### Operations (Ongoing) — 12 hours

| # | Item | Doc | Effort |
|---|------|-----|--------|
| 9 | **Runbooks** | `RUNBOOKS.md` | 2h (review) |
| 10 | **Scheduled Maintenance** | `SCHEDULED_MAINTENANCE.md` | 10h |

Runbooks are critical for incident response. Scheduled maintenance keeps the system healthy long-term.

### Optional (Later) — 20+ hours

| # | Item | Doc | Effort |
|---|------|-----|--------|
| 8 | **Self-Healing** | `PRODUCTION_HARDENING_ROADMAP.md` § Phase 4 | 20h+ |

Diminishing returns. Only implement if Phases 1-3 are stable and business requires it.

---

## Documentation Map

```
langchain-agent/
├── PRODUCTION_READINESS_INDEX.md          ← You are here
├── PRODUCTION_HARDENING_ROADMAP.md        ← Master plan (all phases)
│
├── FIX_RUN_ID_PLAN.md                     ← P0: Fix feedback loop bug
├── LANGSMITH_FEEDBACK_ARCHITECTURE.md     ← How the feedback loop works
├── RUN_ID_FIX_QUICKREF.md                 ← Quick reference for run_id fix
│
├── PROVIDER_CHAIN_FIX.md                  ← P0: Simple provider chain (replaces circuit breaker + failover)
│
├── PHASE_2_3_SUMMARIES.md                 ← P1+P2: Observability & data quality
├── WHY_LOGGING_FIX.md                     ← P2: Capture LLM reasoning
│
├── RUNBOOKS.md                            ← Operations: Incident response procedures
└── SCHEDULED_MAINTENANCE.md               ← Operations: Recurring maintenance tasks
```

**Note**: `CIRCUIT_BREAKER_FIX.md` and `LLM_FAILOVER_FIX.md` are superseded by `PROVIDER_CHAIN_FIX.md`.

---

## Current State vs Target State

### Maintenance & Monitoring

| Capability | Current | After Phase 1-2 | After Operations |
|------------|---------|-----------------|------------------|
| Health checks | ✅ Docker healthchecks | ✅ Same | ✅ Same |
| Metrics endpoint | ✅ In-memory only | ✅ Redis-backed (persistent) | ✅ Same |
| Log aggregation | ❌ Console/file only | ✅ CloudWatch Logs | ✅ Same |
| CloudWatch Alarms | ❌ None | ✅ Error rate, circuit breaker | ✅ Same |
| Scheduled maintenance | ⚠️ Document ingestion only | ✅ + Failed document retry | ✅ + 7 maintenance tasks |
| Incident response | ❌ No runbooks | ❌ No runbooks | ✅ 7 runbooks |
| Security scanning | ❌ None | ❌ None | ✅ Weekly dependency scan |
| Key rotation | ❌ Manual | ❌ Manual | ✅ Quarterly reminders |

### AI Decision Logs

| Capability | Current | After Fixes |
|------------|---------|-------------|
| LangSmith traces | ✅ Working | ✅ Same |
| Feedback correlation | ❌ Broken (run_id bug) | ✅ Fixed |
| Correlation IDs | ✅ Working | ✅ Same |
| Local audit trail | ❌ LangSmith only | ✅ Database + SQL queries |
| Decision analytics | ❌ None | ✅ `/v1/audit/decisions` endpoint |
| LLM reasoning | ❌ Not captured | ✅ `reasoning` field in audit trail |
| Reasoning analysis | ❌ None | ✅ SQL queries on reasoning patterns |

### Automated Error-Handling

| Capability | Current | After Phase 1 |
|------------|---------|---------------|
| Exception hierarchy | ✅ Working | ✅ Same |
| Retry with backoff | ✅ Working (3 retries, exponential) | ✅ Simplified (1 retry, 1s backoff) |
| Discord alerts | ✅ Working | ✅ Same |
| Provider chain | ❌ Hardcoded single provider | ✅ 3-provider chain with failover |
| Failover time | N/A | ✅ <2 seconds (vs 14s in original plan) |
| Self-healing | ❌ Alert humans only | ⚠️ Phase 4 (optional) |

---

## Effort Summary

| Phase | Items | Effort | Priority |
|-------|-------|--------|----------|
| **Immediate** | run_id fix | 2h | P0 |
| **Phase 1** | Provider chain (replaces circuit breaker + failover) | 3h | P0 |
| **Phase 2** | Log aggregation + persistent metrics | 12h | P1 |
| **Phase 3** | Failed doc retry + decision audit + why logging | 22h | P2 |
| **Operations** | Runbooks + scheduled maintenance | 12h | P1 |
| **Phase 4** | Self-healing (optional) | 20h+ | P3 |
| **Total** | All phases | **71h+** | — |

**Note**: Simplified from 78h to 71h by replacing circuit breaker (4h) + LLM failover (6h) with provider chain (3h).

---

## Cost Impact

| Item | Monthly Cost |
|------|-------------|
| CloudWatch Logs (10GB) | $5 |
| Redis (t3.micro) | $15 |
| S3 (log archival) | $0.50 |
| **Total** | **~$20/month** |

LLM failover cost impact: negligible (backup providers used <1% of the time).

---

## Risk Matrix

| Fix | Risk | Rollback |
|-----|------|----------|
| run_id fix | Very Low | Revert 2 files |
| Provider chain | Low | Revert container.py to single provider |
| Log aggregation | Medium | Revert logging config, logs return to console |
| Persistent metrics | Low | Revert metrics.py, metrics return to in-memory |
| Failed doc retry | Medium | Revert cronjob, drop migration table |
| Decision audit | Low | Revert agent code, drop migration table |
| "Why" logging | Low | Revert agent code, drop column |
| Runbooks | None | Documentation only |
| Scheduled maintenance | Low | Remove maintenance service from docker-compose |

All fixes are independent and can be rolled back individually.

**Note**: Provider chain is lower risk than the original circuit breaker + failover approach (simpler logic, easier to test, faster rollback).

---

## Verification Checklist

After each phase, verify:

### Phase 0 (run_id fix)
- [ ] `run_id` in `/v1/chat` matches LangSmith trace ID
- [ ] Feedback appears in LangSmith dashboard

### Phase 1 (Provider chain)
- [ ] Provider chain tries providers in order
- [ ] Single retry on transient errors (1s backoff)
- [ ] Immediate failover on permanent errors
- [ ] Failover completes within 2 seconds
- [ ] `/v1/providers` shows all configured providers
- [ ] Logs show which provider succeeded/failed

### Phase 2 (Observability)
- [ ] Logs appear in CloudWatch Logs
- [ ] Metrics survive container restart
- [ ] CloudWatch Alarms fire on error rate > 5%

### Phase 3 (Data quality)
- [ ] Failed documents are retried automatically
- [ ] Discord alert after 3 failed retries
- [ ] `/v1/audit/decisions` returns decision history
- [ ] SQL queries work on `decision_audit` table
- [ ] LLM reasoning is captured in `reasoning` field
- [ ] `/v1/audit/decisions?include_reasoning=true` returns reasoning text

### Operations (Runbooks + Maintenance)
- [ ] Runbooks reviewed by on-call team
- [ ] Maintenance scheduler deployed
- [ ] Vector store health check runs daily
- [ ] Security scan runs weekly
- [ ] Log rotation runs daily
- [ ] Database vacuum runs weekly
- [ ] Test runbook: simulate circuit breaker open, follow Runbook 1
- [ ] Test runbook: simulate database failure, follow Runbook 3

---

## Next Steps

1. **Review this index** and the master roadmap
2. **Approve Phase 0** (run_id fix) — 2 hours, highest impact
3. **Implement run_id fix** using `FIX_RUN_ID_PLAN.md`
4. **Deploy and verify** in staging → production
5. **Approve Phase 1** (provider chain) — 3 hours
6. **Implement provider chain** using `PROVIDER_CHAIN_FIX.md`
7. **Test failover** by blocking primary provider
8. **Continue to Phase 2-3** as resources allow

**Key Decision**: Phase 1 uses a simplified provider chain (3h) instead of circuit breaker + failover (10h). This achieves all three goals with 70% less code and faster failover times.

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Review
