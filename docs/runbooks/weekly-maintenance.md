# Weekly Maintenance Runbook

Run this checklist during the scheduled weekly maintenance window.

> **Automation status:** Items marked `[AUTO]` are scheduled by `agent/cronjob.py`
> and run without human intervention. Items marked `[MANUAL]` need a human
> to execute and review. Ingestion itself is automated by cronjob's
> ingestion cycle (every 5 minutes by default).
>
> | Job | Cadence | Implementation |
> |---|---|---|
> | Weekly database backup | Sundays 02:00 | `cronjob.py:_run_backup_cycle` → `scripts/backup.py` |
> | Weekly VACUUM ANALYZE | Sundays 04:00 | `cronjob.py:_run_vacuum_analyze` → `psql` |
> | Daily log rotation check | 03:00 daily | `cronjob.py:_run_log_rotation_check` |
>
> Each automated job dispatches an ERROR alert on failure. Tune via
> `MAINTENANCE_*` env vars (see `agent/.env.example`).

---

## Log Review

- [ ] Review application logs for recurring errors:
  ```bash
  grep "ERROR" logs/*.log | sort | uniq -c | sort -rn | head -20
  ```
- [ ] Check for warnings that may indicate emerging issues:
  ```bash
  grep "WARNING" logs/*.log | sort | uniq -c | sort -rn | head -10
  ```
- [ ] Review Discord alert history for patterns (same alert firing repeatedly)

---

## Metrics Review

- [ ] Check request volume trends:
  ```bash
  curl -s http://localhost:8000/v1/metrics | python -m json.tool
  ```
- [ ] Compare error rate to previous week (should be < 5%)
- [ ] Review average latency trends (should be stable or improving)
- [ ] Check dashboard at `http://localhost:4321/dashboard` for visual indicators

---

## Database Health

- [ ] **[AUTO]** Run automated backup script (Sundays 02:00):
  ```bash
  cd agent && python scripts/backup.py --retention 7
  ```
  - Verify backup file created in `backups/`
  - Check backup size is reasonable (not 0 bytes)
  - Review backup logs for warnings
- [ ] Check table sizes:
  ```sql
  SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
  FROM pg_catalog.pg_statio_user_tables
  ORDER BY pg_total_relation_size(relid) DESC;
  ```
- [ ] Review index usage (identify unused indexes):
  ```sql
  SELECT schemaname, tablename, indexname, idx_scan
  FROM pg_stat_user_indexes
  ORDER BY idx_scan ASC;
  ```
- [ ] **[AUTO]** Run VACUUM ANALYZE on high-write tables (Sundays 04:00):
  ```sql
  VACUUM ANALYZE ingestion_logs;
  VACUUM ANALYZE documents;
  VACUUM ANALYZE document_chunks;
  ```

---

## Ingestion Pipeline

> Ingestion cycle itself runs automatically every 5 minutes via
> `agent/cronjob.py` (the ingestion scheduler) with automatic failure
> alerting on `_alert_on_failures` with a 1h cooldown.

- [ ] Check for stuck files in `knowledge/raw_docs/` (older than expected cycle time)
- [ ] Review `knowledge/failed/` for files that need manual intervention
- [ ] Verify processed file count matches expectations:
  ```bash
  ls knowledge/processed/ | wc -l
  ```

---

## Security & Access

- [ ] Review API keys rotation schedule
- [ ] Check Discord webhook URL is still active
- [ ] Verify no unauthorized access in logs (unusual IPs, patterns)

---

## Documentation

- [ ] Update this runbook if new maintenance tasks were discovered
- [ ] Review incident response runbook for accuracy
- [ ] Update deployment checklist if new env vars or dependencies were added
