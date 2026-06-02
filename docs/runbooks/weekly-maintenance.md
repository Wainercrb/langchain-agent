# Weekly Maintenance Runbook

Run this checklist during the scheduled weekly maintenance window.

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
- [ ] Run VACUUM ANALYZE on high-write tables:
  ```sql
  VACUUM ANALYZE ingestion_logs;
  ```

---

## Ingestion Pipeline

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
