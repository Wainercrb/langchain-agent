# Scheduled Maintenance Tasks

## Problem

The current `cronjob.py` only handles document ingestion. Production systems need additional scheduled tasks for health monitoring, cleanup, archival, and security.

---

## Required Maintenance Tasks

| Task | Frequency | Effort | Priority |
|------|-----------|--------|----------|
| Vector store health check | Daily | 2h | High |
| Stale session cleanup | Daily | 1h | Medium |
| Metrics archival to S3 | Monthly | 3h | Medium |
| Dependency security scan | Weekly | 1h | High |
| API key rotation reminders | Quarterly | 0.5h | Low |
| Log rotation | Daily | 1h | Medium |
| Database vacuum/analyze | Weekly | 1h | Medium |

---

## Task 1: Vector Store Health Check (Daily)

### Purpose

Verify that the vector store (Supabase/pgvector) is healthy, indexes are not corrupted, and document counts are consistent.

### Implementation

**File**: `agent/maintenance/vector_store_health.py` (new)

```python
"""Daily vector store health check."""

from datetime import datetime
from sqlalchemy import text

from services.container import db_session, vector_store, alert_service
from services.logging import logger
from utils.exceptions import Severity


def check_vector_store_health():
    """Run comprehensive vector store health checks."""
    issues = []
    
    # Check 1: Database connectivity
    try:
        result = db_session.execute(text("SELECT 1")).scalar()
        if result != 1:
            issues.append("Database connectivity check failed")
    except Exception as e:
        issues.append(f"Database connection error: {str(e)}")
    
    # Check 2: Document count consistency
    try:
        doc_count = db_session.execute(
            text("SELECT COUNT(*) FROM documents")
        ).scalar()
        
        chunk_count = db_session.execute(
            text("SELECT COUNT(*) FROM document_chunks")
        ).scalar()
        
        logger.info(f"Vector store: {doc_count} documents, {chunk_count} chunks")
        
        # Sanity check: chunks should be > documents
        if chunk_count < doc_count:
            issues.append(
                f"Chunk count ({chunk_count}) is less than document count ({doc_count})"
            )
        
        # Check for orphaned chunks (chunks without parent document)
        orphaned = db_session.execute(text("""
            SELECT COUNT(*) FROM document_chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE d.id IS NULL
        """)).scalar()
        
        if orphaned > 0:
            issues.append(f"Found {orphaned} orphaned chunks (no parent document)")
    
    except Exception as e:
        issues.append(f"Document count check failed: {str(e)}")
    
    # Check 3: Index health
    try:
        # Check if embedding index exists and is valid
        index_check = db_session.execute(text("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'document_chunks' 
              AND indexname LIKE '%embedding%'
        """)).fetchall()
        
        if not index_check:
            issues.append("Embedding index not found — vector search will be slow")
        
        logger.info(f"Vector store indexes: {len(index_check)} found")
    
    except Exception as e:
        issues.append(f"Index check failed: {str(e)}")
    
    # Check 4: Recent ingestion activity
    try:
        recent_count = db_session.execute(text("""
            SELECT COUNT(*) FROM documents 
            WHERE created_at > NOW() - INTERVAL '7 days'
        """)).scalar()
        
        logger.info(f"Documents ingested in last 7 days: {recent_count}")
        
        # Alert if no documents ingested in 7 days (might indicate cronjob failure)
        if recent_count == 0:
            issues.append("No documents ingested in last 7 days — check cronjob")
    
    except Exception as e:
        issues.append(f"Recent activity check failed: {str(e)}")
    
    # Report results
    if issues:
        logger.error(f"Vector store health check failed: {len(issues)} issues")
        for issue in issues:
            logger.error(f"  - {issue}")
        
        # Send alert
        import asyncio
        asyncio.run(alert_service.send_alert(
            severity=Severity.WARNING,
            message=f"Vector store health check: {len(issues)} issues found",
            metadata={"issues": "; ".join(issues[:5])},  # First 5 issues
        ))
    else:
        logger.info("Vector store health check passed")
    
    return {
        "status": "healthy" if not issues else "degraded",
        "issues": issues,
        "checked_at": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    result = check_vector_store_health()
    print(result)
```

### Scheduling

**File**: `agent/maintenance/scheduler.py` (new)

```python
"""Simple scheduler for maintenance tasks."""

import schedule
import time
from datetime import datetime

from maintenance.vector_store_health import check_vector_store_health
from maintenance.session_cleanup import cleanup_stale_sessions
from maintenance.log_rotation import rotate_logs
from maintenance.db_vacuum import vacuum_database
from services.logging import logger


def run_daily_tasks():
    """Run all daily maintenance tasks."""
    logger.info("Running daily maintenance tasks")
    
    try:
        check_vector_store_health()
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
    
    try:
        cleanup_stale_sessions()
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
    
    try:
        rotate_logs()
    except Exception as e:
        logger.error(f"Log rotation failed: {e}")
    
    logger.info("Daily maintenance tasks complete")


def run_weekly_tasks():
    """Run all weekly maintenance tasks."""
    logger.info("Running weekly maintenance tasks")
    
    try:
        vacuum_database()
    except Exception as e:
        logger.error(f"Database vacuum failed: {e}")
    
    logger.info("Weekly maintenance tasks complete")


def start_scheduler():
    """Start the maintenance scheduler."""
    # Daily at 3 AM UTC
    schedule.every().day.at("03:00").do(run_daily_tasks)
    
    # Weekly on Sunday at 4 AM UTC
    schedule.every().sunday.at("04:00").do(run_weekly_tasks)
    
    logger.info("Maintenance scheduler started")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    start_scheduler()
```

### Docker Integration

**File**: `docker-compose.yml` (add maintenance service)

```yaml
services:
  # ... existing services ...
  
  maintenance:
    build:
      context: ./agent
      dockerfile: Dockerfile
    container_name: langchain-maintenance
    command: python -m maintenance.scheduler
    env_file:
      - ./agent/.env
    networks:
      - langchain-network
    restart: unless-stopped
```

---

## Task 2: Stale Session Cleanup (Daily)

### Purpose

Clean up old session data, temporary files, and expired cache entries.

### Implementation

**File**: `agent/maintenance/session_cleanup.py` (new)

```python
"""Clean up stale sessions and temporary data."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from services.container import db_session
from services.logging import logger


def cleanup_stale_sessions():
    """Remove session data older than 30 days."""
    cutoff = datetime.utcnow() - timedelta(days=30)
    
    # Clean up old decision audit records (keep 90 days)
    try:
        audit_cutoff = datetime.utcnow() - timedelta(days=90)
        deleted = db_session.execute(f"""
            DELETE FROM decision_audit 
            WHERE created_at < '{audit_cutoff.isoformat()}'
        """)
        db_session.commit()
        logger.info(f"Deleted {deleted.rowcount} old decision audit records")
    except Exception as e:
        logger.error(f"Failed to clean up decision audit: {e}")
    
    # Clean up old ingestion failure records (keep 30 days)
    try:
        deleted = db_session.execute(f"""
            DELETE FROM ingestion_failures 
            WHERE created_at < '{cutoff.isoformat()}'
              AND status IN ('success', 'failed')
        """)
        db_session.commit()
        logger.info(f"Deleted {deleted.rowcount} old ingestion failure records")
    except Exception as e:
        logger.error(f"Failed to clean up ingestion failures: {e}")
    
    # Clean up temporary files
    try:
        temp_dir = Path("/tmp/langchain-agent")
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                if file.stat().st_mtime < cutoff.timestamp():
                    file.unlink()
                    logger.debug(f"Deleted temp file: {file}")
    except Exception as e:
        logger.error(f"Failed to clean up temp files: {e}")
    
    logger.info("Stale session cleanup complete")


if __name__ == "__main__":
    cleanup_stale_sessions()
```

---

## Task 3: Metrics Archival to S3 (Monthly)

### Purpose

Archive historical metrics data to S3 for long-term storage and analysis.

### Implementation

**File**: `agent/maintenance/metrics_archival.py` (new)

```python
"""Archive metrics to S3 for long-term storage."""

import json
import boto3
from datetime import datetime, timedelta
from pathlib import Path

from services.container import db_session
from services.logging import logger
from config import settings


def archive_metrics_to_s3():
    """Archive last month's metrics to S3."""
    # Calculate date range (last month)
    today = datetime.utcnow()
    first_day = today.replace(day=1)
    last_month_end = first_day - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    logger.info(
        f"Archiving metrics from {last_month_start.date()} to {last_month_end.date()}"
    )
    
    # Query metrics for the month
    try:
        # This assumes you have a metrics_history table or similar
        # Adjust based on your actual metrics storage
        metrics = db_session.execute(f"""
            SELECT 
                DATE_TRUNC('day', created_at) as date,
                COUNT(*) as request_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                AVG(latency_ms) as avg_latency_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms
            FROM request_logs
            WHERE created_at >= '{last_month_start.isoformat()}'
              AND created_at < '{first_day.isoformat()}'
            GROUP BY DATE_TRUNC('day', created_at)
            ORDER BY date
        """).fetchall()
        
        if not metrics:
            logger.info("No metrics to archive")
            return
        
        # Convert to JSON
        metrics_data = [
            {
                "date": row[0].isoformat(),
                "request_count": row[1],
                "error_count": row[2],
                "avg_latency_ms": float(row[3]) if row[3] else 0,
                "p95_latency_ms": float(row[4]) if row[4] else 0,
            }
            for row in metrics
        ]
        
        # Upload to S3
        s3 = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        
        bucket = settings.metrics_s3_bucket or "langchain-agent-metrics"
        key = f"metrics/{last_month_start.strftime('%Y/%m')}/metrics.json"
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metrics_data, indent=2),
            ContentType='application/json',
        )
        
        logger.info(f"Archived {len(metrics_data)} days of metrics to s3://{bucket}/{key}")
        
        # Optionally delete old metrics from database
        # db_session.execute(f"""
        #     DELETE FROM request_logs 
        #     WHERE created_at < '{first_day.isoformat()}'
        # """)
        # db_session.commit()
        
    except Exception as e:
        logger.error(f"Failed to archive metrics: {e}")
        raise


if __name__ == "__main__":
    archive_metrics_to_s3()
```

### Scheduling

Add to `scheduler.py`:

```python
from maintenance.metrics_archival import archive_metrics_to_s3

def run_monthly_tasks():
    """Run all monthly maintenance tasks."""
    logger.info("Running monthly maintenance tasks")
    
    try:
        archive_metrics_to_s3()
    except Exception as e:
        logger.error(f"Metrics archival failed: {e}")
    
    logger.info("Monthly maintenance tasks complete")

# In start_scheduler():
schedule.every().day.at("05:00").do(
    lambda: datetime.utcnow().day == 1 and run_monthly_tasks()
)
```

---

## Task 4: Dependency Security Scan (Weekly)

### Purpose

Scan Python dependencies for known security vulnerabilities.

### Implementation

**File**: `agent/maintenance/security_scan.py` (new)

```python
"""Scan dependencies for security vulnerabilities."""

import subprocess
import json

from services.container import alert_service
from services.logging import logger
from utils.exceptions import Severity


def scan_dependencies():
    """Scan Python dependencies for vulnerabilities using safety."""
    logger.info("Running dependency security scan")
    
    try:
        # Run safety check
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode == 0:
            logger.info("No vulnerabilities found")
            return {"status": "clean", "vulnerabilities": []}
        
        # Parse vulnerabilities
        vulnerabilities = json.loads(result.stdout)
        
        if vulnerabilities:
            logger.warning(f"Found {len(vulnerabilities)} vulnerabilities")
            
            # Send alert
            import asyncio
            asyncio.run(alert_service.send_alert(
                severity=Severity.WARNING,
                message=f"Security scan: {len(vulnerabilities)} vulnerabilities found",
                metadata={
                    "vulnerabilities": "; ".join([
                        f"{v['package']} {v['installed_version']}: {v['vulnerability_id']}"
                        for v in vulnerabilities[:5]
                    ])
                },
            ))
            
            return {
                "status": "vulnerable",
                "vulnerabilities": vulnerabilities,
            }
        
    except subprocess.TimeoutExpired:
        logger.error("Security scan timed out")
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
    
    return {"status": "error", "vulnerabilities": []}


if __name__ == "__main__":
    result = scan_dependencies()
    print(json.dumps(result, indent=2))
```

### Requirements

Add to `requirements.txt`:

```
safety>=2.3.0
```

### Scheduling

Add to `scheduler.py`:

```python
from maintenance.security_scan import scan_dependencies

def run_weekly_tasks():
    # ... existing weekly tasks ...
    
    try:
        scan_dependencies()
    except Exception as e:
        logger.error(f"Security scan failed: {e}")

# Already scheduled for Sundays at 4 AM
```

---

## Task 5: API Key Rotation Reminders (Quarterly)

### Purpose

Send reminders to rotate API keys every 90 days.

### Implementation

**File**: `agent/maintenance/key_rotation.py` (new)

```python
"""Remind about API key rotation."""

from datetime import datetime, timedelta

from services.container import alert_service
from services.logging import logger
from utils.exceptions import Severity


# Store last rotation dates in database or config
KEY_ROTATION_DATES = {
    "OPENROUTER_API_KEY": "2026-01-15",
    "GOOGLE_API_KEY": "2026-02-01",
    "OPENAI_API_KEY": "2026-01-20",
    "SUPABASE_KEY": "2026-03-01",
}


def check_key_rotation():
    """Check if any API keys are due for rotation."""
    today = datetime.utcnow()
    rotation_interval = timedelta(days=90)
    warning_threshold = timedelta(days=14)  # Warn 14 days before
    
    keys_due = []
    
    for key_name, last_rotation in KEY_ROTATION_DATES.items():
        last_rotation_date = datetime.fromisoformat(last_rotation)
        next_rotation = last_rotation_date + rotation_interval
        days_until_rotation = (next_rotation - today).days
        
        if days_until_rotation <= 0:
            keys_due.append({
                "key": key_name,
                "status": "overdue",
                "days_overdue": abs(days_until_rotation),
            })
        elif days_until_rotation <= warning_threshold.days:
            keys_due.append({
                "key": key_name,
                "status": "due_soon",
                "days_until": days_until_rotation,
            })
    
    if keys_due:
        logger.warning(f"{len(keys_due)} API keys need rotation")
        
        # Send alert
        import asyncio
        asyncio.run(alert_service.send_alert(
            severity=Severity.WARNING,
            message=f"API key rotation reminder: {len(keys_due)} keys need attention",
            metadata={
                "keys": "; ".join([
                    f"{k['key']} ({k['status']})" for k in keys_due
                ])
            },
        ))
        
        return {"status": "rotation_needed", "keys": keys_due}
    
    logger.info("All API keys are up to date")
    return {"status": "up_to_date", "keys": []}


if __name__ == "__main__":
    result = check_key_rotation()
    print(result)
```

### Scheduling

Add to `scheduler.py`:

```python
from maintenance.key_rotation import check_key_rotation

# Run daily (it will only alert when keys are actually due)
schedule.every().day.at("09:00").do(check_key_rotation)
```

---

## Task 6: Log Rotation (Daily)

### Purpose

Rotate log files to prevent disk space issues.

### Implementation

**File**: `agent/maintenance/log_rotation.py` (new)

```python
"""Rotate log files to prevent disk space issues."""

import os
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from services.logging import logger
from config import settings


def rotate_logs():
    """Rotate log files older than 1 day, delete logs older than 30 days."""
    if not settings.log_file:
        logger.info("Log rotation skipped: file logging not enabled")
        return
    
    log_dir = Path(settings.log_file).parent
    if not log_dir.exists():
        return
    
    today = datetime.utcnow()
    rotation_cutoff = today - timedelta(days=1)
    deletion_cutoff = today - timedelta(days=30)
    
    rotated = 0
    deleted = 0
    
    for log_file in log_dir.glob("*.log"):
        file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        
        # Delete old logs
        if file_mtime < deletion_cutoff:
            log_file.unlink()
            deleted += 1
            logger.debug(f"Deleted old log: {log_file.name}")
        
        # Rotate recent logs
        elif file_mtime < rotation_cutoff and not log_file.name.endswith(".gz"):
            # Compress and rename
            compressed = log_file.with_suffix(".log.gz")
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            log_file.unlink()
            rotated += 1
            logger.debug(f"Rotated log: {log_file.name} -> {compressed.name}")
    
    logger.info(f"Log rotation complete: {rotated} rotated, {deleted} deleted")


if __name__ == "__main__":
    rotate_logs()
```

---

## Task 7: Database Vacuum/Analyze (Weekly)

### Purpose

Run VACUUM and ANALYZE on database tables to maintain query performance.

### Implementation

**File**: `agent/maintenance/db_vacuum.py` (new)

```python
"""Run database VACUUM and ANALYZE for performance maintenance."""

from sqlalchemy import text

from services.container import db_session
from services.logging import logger


def vacuum_database():
    """Run VACUUM ANALYZE on all application tables."""
    logger.info("Running database VACUUM ANALYZE")
    
    tables = [
        "documents",
        "document_chunks",
        "decision_audit",
        "ingestion_failures",
    ]
    
    for table in tables:
        try:
            # Note: VACUUM cannot run inside a transaction
            # Use autocommit mode
            db_session.execute(text(f"VACUUM ANALYZE {table}"))
            logger.info(f"VACUUM ANALYZE complete for {table}")
        except Exception as e:
            logger.error(f"VACUUM ANALYZE failed for {table}: {e}")
    
    logger.info("Database VACUUM ANALYZE complete")


if __name__ == "__main__":
    vacuum_database()
```

---

## Summary

| Task | File | Frequency | Effort |
|------|------|-----------|--------|
| Vector store health | `maintenance/vector_store_health.py` | Daily | 2h |
| Session cleanup | `maintenance/session_cleanup.py` | Daily | 1h |
| Metrics archival | `maintenance/metrics_archival.py` | Monthly | 3h |
| Security scan | `maintenance/security_scan.py` | Weekly | 1h |
| Key rotation | `maintenance/key_rotation.py` | Daily (check) | 0.5h |
| Log rotation | `maintenance/log_rotation.py` | Daily | 1h |
| DB vacuum | `maintenance/db_vacuum.py` | Weekly | 1h |

**Total effort**: ~10 hours

---

## Deployment

1. Create `agent/maintenance/` directory
2. Add all task files
3. Create `maintenance/scheduler.py`
4. Add maintenance service to `docker-compose.yml`
5. Deploy and verify tasks run on schedule

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Implementation
