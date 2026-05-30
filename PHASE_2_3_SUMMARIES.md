# Phase 2 & 3: Implementation Summaries

This document provides implementation sketches for P1 (Observability) and P2 (Data Quality) items. These are less detailed than the P0 guides but contain enough to execute.

---

## P1.1: Log Aggregation to CloudWatch (8 hours)

### Problem

Logs go to console or file (`infrastructure/logging/console.py`, `file.py`). No structured JSON shipping to CloudWatch Logs. Can't set CloudWatch Alarms or search logs across restarts.

### Solution

Use the `watchtower` library to ship structured JSON logs to CloudWatch Logs.

### Implementation Sketch

**File**: `agent/infrastructure/logging/cloudwatch.py` (new)

```python
"""CloudWatch Logs backend — ships structured JSON logs to AWS."""

import logging
import boto3
from watchtower import CloudWatchLogHandler

from .base import Logger
from config import settings


class CloudWatchLogger(Logger):
    """Structured JSON logger that ships to CloudWatch Logs."""
    
    def __init__(self):
        self._logger = logging.getLogger("langchain-agent")
        self._logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # CloudWatch handler
        handler = CloudWatchLogHandler(
            log_group_name=f"/aws/ecs/{settings.langsmith_project or 'langchain-agent'}",
            log_stream_name="application",
            boto3_client=boto3.client(
                "logs",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            ),
        )
        
        # JSON formatter
        handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"correlation_id": "%(correlation_id)s", "message": %(message)s}'
        ))
        
        self._logger.addHandler(handler)
    
    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    # ... implement debug, warning, error similarly
```

**File**: `agent/config/settings.py` (add)

```python
# ── AWS / CloudWatch ─────────────────────────────────────────────────
aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
cloudwatch_log_group: str = Field(default="langchain-agent", alias="CLOUDWATCH_LOG_GROUP")
```

**File**: `agent/requirements.txt` (add)

```
watchtower>=3.0.0
boto3>=1.28.0
```

**File**: `agent/infrastructure/logging/__init__.py` (modify)

```python
from config import settings

if settings.logger_backend == "cloudwatch":
    from .cloudwatch import CloudWatchLogger
    logger = CloudWatchLogger()
elif settings.logger_backend == "file":
    from .file import FileLogger
    logger = FileLogger()
else:
    from .console import ConsoleLogger
    logger = ConsoleLogger()
```

### Verification

1. Deploy to staging with `LOGGER_BACKEND=cloudwatch`
2. Send test queries
3. Check CloudWatch Logs console → log group should have entries
4. Verify JSON structure: `timestamp`, `level`, `correlation_id`, `message`
5. Create CloudWatch Alarm: error rate > 5% in 5 minutes → Discord alert

### CloudWatch Alarm Example

```yaml
ErrorRateAlarm:
  MetricName: ErrorCount
  Namespace: LangChainAgent
  Statistic: Sum
  Period: 300  # 5 minutes
  Threshold: 50  # 50 errors in 5 minutes
  ComparisonOperator: GreaterThanThreshold
  AlarmActions:
    - arn:aws:sns:us-east-1:123456789:discord-alerts
```

---

## P1.2: Persistent Metrics with Redis (4 hours)

### Problem

`SimpleMetrics` is in-memory only. Metrics reset to zero on every container restart. Can't track trends or set alerts on historical data.

### Solution

Replace `SimpleMetrics` with `RedisMetrics` that persists to Redis.

### Implementation Sketch

**File**: `agent/infrastructure/metrics/redis_metrics.py` (new)

```python
"""Redis-backed metrics — survives container restarts."""

import time
import redis
from config import settings
from services.logging import logger


class RedisMetrics:
    """Thread-safe Redis-backed request/error/latency counters."""
    
    def __init__(self):
        self._redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
        self._prefix = "metrics:"
    
    def record_request(self, latency_ms: float) -> None:
        """Record a completed request with its latency."""
        pipe = self._redis.pipeline()
        pipe.incr(f"{self._prefix}request_count")
        pipe.incrbyfloat(f"{self._prefix}total_latency_ms", latency_ms)
        
        # Time-series: record latency for current minute
        minute_key = f"{self._prefix}latency:{int(time.time() // 60)}"
        pipe.lpush(minute_key, latency_ms)
        pipe.expire(minute_key, 86400)  # Retain for 24 hours
        
        pipe.execute()
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self._redis.incr(f"{self._prefix}error_count")
    
    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of all counters."""
        request_count = int(self._redis.get(f"{self._prefix}request_count") or 0)
        error_count = int(self._redis.get(f"{self._prefix}error_count") or 0)
        total_latency = float(self._redis.get(f"{self._prefix}total_latency_ms") or 0)
        
        avg_latency = round(total_latency / request_count, 2) if request_count > 0 else 0.0
        
        return {
            "request_count": request_count,
            "error_count": error_count,
            "avg_latency_ms": avg_latency,
        }
```

**File**: `agent/api/metrics.py` (modify)

```python
from config import settings

if settings.enable_redis_metrics:
    from services.metrics.redis_metrics import RedisMetrics
    _metrics = RedisMetrics()
else:
    _metrics = SimpleMetrics()  # Fallback for local dev
```

**File**: `docker-compose.yml` (add Redis service)

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: langchain-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - langchain-network
    restart: unless-stopped

volumes:
  redis-data:
```

### Verification

1. Start Redis: `docker-compose up redis`
2. Deploy agent with `ENABLE_REDIS_METRICS=true`
3. Send 10 test queries
4. Check `/v1/metrics` → should show `request_count: 10`
5. Restart agent container
6. Check `/v1/metrics` → should STILL show `request_count: 10` (not reset to 0)

---

## P2.1: Failed Document Retry & Notification (6 hours)

### Problem

Failed document ingestions go to `knowledge/failed/` directory. No automated retry, no notification, no dashboard showing ingestion health.

### Solution

Implement a retry queue in the database with automatic retry and Discord notifications.

### Implementation Sketch

**Database Migration**: `agent/migrations/versions/xxx_add_ingestion_failures.py`

```python
"""Add ingestion_failures table for retry queue."""

def upgrade():
    op.create_table(
        'ingestion_failures',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('error_message', sa.Text),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('max_retries', sa.Integer, default=3),
        sa.Column('next_retry_at', sa.DateTime),
        sa.Column('status', sa.String(50), default='pending'),  # pending, retrying, failed, success
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
    )
```

**File**: `agent/infrastructure/ingestion/retry_queue.py` (new)

```python
"""Retry queue for failed document ingestions."""

from datetime import datetime, timedelta
from typing import List
from sqlalchemy import select, update

from models.ingestion import IngestionFailure
from services.logging import logger
from services.container import alert_service
from utils.exceptions import Severity


class RetryQueue:
    """Manages retry queue for failed document ingestions."""
    
    def __init__(self, db_session, max_retries: int = 3):
        self._db = db_session
        self._max_retries = max_retries
    
    def add_failure(self, filename: str, error_message: str) -> None:
        """Add a failed ingestion to the retry queue."""
        # Check if already in queue
        existing = self._db.execute(
            select(IngestionFailure).where(
                IngestionFailure.filename == filename,
                IngestionFailure.status.in_(["pending", "retrying"]),
            )
        ).scalar_one_or_none()
        
        if existing:
            # Update existing failure
            existing.retry_count += 1
            existing.error_message = error_message
            existing.next_retry_at = datetime.utcnow() + timedelta(
                minutes=2 ** existing.retry_count  # Exponential backoff
            )
        else:
            # Create new failure
            failure = IngestionFailure(
                filename=filename,
                error_message=error_message,
                max_retries=self._max_retries,
                next_retry_at=datetime.utcnow() + timedelta(minutes=2),
            )
            self._db.add(failure)
        
        self._db.commit()
    
    def get_retryable(self) -> List[IngestionFailure]:
        """Get failures that are ready for retry."""
        return self._db.execute(
            select(IngestionFailure).where(
                IngestionFailure.status == "pending",
                IngestionFailure.next_retry_at <= datetime.utcnow(),
                IngestionFailure.retry_count < IngestionFailure.max_retries,
            )
        ).scalars().all()
    
    async def notify_permanent_failure(self, failure: IngestionFailure) -> None:
        """Send Discord alert for permanently failed documents."""
        await alert_service.send_alert(
            severity=Severity.ERROR,
            message=f"Document ingestion permanently failed: {failure.filename}",
            metadata={
                "error": failure.error_message[:200],
                "retry_count": failure.retry_count,
            },
        )
```

**File**: `agent/cronjob.py` (modify)

```python
# Add retry queue processing to the ingestion loop

from services.ingestion.retry_queue import RetryQueue

retry_queue = RetryQueue(db_session)

while True:
    # Process new documents
    results = pipeline.ingest_directory(settings.knowledge_dir)
    
    # Add failures to retry queue
    for result in results:
        if result.status.value == "failed":
            retry_queue.add_failure(result.filename, result.error_message)
    
    # Process retry queue
    retryable = retry_queue.get_retryable()
    for failure in retryable:
        try:
            pipeline.ingest_file(failure.filename)
            failure.status = "success"
        except Exception as e:
            failure.retry_count += 1
            if failure.retry_count >= failure.max_retries:
                failure.status = "failed"
                await retry_queue.notify_permanent_failure(failure)
    
    db_session.commit()
    time.sleep(settings.cron_interval_minutes * 60)
```

### Verification

1. Deploy with database migration
2. Place an invalid document in `knowledge/` directory
3. Wait for cronjob to process → should fail and add to retry queue
4. Check `ingestion_failures` table → should have 1 row with `status=pending`
5. Wait for retry → should retry with exponential backoff
6. After 3 retries → should send Discord alert and mark as `failed`

---

## P2.2: Local Decision Audit Trail (8 hours)

### Problem

AI decision logs (which tool was chosen, why) exist only in LangSmith. Can't query "show me all queries where agent chose web_search instead of search_documents" from your own system.

### Solution

Store decision audit trail in database — queryable via SQL, independent of LangSmith.

### Implementation Sketch

**Database Migration**: `agent/migrations/versions/xxx_add_decision_audit.py`

```python
"""Add decision_audit table for local AI decision logging."""

def upgrade():
    op.create_table(
        'decision_audit',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('run_id', sa.String(36), index=True),
        sa.Column('query', sa.Text, nullable=False),
        sa.Column('tool_chosen', sa.String(100)),  # search_documents, web_search, or None
        sa.Column('sources_count', sa.Integer, default=0),
        sa.Column('latency_ms', sa.Float),
        sa.Column('model', sa.String(100)),
        sa.Column('provider', sa.String(100)),  # openrouter, gemini, openai
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), index=True),
    )
```

**File**: `agent/infrastructure/audit/decision_logger.py` (new)

```python
"""Local decision audit trail — stores AI decisions in database."""

from datetime import datetime
from sqlalchemy.orm import Session

from models.audit import DecisionAudit
from services.logging import logger


class DecisionLogger:
    """Logs AI decisions to database for local querying."""
    
    def __init__(self, db_session: Session):
        self._db = db_session
    
    def log_decision(
        self,
        run_id: str,
        query: str,
        tool_chosen: str | None,
        sources_count: int,
        latency_ms: float,
        model: str,
        provider: str,
    ) -> None:
        """Log an AI decision."""
        audit = DecisionAudit(
            run_id=run_id,
            query=query[:2000],  # Truncate long queries
            tool_chosen=tool_chosen,
            sources_count=sources_count,
            latency_ms=latency_ms,
            model=model,
            provider=provider,
        )
        self._db.add(audit)
        self._db.commit()
        
        logger.debug(
            f"Decision logged: run_id={run_id}, tool={tool_chosen}, "
            f"sources={sources_count}, latency={latency_ms:.0f}ms"
        )
```

**File**: `agent/infrastructure/agent/tool_calling.py` (modify)

```python
# After building the response, log the decision

from services.audit.decision_logger import DecisionLogger
from services.container import db_session

# Inside invoke() method, before returning:
decision_logger = DecisionLogger(db_session)
decision_logger.log_decision(
    run_id=run_id,
    query=query,
    tool_chosen=self._detect_tool_chosen(result),  # Parse from agent output
    sources_count=len(sources_list or []),
    latency_ms=execution_time_ms,
    model=getattr(self._llm, "model", "unknown"),
    provider=self._llm.active_provider if hasattr(self._llm, 'active_provider') else "unknown",
)
```

**File**: `agent/api/routes.py` (add endpoint)

```python
@router.get("/v1/audit/decisions", status_code=200)
async def get_decisions(
    limit: int = 100,
    offset: int = 0,
    tool: str | None = None,
    provider: str | None = None,
) -> list:
    """Query decision audit trail.
    
    Args:
        limit: Max results (default 100)
        offset: Pagination offset
        tool: Filter by tool chosen (e.g., "search_documents")
        provider: Filter by LLM provider (e.g., "openrouter")
    
    Returns:
        List of decision audit records
    """
    from services.container import db_session
    from models.audit import DecisionAudit
    from sqlalchemy import select
    
    query = select(DecisionAudit)
    
    if tool:
        query = query.where(DecisionAudit.tool_chosen == tool)
    if provider:
        query = query.where(DecisionAudit.provider == provider)
    
    query = query.order_by(DecisionAudit.created_at.desc()).limit(limit).offset(offset)
    
    results = db_session.execute(query).scalars().all()
    return [r.to_dict() for r in results]
```

### Verification

1. Deploy with database migration
2. Send 10 test queries with different types (document search, web search, direct)
3. Query `/v1/audit/decisions` → should return 10 records
4. Filter by tool: `/v1/audit/decisions?tool=search_documents` → should return only document searches
5. Check database directly: `SELECT * FROM decision_audit ORDER BY created_at DESC LIMIT 10`

### Example Queries

```sql
-- Find all queries where agent chose web_search instead of search_documents
SELECT query, tool_chosen, created_at
FROM decision_audit
WHERE tool_chosen = 'web_search'
  AND query ILIKE '%find%'
ORDER BY created_at DESC;

-- Average latency by provider
SELECT provider, AVG(latency_ms) as avg_latency, COUNT(*) as count
FROM decision_audit
GROUP BY provider;

-- Queries with no sources (potential quality issues)
SELECT query, tool_chosen, created_at
FROM decision_audit
WHERE sources_count = 0
  AND tool_chosen IS NOT NULL
ORDER BY created_at DESC;
```

---

## Summary

| Item | Effort | Files Changed | Risk | Priority |
|------|--------|---------------|------|----------|
| **P1.1: Log Aggregation** | 8h | 4 files | Medium | High |
| **P1.2: Persistent Metrics** | 4h | 4 files | Low | High |
| **P2.1: Failed Document Retry** | 6h | 5 files + migration | Medium | Medium |
| **P2.2: Decision Audit Trail** | 8h | 5 files + migration | Low | Medium |

**Total P1+P2 Effort**: 26 hours

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Implementation
