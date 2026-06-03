"""Watch knowledge/ for new documents, process and store them.

Also runs scheduled maintenance tasks (backup, log rotation, VACUUM ANALYZE)
to automate items from docs/runbooks/weekly-maintenance.md.

Uses APScheduler for reliable scheduling with missed-run recovery and
graceful shutdown support.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from domain.ingestion import DocumentIngestionPipeline
from domain.ingestion.pipeline import IngestionStatus
from infrastructure.container import alert_service, embeddings, vector_store
from infrastructure.logging import logger
from utils.exceptions import Severity

# ── Ingestion Alert Deduplication ─────────────────────────────────────
# Prevents alert fatigue during mass failures or repeated cron cycles.
# Uses a stable fingerprint ("ingestion_failure") so different failure
# batches within the cooldown window are suppressed.
_last_ingestion_alert_time: float = 0
_ingestion_alert_cooldown = 3600  # 1 hour between ingestion failure alerts


def reset_ingestion_alert_cooldown() -> None:
    """Reset the ingestion alert cooldown. Useful for testing."""
    global _last_ingestion_alert_time
    _last_ingestion_alert_time = 0


def _alert_on_failures(results) -> None:
    """Send a simple Discord alert when ingestion failures are detected.

    Deduplicates alerts using a cooldown window so that repeated cron cycles
    or mass failures don't cause alert fatigue. Only one alert per cooldown
    period is sent, regardless of how many failure batches occur.
    """
    global _last_ingestion_alert_time

    failures = [r for r in results if r.status == IngestionStatus.FAILED]
    if not failures:
        return

    # Dedup check: skip if within cooldown window
    now = time.time()
    elapsed = now - _last_ingestion_alert_time
    if elapsed < _ingestion_alert_cooldown:
        remaining = _ingestion_alert_cooldown - elapsed
        logger.info(
            f"Ingestion alert suppressed (cooldown): {len(failures)} failures, "
            f"next alert in {remaining:.0f}s"
        )
        return

    message = f"Ingestion pipeline: {len(failures)} file(s) failed\n\n"
    for f in failures[:5]:
        message += f"- **{f.filename}**: {f.error or 'Unknown error'}\n"

    _last_ingestion_alert_time = now

    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(alert_service.send_alert(
            severity=Severity.ERROR,
            message=message[:2000],
            metadata={"failed_count": len(failures)},
        ))
    except RuntimeError:
        asyncio.run(alert_service.send_alert(
            severity=Severity.ERROR,
            message=message[:2000],
            metadata={"failed_count": len(failures)},
        ))


def _run_ingestion_cycle(pipeline: DocumentIngestionPipeline) -> None:
    """Execute one ingestion cycle. Called by APScheduler on each trigger."""
    try:
        results = pipeline.ingest_directory(settings.knowledge_dir)
    except Exception:
        logger.error("Ingestion cycle crashed", exc_info=True)
        return

    if results:
        success = sum(1 for r in results if r.status.value == "success")
        skipped = sum(1 for r in results if r.status.value == "skipped")
        failed = sum(1 for r in results if r.status.value == "failed")
        logger.info(
            f"Cycle complete: {success} success, {skipped} skipped, {failed} failed"
        )

        _alert_on_failures(results)

    # Dead letter queue: retry failed files from previous cycles
    try:
        retry_results = pipeline.retry_failed_files(max_retries=settings.ingestion_max_retries)
        if retry_results:
            retry_success = sum(1 for r in retry_results if r.status.value == "success")
            retry_failed = sum(1 for r in retry_results if r.status.value == "failed")
            if retry_success:
                logger.info(f"DLQ: {retry_success} file(s) recovered")
            if retry_failed:
                logger.warning(f"DLQ: {retry_failed} file(s) still failing")
                _alert_on_failures(retry_results)
    except Exception:
        logger.error("DLQ retry cycle crashed", exc_info=True)


# ── Maintenance Jobs (automated runbook items) ────────────────────────
# These functions implement the 3 scheduled items from
# docs/runbooks/weekly-maintenance.md that previously required manual
# execution. They follow the same alert-on-failure pattern as ingestion.


def _send_maintenance_alert(job_name: str, error: str) -> None:
    """Send an ERROR alert when a maintenance job fails.

    Mirrors the asyncio-loop detection pattern in `_alert_on_failures` so
    we work correctly inside APScheduler's BlockingScheduler.

    Args:
        job_name: Identifier of the failing job (e.g. "weekly_backup").
        error: Human-readable error description (truncated to 1000 chars).
    """
    import asyncio

    message = f"Maintenance job '{job_name}' failed: {error[:500]}"
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(alert_service.send_alert(
            severity=Severity.ERROR,
            message=message,
            metadata={"job": job_name, "error": error[:1000]},
        ))
    except RuntimeError:
        asyncio.run(alert_service.send_alert(
            severity=Severity.ERROR,
            message=message,
            metadata={"job": job_name, "error": error[:1000]},
        ))


def _run_backup_cycle() -> None:
    """Weekly database backup via scripts/backup.py.

    Wraps the existing backup script as a subprocess so we reuse the
    pg_dump + API fallback logic and the script's own CRITICAL alerting
    on catastrophic failure. We add an additional ERROR alert at the
    cronjob level so failures are visible in the maintenance summary.
    """
    backup_script = Path(__file__).parent / "scripts" / "backup.py"
    if not backup_script.exists():
        logger.error(f"Backup script not found: {backup_script}")
        _send_maintenance_alert("weekly_backup", f"Script not found: {backup_script}")
        return

    cmd = [
        sys.executable,
        str(backup_script),
        "--retention",
        str(settings.maintenance_backup_retention),
    ]
    logger.info(
        f"Starting weekly backup (retention={settings.maintenance_backup_retention})"
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            error = result.stderr.strip() or "Unknown error"
            logger.error(f"Backup failed (exit {result.returncode}): {error}")
            _send_maintenance_alert("weekly_backup", error)
        else:
            logger.info("Weekly backup completed successfully")
    except subprocess.TimeoutExpired:
        logger.error("Backup timed out after 600s")
        _send_maintenance_alert("weekly_backup", "Timeout after 600s")
    except Exception as e:
        logger.error(f"Backup crashed: {e}", exc_info=True)
        _send_maintenance_alert("weekly_backup", str(e))


def _run_vacuum_analyze() -> None:
    """Weekly VACUUM ANALYZE on high-write tables in Supabase.

    Requires the psql client in PATH and SUPABASE_DIRECT_URL. If psql is
    not available, logs a warning and skips (matches the pg_dump fallback
    pattern in scripts/backup.py).
    """
    psql = shutil.which("psql")
    if not psql:
        logger.warning("psql not found in PATH, skipping VACUUM ANALYZE")
        return

    if not settings.supabase_direct_url:
        logger.warning("SUPABASE_DIRECT_URL not set, cannot run VACUUM ANALYZE")
        return

    # High-write tables identified in docs/runbooks/weekly-maintenance.md
    tables = ["ingestion_logs", "documents", "document_chunks"]
    env = os.environ.copy()
    env["PGPASSWORD"] = settings.supabase_key

    failures: list[tuple[str, str]] = []
    for table in tables:
        cmd = [psql, settings.supabase_direct_url, "-c", f"VACUUM ANALYZE {table};"]
        logger.info(f"Running VACUUM ANALYZE on {table}")
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                error = result.stderr.strip() or "Unknown error"
                logger.error(f"VACUUM ANALYZE {table} failed: {error}")
                failures.append((table, error))
        except subprocess.TimeoutExpired:
            logger.error(f"VACUUM ANALYZE {table} timed out")
            failures.append((table, "Timeout after 300s"))
        except Exception as e:
            logger.error(f"VACUUM ANALYZE {table} crashed: {e}", exc_info=True)
            failures.append((table, str(e)))

    if failures:
        error_summary = "; ".join(f"{t}: {e}" for t, e in failures)
        _send_maintenance_alert("weekly_vacuum_analyze", error_summary)
    else:
        logger.info(
            f"Weekly VACUUM ANALYZE completed successfully ({len(tables)} tables)"
        )


def main() -> None:
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.failed_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DocumentIngestionPipeline(
        embeddings=embeddings,
        vector_store=vector_store,
        processed_dir=settings.processed_dir,
        failed_dir=settings.failed_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _run_ingestion_cycle,
        trigger=IntervalTrigger(minutes=settings.cron_interval_minutes),
        args=[pipeline],
        id="ingestion_cycle",
        name="Document ingestion cycle",
        replace_existing=True,
        max_instances=1,  # Prevent overlapping runs
        misfire_grace_time=60,  # Tolerate up to 60s late start
    )

    logger.info(
        f"APScheduler watching {settings.knowledge_dir} "
        f"every {settings.cron_interval_minutes}min"
    )

    # ── Automated maintenance jobs (from docs/runbooks/weekly-maintenance.md) ──
    if settings.maintenance_backup_enabled:
        scheduler.add_job(
            _run_backup_cycle,
            trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="weekly_backup",
            name="Weekly database backup",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,  # 1h grace for weekly jobs
        )
        logger.info("Scheduled weekly backup: Sundays 02:00")

    if settings.maintenance_vacuum_enabled:
        scheduler.add_job(
            _run_vacuum_analyze,
            trigger=CronTrigger(day_of_week="sun", hour=4, minute=0),
            id="weekly_vacuum_analyze",
            name="Weekly VACUUM ANALYZE",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        logger.info("Scheduled weekly VACUUM ANALYZE: Sundays 04:00")

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down scheduler...")
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
