"""Watch knowledge/ for new documents, process and store them.

Uses APScheduler for reliable scheduling with missed-run recovery and
graceful shutdown support.
"""

import signal
import time

from apscheduler.schedulers.blocking import BlockingScheduler
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
