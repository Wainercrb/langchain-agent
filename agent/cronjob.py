"""Watch knowledge/ for new documents, process and store them."""

import time

from config import settings
from domain.ingestion import DocumentIngestionPipeline
from domain.ingestion.pipeline import IngestionStatus
from infrastructure.container import alert_service, embeddings, vector_store
from infrastructure.logging import logger
from utils.exceptions import Severity


def _alert_on_failures(results) -> None:
    """Send a simple Discord alert when ingestion failures are detected."""
    failures = [r for r in results if r.status == IngestionStatus.FAILED]
    if not failures:
        return

    message = f"Ingestion pipeline: {len(failures)} file(s) failed\n\n"
    for f in failures[:5]:
        message += f"- **{f.filename}**: {f.error or 'Unknown error'}\n"

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

    logger.info(
        f"Watching {settings.knowledge_dir} every {settings.cron_interval_minutes}min"
    )

    while True:
        results = pipeline.ingest_directory(settings.knowledge_dir)

        if results:
            success = sum(1 for r in results if r.status.value == "success")
            skipped = sum(1 for r in results if r.status.value == "skipped")
            failed = sum(1 for r in results if r.status.value == "failed")
            logger.info(
                f"Cycle complete: {success} success, {skipped} skipped, {failed} failed"
            )

            _alert_on_failures(results)

        time.sleep(settings.cron_interval_minutes * 60)


if __name__ == "__main__":
    main()
