"""Cron scheduler for batch document ingestion."""

import logging
import signal
import sys
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class CronScheduler:
    """Schedule and execute document ingestion jobs at intervals."""

    def __init__(self, interval_minutes: int = 5):
        """
        Initialize cron scheduler.

        Args:
            interval_minutes: Interval in minutes between job executions
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger

            self.interval_minutes = interval_minutes
            self.scheduler = BackgroundScheduler()
            self.trigger = IntervalTrigger(minutes=interval_minutes)
            self.job = None
            logger.info(f"CronScheduler initialized (interval={interval_minutes}min)")
        except ImportError as e:
            logger.error(f"Failed to import APScheduler: {str(e)}")
            raise

    def add_job(
        self,
        job_func: Callable,
        job_id: str = "document_ingestion",
        replace_existing: bool = True,
    ) -> None:
        """
        Add a job to the scheduler.

        Args:
            job_func: Callable to execute
            job_id: Unique job ID
            replace_existing: Whether to replace existing job with same ID
        """
        try:
            self.job = self.scheduler.add_job(
                job_func,
                self.trigger,
                id=job_id,
                name=f"Document ingestion ({self.interval_minutes}min)",
                replace_existing=replace_existing,
                max_instances=1,  # Prevent concurrent executions
                coalesce=True,  # Skip missed runs if scheduler falls behind
            )
            logger.info(f"Added job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to add job: {str(e)}")
            raise

    def start(self, blocking: bool = True) -> None:
        """
        Start the scheduler.

        Args:
            blocking: Whether to block until scheduler stops (default: True)
        """
        try:
            self.scheduler.start()
            logger.info("Scheduler started")

            if blocking:
                # Setup signal handlers for graceful shutdown
                def signal_handler(signum, frame):
                    logger.info(f"Received signal {signum}, shutting down...")
                    self.stop()
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                logger.info("Scheduler running (Ctrl+C to stop)...")
                try:
                    import time

                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.stop()
                    logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
            raise

    def stop(self, wait: bool = True) -> None:
        """
        Stop the scheduler.

        Args:
            wait: Whether to wait for jobs to complete
        """
        try:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {str(e)}")
            raise

    def get_jobs(self):
        """
        Get all scheduled jobs.

        Returns:
            List of scheduled jobs
        """
        return self.scheduler.get_jobs()

    def trigger_job_now(self) -> None:
        """Manually trigger the job immediately."""
        try:
            if self.job:
                self.job.func()
                logger.info("Job triggered manually")
            else:
                logger.warning("No job registered to trigger")
        except Exception as e:
            logger.error(f"Failed to trigger job: {str(e)}")
            raise

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self.scheduler.running
