"""Monitoring scheduler — background task loop for automated health checks.

Runs all HealthVerifier checks on a configurable interval using asyncio.
Dispatches Discord alerts when any check fails. Stores last results for
the /v1/monitoring/status endpoint.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional

from config import settings
from infrastructure.logging import logger
from infrastructure.observability.health.checks import HealthVerifier
from models.observability.health import HealthCheckResult
from utils.exceptions import Severity


class MonitoringScheduler:
    """Background scheduler that runs health checks on an interval."""

    def __init__(
        self,
        health_verifier: HealthVerifier,
        alert_service,
        settings_obj=None,
        decision_tracker=None,
    ) -> None:
        self._health_verifier = health_verifier
        self._alert_service = alert_service
        self._settings = settings_obj or settings
        self._decision_tracker = decision_tracker
        self._task: Optional[asyncio.Task] = None
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._last_check: Optional[datetime] = None

    async def start(self) -> None:
        """Start the background monitoring task."""
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Monitoring scheduler started")

    async def stop(self) -> None:
        """Cancel the background monitoring task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Monitoring scheduler stopped")

    async def _run_loop(self) -> None:
        """Main loop: run all checks on interval with crash protection."""
        interval = self._settings.monitoring_interval_seconds
        check_methods = [
            ("db", self._health_verifier.check_db),
            ("langsmith", self._health_verifier.check_langsmith),
            ("embeddings", self._health_verifier.check_embeddings),
            ("tracing_completeness", self._health_verifier.check_tracing_completeness),
            ("memory_usage", self._health_verifier.check_memory_usage),
            ("decision_drift", lambda: self._health_verifier.check_decision_drift(self._decision_tracker)),
        ]

        while True:
            any_failed = False
            try:
                for check_name, check_method in check_methods:
                    try:
                        ok, detail = await check_method()
                    except Exception as e:
                        ok = False
                        detail = f"Check crashed: {str(e)}"
                        logger.error(f"Monitoring check {check_name} crashed: {e}", exc_info=True)

                    result = HealthCheckResult(
                        check_name=check_name,
                        ok=ok,
                        detail=detail,
                        last_checked=datetime.now(timezone.utc),
                    )
                    self._last_results[check_name] = result

                    if not ok:
                        any_failed = True
                        logger.warning(f"Monitoring check {check_name} failed: {detail}")
                        await self._alert_service.send_alert(
                            severity=Severity.ERROR,
                            message=f"Monitoring: {check_name} check failed",
                            metadata={"check_name": check_name, "detail": detail},
                        )

                self._last_check = datetime.now(timezone.utc)

                if any_failed:
                    logger.warning("One or more monitoring checks failed")
                else:
                    logger.debug("All monitoring checks passed")

            except asyncio.CancelledError:
                logger.info("Monitoring scheduler loop cancelled")
                return
            except Exception as e:
                logger.error(f"Monitoring scheduler loop crashed: {e}", exc_info=True)
                try:
                    await self._alert_service.send_alert(
                        severity=Severity.CRITICAL,
                        message=f"Monitoring scheduler crashed: {str(e)}",
                        error=e,
                    )
                except Exception:
                    pass

            await asyncio.sleep(interval)

    @property
    def last_results(self) -> Dict[str, HealthCheckResult]:
        """Return the last check results for the status endpoint."""
        return self._last_results

    @property
    def last_check(self) -> Optional[datetime]:
        """Return the timestamp of the last complete check cycle."""
        return self._last_check

    @property
    def overall_status(self) -> str:
        """Return overall status based on last results."""
        if not self._last_results:
            return "error"
        if all(r.ok for r in self._last_results.values()):
            return "ok"
        return "degraded"
