"""Monitoring scheduler — background task loop for automated health checks.

Runs configured health checks on a configurable interval using asyncio.
Dispatches alerts when any check fails. Stores last results for
the /v1/monitoring/status endpoint.
"""

import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from alerts.base import AlertProviderBase
from config import Settings
from config import settings as default_settings
from logging import logger
from models.observability.health import HealthCheckResult
from observability.health.checks import CheckResult
from shared.exceptions import Severity

# Type alias for a named check: (check_name, async_callable returning CheckResult)
CheckDefinition = Tuple[str, Callable[[], Awaitable[CheckResult]]]


class MonitoringScheduler:
    """Background scheduler that runs health checks on a configurable interval.

    Accepts a list of check definitions at construction time, allowing
    new checks to be added without modifying this class (OCP).

    Args:
        checks: List of (name, async_callable) check definitions.
        alert_service: Alert provider for failure notifications.
        settings: Settings instance (uses global settings if None).
    """

    def __init__(
        self,
        checks: List[CheckDefinition],
        alert_service: AlertProviderBase,
        settings: Optional[Settings] = None,
    ) -> None:
        self._checks = checks
        self._alert_service = alert_service
        self._settings = settings or default_settings
        self._task: Optional[asyncio.Task[None]] = None
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._last_check: Optional[datetime] = None

    async def start(self) -> None:
        """Start the background monitoring task."""
        if not self._settings.monitoring_enabled:
            logger.debug("Monitoring disabled, skipping start")
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Monitoring scheduler started")

    async def stop(self) -> None:
        """Cancel the background monitoring task."""
        if not self._settings.monitoring_enabled:
            return
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

        while True:
            try:
                any_failed = await self._execute_checks()

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
                await self._send_critical_alert(f"Monitoring scheduler crashed: {str(e)}")

            await asyncio.sleep(interval)

    async def _execute_checks(self) -> bool:
        """Run all configured checks and store results.

        Returns:
            True if any check failed, False otherwise.
        """
        any_failed = False

        for check_name, check_method in self._checks:
            result = await self._run_single_check(check_name, check_method)
            if not result.ok:
                any_failed = True
                await self._send_alert(check_name, result.detail)

        return any_failed

    async def _run_single_check(
        self, check_name: str, check_method: Callable[[], Awaitable[CheckResult]]
    ) -> HealthCheckResult:
        """Execute a single check with error handling and result recording."""
        try:
            result = await check_method()
        except Exception as e:
            result = CheckResult.failure(f"Check crashed: {str(e)}")
            logger.error(f"Monitoring check {check_name} crashed: {e}", exc_info=True)

        health_result = HealthCheckResult(
            check_name=check_name,
            ok=result.ok,
            detail=result.detail,
            last_checked=datetime.now(timezone.utc),
        )
        self._last_results[check_name] = health_result

        if not result.ok:
            logger.warning(f"Monitoring check {check_name} failed: {result.detail}")

        return health_result

    async def _send_alert(self, check_name: str, detail: str) -> None:
        """Send an alert for a failed check."""
        try:
            await self._alert_service.send_alert(
                severity=Severity.ERROR,
                message=f"Monitoring: {check_name} check failed",
                metadata={"check_name": check_name, "detail": detail},
            )
        except Exception:
            logger.error(f"Failed to send alert for {check_name}", exc_info=True)

    async def _send_critical_alert(self, message: str) -> None:
        """Send a critical-level alert for scheduler failures."""
        try:
            await self._alert_service.send_alert(
                severity=Severity.CRITICAL,
                message=message,
            )
        except Exception:
            pass

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
