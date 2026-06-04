"""Monitoring scheduler — background task loop for automated health checks.

Runs configured health checks on a configurable interval using asyncio.
Dispatches alerts when any check fails. Stores last results for
the /v1/monitoring/status endpoint.
"""

import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from alerts.base import AlertProviderBase
from config import settings
from loggers import logger
from models.observability.health import HealthCheckResult
from observability.base import CheckResult
from shared.exceptions import Severity

# Type alias for a named check: (check_name, async_callable returning CheckResult)
CheckDefinition = Tuple[str, Callable[[], Awaitable[CheckResult]]]


class MonitoringScheduler:
    """Background scheduler that runs health checks on a configurable interval.

    Accepts a list of check definitions at construction time, allowing
    new checks to be added without modifying this class (OCP).

    The caller decides whether to call ``start()`` — do NOT gate on config
    inside this class (SRV: let the caller own the decision).

    Args:
        checks: List of (name, async_callable) check definitions.
        alert_service: Alert provider for failure notifications.
        interval_seconds: Seconds between check cycles.
    """

    @classmethod
    def is_configured(cls) -> bool:
        """Return True when monitoring is enabled in settings."""
        return bool(settings.monitoring_enabled)

    def __init__(
        self,
        checks: List[CheckDefinition],
        alert_service: AlertProviderBase,
        interval_seconds: int = 300,
    ) -> None:
        self._checks = checks
        self._alert_service = alert_service
        self._interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task[None]] = None
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._last_check: Optional[datetime] = None

    # ── Public lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background monitoring task.

        When monitoring is disabled (``is_configured()`` returns False),
        this is a no-op — the single source of truth lives on the class.
        """
        if not self.is_configured():
            logger.debug("Monitoring disabled, skipping start")
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Monitoring scheduler started")

    async def stop(self) -> None:
        """Cancel the background monitoring task."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("Monitoring scheduler stopped")

    # ── Loop ───────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main loop: tick on a fixed interval with crash protection."""
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.info("Monitoring scheduler loop cancelled")
                return
            except Exception as e:
                logger.error(f"Monitoring scheduler loop crashed: {e}", exc_info=True)
                await self._alert(
                    Severity.CRITICAL,
                    f"Monitoring scheduler crashed: {e}",
                    suppress_errors=True,
                )
            await asyncio.sleep(self._interval_seconds)

    async def _tick(self) -> None:
        """Run one complete cycle of all checks."""
        any_failed = await self._execute_checks()
        self._last_check = datetime.now(timezone.utc)
        if any_failed:
            logger.warning("One or more monitoring checks failed")
        else:
            logger.debug("All monitoring checks passed")

    # ── Check execution ────────────────────────────────────────────────

    async def _execute_checks(self) -> bool:
        """Run all configured checks. Returns True if any failed."""
        failed = False

        for check_name, check_method in self._checks:
            health_result = await self._run_single_check(check_name, check_method)
            self._last_results[check_name] = health_result

            if not health_result.ok:
                failed = True
                await self._alert(
                    Severity.ERROR,
                    f"Monitoring: {check_name} check failed",
                    metadata={"check_name": check_name, "detail": health_result.detail},
                )

        return failed

    @staticmethod
    async def _run_single_check(
        check_name: str,
        check_method: Callable[[], Awaitable[CheckResult]],
    ) -> HealthCheckResult:
        """Execute one check with error isolation and result recording.

        Returns a HealthCheckResult regardless of success or failure —
        the caller decides how to react.
        """
        try:
            result = await check_method()
        except Exception as e:
            result = CheckResult.failure(f"Check crashed: {e}")
            logger.error(f"Monitoring check {check_name} crashed: {e}", exc_info=True)

        health_result = HealthCheckResult(
            check_name=check_name,
            ok=result.ok,
            detail=result.detail,
            last_checked=datetime.now(timezone.utc),
        )

        if not result.ok:
            logger.warning(f"Monitoring check {check_name} failed: {result.detail}")

        return health_result

    # ── Alerting ───────────────────────────────────────────────────────

    async def _alert(
        self,
        severity: Severity,
        message: str,
        metadata: Optional[Dict[str, str]] = None,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Send an alert through the configured provider.

        Args:
            severity: Alert severity level.
            message: Human-readable alert message.
            metadata: Optional structured context.
            suppress_errors: If True, provider failures are silently ignored
                (use for critical alerts to avoid alert-on-alert loops).
        """
        try:
            await self._alert_service.send_alert(
                severity=severity,
                message=message,
                metadata=metadata,
            )
        except Exception:
            if not suppress_errors:
                logger.error("Failed to send alert", exc_info=True)

    # ── Status ─────────────────────────────────────────────────────────

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
