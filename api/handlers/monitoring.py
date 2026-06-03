"""Monitoring endpoint — return last check results for monitoring tasks."""

from fastapi import APIRouter

from config import settings
from infrastructure.container import _monitoring_scheduler
from models import MonitoringStatusResponse

router = APIRouter(prefix="/v1", tags=["monitoring"])


@router.get(
    "/monitoring/status",
    response_model=MonitoringStatusResponse,
    status_code=200,
)
async def monitoring_status() -> MonitoringStatusResponse:
    """Return the last check results for each monitoring verification task.

    Returns:
        MonitoringStatusResponse containing:
            - enabled: Whether monitoring is enabled
            - last_check: Timestamp of last complete check cycle
            - interval_seconds: Configured check interval
            - checks: List of individual check results
            - overall_status: "ok", "degraded", or "error"
    """
    results = _monitoring_scheduler.last_results
    checks = list(results.values())

    return MonitoringStatusResponse(
        enabled=settings.monitoring_enabled,
        last_check=_monitoring_scheduler.last_check,
        interval_seconds=settings.monitoring_interval_seconds,
        checks=checks,
        overall_status=_monitoring_scheduler.overall_status,
    )
