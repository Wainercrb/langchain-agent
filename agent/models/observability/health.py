"""Pydantic models for health monitoring responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class HealthCheckResult(BaseModel):
    """Result of a single health check."""

    check_name: str
    ok: bool
    detail: str
    last_checked: Optional[datetime] = None


class MonitoringStatusResponse(BaseModel):
    """Response model for GET /v1/monitoring/status."""

    enabled: bool
    last_check: Optional[datetime] = None
    interval_seconds: int
    checks: list[HealthCheckResult]
    overall_status: str  # "ok" | "degraded" | "error"
