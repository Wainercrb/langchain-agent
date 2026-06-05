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



