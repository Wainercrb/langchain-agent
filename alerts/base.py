"""Abstract alert provider interface.

Strategy Pattern: swap Discord ↔ Slack ↔ Teams by changing the
concrete class wired in services/container.py.
"""

import hashlib
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from utils.exceptions import Severity

from config.constants import (
    DEDUP_COOLDOWN_SECONDS,
    RATE_LIMIT_WINDOW_SECONDS,
    TRUNCATE_FINGERPRINT,
)
from infrastructure.logging import logger


# Severities that trigger alerts — hardcoded, not env-configurable.
ENABLED_SEVERITIES: Set[str] = {"ERROR", "CRITICAL"}


class AlertProviderBase(ABC):
    """Base class with shared rate limiting, dedup, and severity filtering.

    Subclasses only implement:
    - `_build_payload()` — transport-specific JSON/body construction
    - `_send()` — transport-specific HTTP call
    """

    ENABLED_SEVERITIES = ENABLED_SEVERITIES

    def __init__(
        self,
        rate_limit_per_minute: int = 5,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
        dedup_cooldown: int = DEDUP_COOLDOWN_SECONDS,
    ) -> None:
        self._rate_limit = rate_limit_per_minute
        self._window_seconds = window_seconds
        self._dedup_cooldown = dedup_cooldown
        # fingerprint -> list of timestamps (sliding window)
        self._sliding_window: Dict[str, list] = {}
        # fingerprint -> last_sent_timestamp (dedup cooldown)
        self._dedup_cache: Dict[str, float] = {}

    @abstractmethod
    async def send_alert(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an alert via the concrete backend."""
        ...

    @abstractmethod
    def _build_payload(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception],
        metadata: Optional[Dict[str, Any]],
    ) -> dict:
        """Build transport-specific payload."""
        ...

    @abstractmethod
    async def _send(self, payload: dict) -> None:
        """Send payload to transport."""
        ...

    def _fingerprint(
        self, severity: Severity, message: str, error: Optional[Exception]
    ) -> str:
        """Create a stable fingerprint for deduplication."""
        error_type = type(error).__name__ if error is not None else "NO_ERROR"
        key = f"{severity.value}:{error_type}:{message[:TRUNCATE_FINGERPRINT]}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _rate_limited(self, fingerprint: str, now: float) -> bool:
        """Sliding-window rate limiter. Returns True if allowed."""
        if fingerprint not in self._sliding_window:
            self._sliding_window[fingerprint] = []

        window = self._sliding_window[fingerprint]
        cutoff = now - self._window_seconds

        while window and window[0] < cutoff:
            window.pop(0)

        if len(window) >= self._rate_limit:
            return False

        window.append(now)
        return True

    def _deduplicated(self, fingerprint: str, now: float) -> bool:
        """Cooldown-based dedup. Returns True if allowed (not seen recently)."""
        last_sent = self._dedup_cache.get(fingerprint)
        if last_sent is not None and (now - last_sent) < self._dedup_cooldown:
            return False

        self._dedup_cache[fingerprint] = now
        return True

    def _should_send(self, severity: Severity, fingerprint: str) -> bool:
        """Check whether this alert should be dispatched."""
        if severity.value not in self.ENABLED_SEVERITIES:
            return False

        now = time.time()

        if not self._deduplicated(fingerprint, now):
            logger.debug(
                f"Alert dedup'd: fingerprint={fingerprint[:12]}, "
                f"last_sent={datetime.fromtimestamp(now, tz=timezone.utc).isoformat()}"
            )
            return False

        if not self._rate_limited(fingerprint, now):
            logger.debug(
                f"Alert rate-limited: fingerprint={fingerprint[:12]}, "
                f"limit={self._rate_limit}"
            )
            return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Backwards-compatible alias
AlertProvider = AlertProviderBase



