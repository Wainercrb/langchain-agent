"""Discord webhook alert provider with rate limiting and dedup decorators.

Combines transport (Discord embeds) with flow control (severity filter,
sliding-window rate limit, fingerprint dedup) in one class.

Normal usage:
    from infrastructure.alerts import DiscordAlertProvider, Severity
    provider = DiscordAlertProvider(
        webhook_url="...",
        rate_limit_per_minute=5,
        enabled_severities="ERROR,CRITICAL",
    )
    await provider.send_alert(Severity.ERROR, "Database timeout", error=exc)
"""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from infrastructure.logging import logger
from utils.exceptions import Severity

from .base import AlertProvider

# Severities that trigger Discord alerts — hardcoded, not env-configurable.
# Add or remove levels here (code change) rather than via environment variables.
ENABLED_SEVERITIES = {"ERROR", "CRITICAL"}


# ── Decorator Functions ──────────────────────────────────────────────


def _rate_limited(
    fingerprint: str,
    window: list,
    rate_limit: int,
    window_seconds: int,
    now: float,
) -> bool:
    """Sliding-window rate limiter.

    Returns True if the call is allowed (within limit), False if rate-limited.
    Mutates *window* in place by appending *now* and purging expired entries.
    """
    cutoff = now - window_seconds
    # Purge expired entries
    while window and window[0] < cutoff:
        window.pop(0)

    if len(window) >= rate_limit:
        return False

    window.append(now)
    return True


def _deduplicated(
    fingerprint: str,
    dedup_cache: dict,
    cooldown: int,
    now: float,
) -> bool:
    """Cooldown-based deduplication.

    Returns True if the call is allowed (not seen recently), False if deduplicated.
    Mutates *dedup_cache* in place by storing *now* for *fingerprint*.
    """
    last_sent = dedup_cache.get(fingerprint)
    if last_sent is not None and (now - last_sent) < cooldown:
        return False

    dedup_cache[fingerprint] = now
    return True


# ── Main Provider Class ──────────────────────────────────────────────


class DiscordAlertProvider(AlertProvider):
    """Sends alerts as Discord embeds with rate limiting and dedup."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        rate_limit_per_minute: int = 5,
    ) -> None:
        self._webhook_url = webhook_url
        self._rate_limit = rate_limit_per_minute
        self._enabled_severities = ENABLED_SEVERITIES
        self._window_seconds = 60
        # fingerprint -> list of timestamps (sliding window)
        self._sliding_window: Dict[str, list] = {}
        # fingerprint -> last_sent_timestamp (dedup cooldown)
        self._dedup_cache: Dict[str, float] = {}
        self._dedup_cooldown = 300  # 5 minutes

    # ── Public API ────────────────────────────────────────────────────

    async def send_alert(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send alert to Discord if it passes severity, dedup, and rate limit filters."""
        fingerprint = self._fingerprint(severity, message, error)

        if not self._should_send(severity, fingerprint):
            return

        if not self._webhook_url:
            logger.warning(
                f"DiscordAlertProvider: no webhook URL. "
                f"severity={severity.value}, message={message[:80]}"
            )
            return

        payload = self._build_payload(severity, message, error, metadata)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self._webhook_url, json=payload)
                response.raise_for_status()
                logger.info(f"Discord alert sent: severity={severity.value}")
        except Exception as e:
            logger.error(f"Discord webhook failed: {e}")

    # ── Rate Limiting / Dedup (composes decorator functions) ──────────

    def _should_send(self, severity: Severity, fingerprint: str) -> bool:
        """Check whether this alert should be dispatched.

        Fast-path: skip if severity is disabled.
        Dedup path: skip if same fingerprint sent within cooldown window.
        Rate-limit path: skip if sliding window for fingerprint is full.
        """
        # Severity gate
        if severity.value not in self._enabled_severities:
            return False

        now = time.time()

        # Dedup gate
        if not _deduplicated(
            fingerprint,
            self._dedup_cache,
            self._dedup_cooldown,
            now,
        ):
            logger.debug(
                f"Alert dedup'd: fingerprint={fingerprint[:12]}, "
                f"last_sent={datetime.fromtimestamp(now, tz=timezone.utc).isoformat()}"
            )
            return False

        # Sliding-window rate-limit gate
        if fingerprint not in self._sliding_window:
            self._sliding_window[fingerprint] = []

        window = self._sliding_window[fingerprint]
        if not _rate_limited(
            fingerprint, window, self._rate_limit, self._window_seconds, now
        ):
            logger.debug(
                f"Alert rate-limited: fingerprint={fingerprint[:12]}, "
                f"current_count={len(window)}, limit={self._rate_limit}"
            )
            return False

        return True

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(
        severity: Severity, message: str, error: Optional[Exception]
    ) -> str:
        """Create a stable fingerprint for deduplication."""
        error_type = type(error).__name__ if error is not None else "NO_ERROR"
        key = f"{severity.value}:{error_type}:{message[:50]}"
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def _build_payload(
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Build the JSON payload for Discord embeds."""
        color_map = {
            Severity.INFO: 0x3498DB,  # blue
            Severity.WARNING: 0xF39C12,  # amber
            Severity.ERROR: 0xE74C3C,  # red
            Severity.CRITICAL: 0x992D22,  # dark red
        }

        embed: Dict[str, Any] = {
            "title": f"[{severity.value}] {message[:80]}",
            "description": message[:2000],
            "color": color_map.get(severity, 0x000000),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        fields: list = []

        if error is not None:
            fields.append(
                {"name": "Error Type", "value": type(error).__name__, "inline": True}
            )
            fields.append(
                {"name": "Details", "value": str(error)[:1000], "inline": False}
            )

        if metadata:
            for k, v in metadata.items():
                fields.append({"name": str(k), "value": str(v)[:200], "inline": True})

        if fields:
            embed["fields"] = fields

        return {"embeds": [embed]}
