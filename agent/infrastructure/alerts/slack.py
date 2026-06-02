"""Slack webhook alert provider with rate limiting and dedup decorators.

Combines transport (Slack blocks) with flow control (severity filter,
sliding-window rate limit, fingerprint dedup) in one class.

Normal usage:
    from infrastructure.alerts import SlackAlertProvider, Severity
    provider = SlackAlertProvider(
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
from .discord import ENABLED_SEVERITIES, _rate_limited, _deduplicated


class SlackAlertProvider(AlertProvider):
    """Sends alerts as Slack block kit messages with rate limiting and dedup."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        rate_limit_per_minute: int = 5,
    ) -> None:
        self._webhook_url = webhook_url
        self._rate_limit = rate_limit_per_minute
        self._enabled_severities = ENABLED_SEVERITIES
        self._window_seconds = 60
        self._sliding_window: Dict[str, list] = {}
        self._dedup_cache: Dict[str, float] = {}
        self._dedup_cooldown = 300  # 5 minutes

    async def send_alert(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send alert to Slack if it passes severity, dedup, and rate limit filters."""
        fingerprint = self._fingerprint(severity, message, error)

        if not self._should_send(severity, fingerprint):
            return

        if not self._webhook_url:
            logger.warning(
                f"SlackAlertProvider: no webhook URL. "
                f"severity={severity.value}, message={message[:80]}"
            )
            return

        payload = self._build_payload(severity, message, error, metadata)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self._webhook_url, json=payload)
                response.raise_for_status()
                logger.info(f"Slack alert sent: severity={severity.value}")
        except Exception as e:
            logger.error(f"Slack webhook failed: {e}")

    def _should_send(self, severity: Severity, fingerprint: str) -> bool:
        """Check whether this alert should be dispatched."""
        if severity.value not in self._enabled_severities:
            return False

        now = time.time()

        if not _deduplicated(
            fingerprint,
            self._dedup_cache,
            self._dedup_cooldown,
            now,
        ):
            return False

        if fingerprint not in self._sliding_window:
            self._sliding_window[fingerprint] = []

        window = self._sliding_window[fingerprint]
        if not _rate_limited(
            fingerprint, window, self._rate_limit, self._window_seconds, now
        ):
            return False

        return True

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
        """Build the JSON payload for Slack block kit."""
        color_map = {
            Severity.INFO: "#3498DB",
            Severity.WARNING: "#F39C12",
            Severity.ERROR: "#E74C3C",
            Severity.CRITICAL: "#992D22",
        }

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"[{severity.value}] {message[:80]}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message[:2000],
                },
            },
        ]

        fields = []
        if error is not None:
            fields.append(
                {"type": "mrkdwn", "text": f"*Error Type:* {type(error).__name__}"}
            )
            fields.append(
                {"type": "mrkdwn", "text": f"*Details:* {str(error)[:1000]}"}
            )

        if metadata:
            for k, v in metadata.items():
                fields.append({"type": "mrkdwn", "text": f"*{k}:* {str(v)[:200]}"})

        if fields:
            blocks.append({
                "type": "section",
                "fields": fields,
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "plain_text",
                    "text": f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
                }
            ],
        })

        return {
            "blocks": blocks,
            "attachments": [
                {"color": color_map.get(severity, "#000000")}
            ],
        }
