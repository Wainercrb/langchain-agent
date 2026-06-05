"""Discord webhook alert provider.

Sends alerts as Discord embeds. Rate limiting and dedup are handled
by AlertProviderBase.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from config import settings
from config.constants import (
    TRUNCATE_DESCRIPTION,
    TRUNCATE_ERROR_DETAIL,
    TRUNCATE_METADATA,
    TRUNCATE_TITLE,
)
from loggers import logger
from shared.exceptions import Severity

from .base import AlertProviderBase


class DiscordAlertProvider(AlertProviderBase):
    """Sends alerts as Discord embeds with rate limiting and dedup."""

    _enabled: bool = True

    @classmethod
    def is_configured(cls) -> bool:
        return bool(settings.discord_webhook_url)

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        rate_limit_per_minute: int = 5,
    ) -> None:
        super().__init__(rate_limit_per_minute=rate_limit_per_minute)
        self._webhook_url = webhook_url or settings.discord_webhook_url
        if not self._webhook_url:
            self._enabled = False
            logger.warning("DiscordAlertProvider: no webhook URL configured, alerts disabled")

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
            await self._send(payload)
            logger.info(f"Discord alert sent: severity={severity.value}")
        except Exception as e:
            logger.error(f"Discord webhook failed: {e}")

    async def _send(self, payload: dict) -> None:
        """Send payload to Discord webhook."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self._webhook_url, json=payload)
            response.raise_for_status()

    @staticmethod
    def _build_payload(
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Build the JSON payload for Discord embeds."""
        color_map = {
            Severity.INFO: 0x3498DB,
            Severity.WARNING: 0xF39C12,
            Severity.ERROR: 0xE74C3C,
            Severity.CRITICAL: 0x992D22,
        }

        embed: Dict[str, Any] = {
            "title": f"[{severity.value}] {message[:TRUNCATE_TITLE]}",
            "description": message[:TRUNCATE_DESCRIPTION],
            "color": color_map.get(severity, 0x000000),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        fields: list = []

        if error is not None:
            fields.append(
                {"name": "Error Type", "value": type(error).__name__, "inline": True}
            )
            fields.append(
                {"name": "Details", "value": str(error)[:TRUNCATE_ERROR_DETAIL], "inline": False}
            )

        if metadata:
            for k, v in metadata.items():
                fields.append({"name": str(k), "value": str(v)[:TRUNCATE_METADATA], "inline": True})

        if fields:
            embed["fields"] = fields

        return {"embeds": [embed]}
