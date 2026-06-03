"""Slack webhook alert provider.

Sends alerts as Slack block kit messages. Rate limiting and dedup are handled
by AlertProviderBase.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from config.constants import (
    TRUNCATE_DESCRIPTION,
    TRUNCATE_ERROR_DETAIL,
    TRUNCATE_METADATA,
    TRUNCATE_TITLE,
)
from infrastructure.logging import logger
from utils.exceptions import Severity

from .base import AlertProviderBase


class SlackAlertProvider(AlertProviderBase):
    """Sends alerts as Slack block kit messages with rate limiting and dedup."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        rate_limit_per_minute: int = 5,
    ) -> None:
        super().__init__(rate_limit_per_minute=rate_limit_per_minute)
        self._webhook_url = webhook_url

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
            await self._send(payload)
            logger.info(f"Slack alert sent: severity={severity.value}")
        except Exception as e:
            logger.error(f"Slack webhook failed: {e}")

    async def _send(self, payload: dict) -> None:
        """Send payload to Slack webhook."""
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
                    "text": f"[{severity.value}] {message[:TRUNCATE_TITLE]}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message[:TRUNCATE_DESCRIPTION],
                },
            },
        ]

        fields = []
        if error is not None:
            fields.append(
                {"type": "mrkdwn", "text": f"*Error Type:* {type(error).__name__}"}
            )
            fields.append(
                {"type": "mrkdwn", "text": f"*Details:* {str(error)[:TRUNCATE_ERROR_DETAIL]}"}
            )

        if metadata:
            for k, v in metadata.items():
                fields.append({"type": "mrkdwn", "text": f"*{k}:* {str(v)[:TRUNCATE_METADATA]}"})

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
