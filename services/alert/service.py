"""Alert service for success/failure notifications."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AlertService:
    """Send alerts via console, Discord webhook, or email."""

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        alert_email: Optional[str] = None,
    ):
        self.discord_webhook_url = discord_webhook_url
        self.alert_email = alert_email
        logger.info(
            f"AlertService initialized (discord={bool(discord_webhook_url)}, email={bool(alert_email)})"
        )

    def send_success_alert(
        self,
        filename: str,
        chunk_count: int,
        document_id: str,
        version_date: str,
    ) -> None:
        message = (
            f"Successfully ingested: {filename}\n"
            f"   Chunks: {chunk_count}\n"
            f"   Document ID: {document_id[:8]}...\n"
            f"   Version: {version_date}"
        )

        logger.info(f"Success alert: {filename}")
        self._send_console_alert(message, level="INFO")

        if self.discord_webhook_url:
            self._send_discord_alert(message, color=3066993)

    def send_failure_alert(
        self,
        filename: str,
        error_message: str,
        error_code: Optional[str] = None,
    ) -> None:
        message = f"Failed to ingest: {filename}\n" f"   Error: {error_message}"
        if error_code:
            message += f"\n   Code: {error_code}"

        logger.warning(f"Failure alert: {filename}")
        self._send_console_alert(message, level="WARNING")

        if self.discord_webhook_url:
            self._send_discord_alert(message, color=15158332)

    def send_batch_alert(self, summary: Dict[str, Any]) -> None:
        """Send alert for batch ingestion results."""
        message = (
            f"Batch ingestion complete\n"
            f"   Total: {summary['total_files']}\n"
            f"   Successful: {summary['successful']}\n"
            f"   Failed: {summary['failed']}\n"
            f"   Chunks: {summary['ingested_chunks']}"
        )

        # Determine color based on success rate
        success_rate = (
            summary["successful"] / summary["total_files"] if summary["total_files"] > 0 else 0
        )
        if success_rate == 1.0:
            color = 3066993
        elif success_rate >= 0.5:
            color = 15105570
        else:
            color = 15158332

        logger.info(f"Batch alert: {summary['successful']}/{summary['total_files']} successful")
        self._send_console_alert(message, level="INFO")

        if self.discord_webhook_url:
            self._send_discord_alert(message, color=color)

    def _send_console_alert(self, message: str, level: str = "INFO") -> None:
        """Send alert to console."""
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.error(message)

    def _send_discord_alert(self, message: str, color: int = 3066993) -> None:
        """Send alert to Discord webhook."""
        if not self.discord_webhook_url:
            return

        try:
            import requests

            payload = {
                "embeds": [
                    {
                        "description": message,
                        "color": color,
                        "title": "Document Ingestion Alert",
                    }
                ]
            }

            response = requests.post(self.discord_webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                logger.debug("Discord alert sent successfully")
            else:
                logger.warning(f"Discord alert failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {str(e)}")

    def _send_email_alert(self, subject: str, message: str) -> None:
        """Send alert via email."""
        if not self.alert_email:
            return

        try:
            # TODO: Configure SMTP settings from environment
            logger.warning("Email alerts not yet configured")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
