"""Alert services — AlertProvider ABC + Discord concrete.

Normal usage:
    from infrastructure.alerts import DiscordAlertProvider, Severity
    provider = DiscordAlertProvider(webhook_url="...")
    await provider.send_alert(Severity.ERROR, "Database timeout", error=exc)

Future: add SlackAlertProvider or TeamsAlertProvider extending AlertProvider.
"""

from .base import AlertProvider
from .discord import DiscordAlertProvider

__all__ = ["AlertProvider", "DiscordAlertProvider"]
