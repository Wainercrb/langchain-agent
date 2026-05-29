"""Alert services — AlertProvider ABC + Discord concrete.

Uso normal:
    from services.alerts import DiscordAlertProvider, Severity
    provider = DiscordAlertProvider(webhook_url="...")
    await provider.send_alert(Severity.ERROR, "Database timeout", error=exc)

Mañana: agregá SlackAlertProvider o TeamsAlertProvider heredando de AlertProvider.
"""

from .base import AlertProvider
from .discord import DiscordAlertProvider

__all__ = ["AlertProvider", "DiscordAlertProvider"]
