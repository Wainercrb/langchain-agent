"""Alert services — AlertProvider ABC + Discord + Slack + Multi-provider.

Normal usage:
    from infrastructure.alerts import DiscordAlertProvider, Severity
    provider = DiscordAlertProvider(webhook_url="...")
    await provider.send_alert(Severity.ERROR, "Database timeout", error=exc)

Multi-provider (send to all configured backends):
    from infrastructure.alerts import MultiAlertProvider, DiscordAlertProvider, SlackAlertProvider
    multi = MultiAlertProvider([
        DiscordAlertProvider(webhook_url="discord-url"),
        SlackAlertProvider(webhook_url="slack-url"),
    ])
    await multi.send_alert(Severity.ERROR, "Something broke", error=exc)
"""

from .base import AlertProvider
from .discord import DiscordAlertProvider
from .slack import SlackAlertProvider
from .multi import MultiAlertProvider

__all__ = ["AlertProvider", "DiscordAlertProvider", "SlackAlertProvider", "MultiAlertProvider"]
