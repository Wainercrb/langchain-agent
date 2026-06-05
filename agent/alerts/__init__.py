"""Alert services — AlertProvider ABC + Discord + Slack.

Normal usage:
    from alerts import DiscordAlertProvider, Severity
    provider = DiscordAlertProvider(webhook_url="...")
    await provider.send_alert(Severity.ERROR, "Database timeout", error=exc)

Multi-provider (send to all configured backends):
    The multi-provider dispatcher lives in :mod:`core.dispatcher`.
    Import it as: ``from core.dispatcher import MultiAlertProvider``
"""

from .base import AlertProviderBase, AlertProvider, ENABLED_SEVERITIES
from .discord import DiscordAlertProvider
from .slack import SlackAlertProvider

__all__ = [
    "AlertProvider",
    "AlertProviderBase",
    "DiscordAlertProvider",
    "SlackAlertProvider",
    "ENABLED_SEVERITIES",
]
