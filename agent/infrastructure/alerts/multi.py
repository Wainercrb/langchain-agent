"""Multi-provider alert dispatcher — sends to all configured backends in parallel.

Normal usage:
    from infrastructure.alerts import MultiAlertProvider, DiscordAlertProvider, SlackAlertProvider
    multi = MultiAlertProvider([
        DiscordAlertProvider(webhook_url="discord-url"),
        SlackAlertProvider(webhook_url="slack-url"),
    ])
    await multi.send_alert(Severity.ERROR, "Something broke", error=exc)
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from infrastructure.logging import logger
from utils.exceptions import Severity


class AlertProvider(ABC):
    """Simple ABC for alert providers that don't need rate limiting/dedup."""

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MultiAlertProvider(AlertProvider):
    """Dispatches alerts to multiple providers in parallel."""

    def __init__(self, providers: List[AlertProvider]) -> None:
        self._providers = providers

    async def send_alert(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send alert to all configured providers in parallel."""
        if not self._providers:
            logger.warning("MultiAlertProvider: no providers configured")
            return

        tasks = []
        for provider in self._providers:
            tasks.append(self._safe_send(provider, severity, message, error, metadata))

        await asyncio.gather(*tasks)

    @staticmethod
    async def _safe_send(
        provider: AlertProvider,
        severity: Severity,
        message: str,
        error: Optional[Exception],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Send alert to a single provider, catching any exceptions."""
        try:
            await provider.send_alert(severity, message, error, metadata)
        except Exception as e:
            logger.error(
                f"MultiAlertProvider: {type(provider).__name__} failed: {e}"
            )
