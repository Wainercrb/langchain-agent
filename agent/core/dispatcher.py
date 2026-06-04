"""Multi-provider alert dispatcher — fans alerts out to N backends in parallel.

The dispatcher has exactly one responsibility: given a list of
``AlertProvider`` instances, send an alert to ALL of them concurrently
and isolate per-provider failures so one broken webhook never aborts
the others.

Typical usage::

    from core.dispatcher import MultiAlertProvider
    from alerts import DiscordAlertProvider, SlackAlertProvider

    multi = MultiAlertProvider([
        DiscordAlertProvider(webhook_url="discord-url"),
        SlackAlertProvider(webhook_url="slack-url"),
    ])
    await multi.send_alert(Severity.ERROR, "Something broke", error=exc)
"""

import asyncio
from typing import Dict, List, Optional, Protocol, runtime_checkable

from loggers import logger
from shared.exceptions import Severity


__all__ = ["AlertProvider", "MultiAlertProvider"]


# ── Provider contract ────────────────────────────────────────────────


@runtime_checkable
class AlertProvider(Protocol):
    """Anything that can receive an alert.

    Concrete implementations (Discord, Slack, Email, ...) live in the
    ``alerts/`` package and adapt each backend to this contract.
    Implementations need only expose ``send_alert`` — no inheritance,
    no base class, just structural typing.
    """

    async def send_alert(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Send one alert through this provider's backend."""
        ...


# ── Fan-out dispatcher ───────────────────────────────────────────────


class MultiAlertProvider:
    """Sends one alert to many providers in parallel, isolating failures.

    Each provider runs in its own task via ``asyncio.gather``. A failure
    in any one provider is logged at ``ERROR`` level and swallowed — the
    whole point of the "multi" prefix is that one broken backend never
    takes the others down with it.

    Args:
        providers: Backends that will receive every alert. May be empty
            (in which case ``send_alert`` is a silent no-op).
    """

    def __init__(self, providers: List[AlertProvider]) -> None:
        # Defensive copy: callers should not be able to mutate the list
        # after construction.
        self._providers: List[AlertProvider] = list(providers)
        if not self._providers:
            logger.warning("MultiAlertProvider: no providers configured")

    async def send_alert(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Fan the alert out to every configured provider in parallel."""
        if not self._providers:
            return
        await asyncio.gather(*(
            self._send_safely(provider, severity, message, error, metadata)
            for provider in self._providers
        ))

    async def _send_safely(
        self,
        provider: AlertProvider,
        severity: Severity,
        message: str,
        error: Optional[Exception],
        metadata: Optional[Dict[str, object]],
    ) -> None:
        """Send to a single provider, logging any exception (never raises)."""
        try:
            await provider.send_alert(severity, message, error, metadata)
        except Exception as exc:
            logger.error(
                f"MultiAlertProvider: {type(provider).__name__} failed: {exc}"
            )
