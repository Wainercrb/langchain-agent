"""Abstract alert provider interface.

Strategy Pattern: swap Discord ↔ Slack ↔ Teams by changing the
concrete class wired in services/container.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from utils.exceptions import Severity


class AlertProvider(ABC):
    """Abstract base for pluggable alert backends.

    Each implementation decides how alerts are sent
    (Discord embed, Slack block, Teams card, etc.).
    """

    @abstractmethod
    async def send_alert(
        self,
        severity: Severity,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an alert via the concrete backend.

        Args:
            severity: Alert severity level.
            message: Human-readable message.
            error: Original exception (optional).
            metadata: Extra data (optional).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
