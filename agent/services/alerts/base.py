"""Abstract alert provider interface.

Strategy Pattern: swap Discord ↔ Slack ↔ Teams cambiando la clase
concreta que se cablea en services/container.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from utils.exceptions import Severity


class AlertProvider(ABC):
    """Abstract base for pluggable alert backends.

    Cada implementación decide cómo se envía la alerta
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
            severity: Nivel de severidad del alerta.
            message: Mensaje legible por humanos.
            error: Excepción original (opcional).
            metadata: Datos extra (opcional).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
