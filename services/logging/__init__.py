"""Logging services — Logger ABC + Console class.

El container crea la instancia. Los módulos en el camino crítico de imports
crean la suya local (stateless).

Uso normal:
    from services.container import logger
    logger.info("mensaje")
"""

from .base import Logger
from .console import Console

__all__ = ["Logger", "Console"]
