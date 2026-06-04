"""Core orchestration — agent strategies, alert dispatcher, and LLM router.

The ``core`` package hosts the orchestration layer:

- ``base`` — the Agent ABC (lightweight, safe to re-export).
- ``tool_calling`` — the concrete ``ToolCallingAgent`` strategy.
- ``dispatcher`` — multi-provider alert dispatcher (imports from ``alerts/``).
- ``router`` — multi-provider LLM router with circuit breaker (imports from ``llm/``).

Public imports:

    from core import Agent                       # re-exported from base
    from core.tool_calling import ToolCallingAgent   # explicit submodule
    from core.dispatcher import MultiAlertProvider   # explicit submodule
    from core.router import MultiProviderLLM         # explicit submodule

The two orchestrators and the tool_calling strategy are NOT re-exported from
``core.__init__`` on purpose: keeping their submodule path explicit makes
their dependency on the leaf packages (``alerts/``, ``llm/``) visible at the
import site, and avoids loading the heavy ``tool_calling`` module when only
``core.router`` is needed (prevents triggering the pre-existing
``loggers`` ↔ ``shared`` circular import at container startup).
"""

from .base import Agent

__all__ = ["Agent"]
