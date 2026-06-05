"""Tool artifact summary registry — OCP-compliant formatter dispatch.

Each tool module registers its own summary formatter by tool name.
``_format_tool_summary`` in ``core/tool_calling`` reads from this
registry, so adding a new tool never requires changing existing code.

Usage::

    # tools/my_tool.py
    from .summaries import register as register_summary

    def _my_summary(artifact):
        ...

    register_summary("my_tool", _my_summary)
"""

from typing import Any, Callable, Optional

# Registry: tool_name -> callable(artifact) -> Optional[str]
# Returning None or empty str falls through to the generic handler.
_summarizers: dict[str, Callable[[Any], Optional[str]]] = {}


def register(tool_name: str, fn: Callable[[Any], Optional[str]]) -> None:
    """Register a summary formatter for *tool_name*.

    Args:
        tool_name: Matches the tool's ``name`` attribute.
        fn: Called with the ``ToolMessage.artifact`` value. Return a
            summary string, or ``None`` to fall back to the generic handler.
    """
    _summarizers[tool_name] = fn


def summarize(tool_name: str, artifact: Any) -> str:
    """Produce a human-readable one-line summary for a tool result.

    Tries the registered formatter first; falls back to a generic
    handler that counts list items.
    """
    fn = _summarizers.get(tool_name)
    if fn is not None and artifact is not None:
        result = fn(artifact)
        if result:
            return result

    # Generic fallback
    if isinstance(artifact, list):
        return f"Found {len(artifact)} results"
    return "Done"
