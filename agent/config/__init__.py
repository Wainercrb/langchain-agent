"""Configuration package.

Loads .env into os.environ BEFORE any LangChain imports so that
LangSmith / LANGCHAIN_TRACING_V2 auto-detection works correctly.
pydantic-settings reads .env into the Settings object, but does NOT
export vars to os.environ — LangChain's auto-tracing reads os.environ.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .settings import Settings, settings  # noqa: E402 — load_dotenv() must run first


def configure_tracing() -> None:
    """Set LangSmith environment variables for LangChain auto-detection.

    LangChain reads LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT from
    os.environ at import time. This function sets defaults so that
    tracing works even if the user hasn't added them to .env.
    Call early in the application lifecycle (before LangChain imports).

    Supports legacy env vars: LANGSMITH_TRACING, LANGCHAIN_TRACING_V2.
    """
    enabled = settings.enable_langsmith_tracing
    if not enabled:
        legacy = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false"))
        enabled = legacy.lower() in ("true", "1", "yes", "on")

    if enabled:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault(
            "LANGCHAIN_PROJECT",
            settings.langsmith_project or "langchain-agent",
        )


def is_langsmith_enabled() -> bool:
    """Return True if LangSmith tracing is enabled and configured with an API key."""
    return settings.enable_langsmith_tracing and bool(settings.langsmith_api_key)


def get_langsmith_dashboard_url() -> str | None:
    """Return the LangSmith dashboard URL or None if tracing is not configured."""
    if not is_langsmith_enabled():
        return None
    project = settings.langsmith_project or "langchain-agent"
    return f"https://smith.langchain.com/o/default/projects/p/{project}"


__all__ = ["Settings", "settings", "configure_tracing", "is_langsmith_enabled", "get_langsmith_dashboard_url"]
