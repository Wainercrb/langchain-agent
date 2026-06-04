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

# Set LangSmith env vars early so LangChain auto-detection picks them up.
if settings.enable_langsmith_tracing:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        settings.langsmith_project or "langchain-agent",
    )
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)


def is_observability_enabled() -> bool:
    """Return True if an observability backend is configured."""
    return settings.observability_backend != "none"


def is_langsmith_enabled() -> bool:
    """Return True if LangSmith tracing is enabled and configured with an API key."""
    return settings.enable_langsmith_tracing and bool(settings.langsmith_api_key)


def get_observability_dashboard_url() -> str | None:
    """Return the observability dashboard URL or None if not configured."""
    from agent.observability.base import get_observability_provider

    return get_observability_provider().dashboard_url()


# Backward compatibility alias
get_langsmith_dashboard_url = get_observability_dashboard_url


__all__ = [
    "Settings",
    "settings",
    "is_observability_enabled",
    "is_langsmith_enabled",
    "get_observability_dashboard_url",
    "get_langsmith_dashboard_url",
]
