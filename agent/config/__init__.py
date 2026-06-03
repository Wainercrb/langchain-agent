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


def is_langsmith_enabled() -> bool:
    """Return True if LangSmith tracing is enabled and configured with an API key."""
    return settings.enable_langsmith_tracing and bool(settings.langsmith_api_key)


def get_langsmith_dashboard_url() -> str | None:
    """Return the LangSmith dashboard URL or None if tracing is not configured."""
    if not is_langsmith_enabled():
        return None
    project = settings.langsmith_project or "langchain-agent"
    return f"https://smith.langchain.com/o/default/projects/p/{project}"


__all__ = ["Settings", "settings", "is_langsmith_enabled", "get_langsmith_dashboard_url"]
