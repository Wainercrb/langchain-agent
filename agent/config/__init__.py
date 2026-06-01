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


__all__ = ["Settings", "settings", "configure_tracing"]
