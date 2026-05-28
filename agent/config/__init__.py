"""Configuration package.

Loads .env into os.environ BEFORE any LangChain imports so that
LangSmith / LANGCHAIN_TRACING_V2 auto-detection works correctly.
pydantic-settings reads .env into the Settings object, but does NOT
export vars to os.environ — LangChain's auto-tracing reads os.environ.
"""

from dotenv import load_dotenv

load_dotenv()

from .settings import Settings, settings

__all__ = ["Settings", "settings"]
