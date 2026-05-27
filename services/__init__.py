"""Services layer - abstracted providers, builders, backends, and utility services."""

# Organized services (document, state, scheduler, alert, file)
from .document import DocumentIngester, DocumentProcessor, ParserFactory
from .state import StateTracker, VersionManager
from .scheduler import CronScheduler
from .alert import AlertService
from .file import FileManager

# New abstraction layers (providers, builders, backends)
from .llm import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    LLMProviderError,
    OpenAIProvider,
    create_llm_provider,
    get_provider_info,
)
from .logging_backend import (
    JSONLoggerBackend,
    LoggerBackend,
    create_logger_backend,
    setup_global_logging,
)
from .messaging import (
    LangChainMessageBuilder,
    MessageBuilder,
    create_message_builder,
)

__all__ = [
    # Document services
    "DocumentIngester",
    "DocumentProcessor",
    "ParserFactory",
    # State services
    "StateTracker",
    "VersionManager",
    # Scheduler services
    "CronScheduler",
    # Alert services
    "AlertService",
    # File services
    "FileManager",
    # LLM Providers
    "LLMProvider",
    "LLMProviderError",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_llm_provider",
    "get_provider_info",
    # Message Builders
    "MessageBuilder",
    "LangChainMessageBuilder",
    "create_message_builder",
    # Logger Backends
    "LoggerBackend",
    "JSONLoggerBackend",
    "create_logger_backend",
    "setup_global_logging",
]

