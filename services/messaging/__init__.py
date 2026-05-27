"""Message builder factory for flexible message formatting."""

import logging

from .base import MessageBuilder
from .langchain import LangChainMessageBuilder

logger = logging.getLogger(__name__)

# Available message builders
BUILDERS = {
    "langchain": LangChainMessageBuilder,
    "default": LangChainMessageBuilder,  # Default to LangChain format
}


def create_message_builder(builder_type: str = "default") -> MessageBuilder:
    """
    Factory function to create message builder instances.
    
    Args:
        builder_type: Builder type identifier ('langchain', 'default')
    
    Returns:
        MessageBuilder: Message builder instance
    
    Raises:
        ValueError: If builder type not found
    
    Example:
        >>> builder = create_message_builder("langchain")
        >>> messages = builder.build_messages(
        ...     system_prompt="You are helpful.",
        ...     user_query="What is AI?"
        ... )
    """
    if builder_type.lower() not in BUILDERS:
        available = ", ".join(BUILDERS.keys())
        raise ValueError(f"Unknown message builder: {builder_type}. Available: {available}")

    builder_class = BUILDERS[builder_type.lower()]
    logger.debug(f"Message builder created: {builder_class.__name__}")
    return builder_class()


__all__ = [
    "MessageBuilder",
    "LangChainMessageBuilder",
    "create_message_builder",
]
