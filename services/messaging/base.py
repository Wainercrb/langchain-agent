"""Abstract message builder interface for flexible message formatting."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class MessageBuilder(ABC):
    """
    Abstract base class for building messages for different LLM providers.
    
    Different providers may have different message formats or requirements.
    This abstraction allows custom message building logic per provider.
    """

    @abstractmethod
    def build_system_message(self, content: str) -> Dict[str, str]:
        """
        Build a system message.
        
        Args:
            content: System prompt content
        
        Returns:
            Message dict with 'role' and 'content'
        """
        pass

    @abstractmethod
    def build_user_message(self, content: str) -> Dict[str, str]:
        """
        Build a user message.
        
        Args:
            content: User query/content
        
        Returns:
            Message dict with 'role' and 'content'
        """
        pass

    @abstractmethod
    def build_assistant_message(self, content: str) -> Dict[str, str]:
        """
        Build an assistant (LLM) message.
        
        Args:
            content: Assistant response
        
        Returns:
            Message dict with 'role' and 'content'
        """
        pass

    @abstractmethod
    def build_messages(
        self, system_prompt: str, user_query: str, context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build complete message chain for LLM.
        
        Args:
            system_prompt: System instruction
            user_query: User query
            context: Optional context (e.g., retrieved documents)
        
        Returns:
            List of message dicts
        """
        pass

    @abstractmethod
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate message format is correct for this builder.
        
        Args:
            messages: Message list to validate
        
        Returns:
            bool: True if valid, False otherwise
        """
        pass
