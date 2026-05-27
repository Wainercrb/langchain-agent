"""Abstract LLM provider interface for pluggable AI backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Message(dict):
    """Message data structure with role and content."""

    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)
        self.role = role
        self.content = content


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Enables seamless switching between different LLM backends (Gemini, OpenAI, Anthropic)
    while maintaining consistent interface throughout the application.
    
    Design Pattern: Strategy pattern + Dependency Injection
    """

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        """
        Initialize LLM provider.
        
        Args:
            model: Model identifier (e.g., "gemini-2.5-flash", "gpt-4", "claude-3-opus")
            temperature: Generation temperature (0.0-1.0)
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.temperature = temperature
        self.config = kwargs

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> "LLMResponse":
        """
        Generate response from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        
        Returns:
            LLMResponse: Response object with content and metadata
            
        Raises:
            LLMProviderError: If provider fails or API error occurs
        """
        pass

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]]):
        """
        Generate streaming response from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
        
        Yields:
            str: Streamed response chunks
            
        Raises:
            LLMProviderError: If provider fails or API error occurs
        """
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        """
        Validate that API credentials are available and valid.
        
        Returns:
            bool: True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get human-readable provider name."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, temperature={self.temperature})"


class LLMResponse:
    """Standardized response from LLM provider."""

    def __init__(self, content: str, model: str, provider: str, usage: Optional[Dict[str, int]] = None, **metadata):
        self.content = content
        self.model = model
        self.provider = provider
        self.usage = usage or {}  # token counts
        self.metadata = metadata

    def __str__(self) -> str:
        return self.content


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
