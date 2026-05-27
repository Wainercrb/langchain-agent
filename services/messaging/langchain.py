"""Standard LangChain-compatible message builder implementation."""

import logging
from typing import Dict, List, Optional

from .base import MessageBuilder

logger = logging.getLogger(__name__)


class LangChainMessageBuilder(MessageBuilder):
    """
    Standard message builder compatible with LangChain format.
    
    Follows the role-based message format:
    - "system": System instructions/prompts
    - "user": User messages/queries
    - "assistant": Assistant responses
    
    This format is used by OpenAI, Anthropic, and other providers through LangChain.
    """

    def build_system_message(self, content: str) -> Dict[str, str]:
        """Build a system message."""
        return {"role": "system", "content": content}

    def build_user_message(self, content: str) -> Dict[str, str]:
        """Build a user message."""
        return {"role": "user", "content": content}

    def build_assistant_message(self, content: str) -> Dict[str, str]:
        """Build an assistant message."""
        return {"role": "assistant", "content": content}

    def build_messages(
        self, system_prompt: str, user_query: str, context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build message chain with system prompt, optional context, and user query.
        
        Message order (importance):
        1. System prompt (instructions)
        2. Context (retrieved documents)
        3. User query (the actual question)
        """
        messages = [self.build_system_message(system_prompt)]

        if context:
            context_msg = f"Context documents:\n\n{context}"
            messages.append(self.build_user_message(context_msg))

        # Final user query
        messages.append(self.build_user_message(user_query))

        return messages

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate messages have required role and content fields.
        
        Args:
            messages: Message list to validate
        
        Returns:
            bool: True if all messages have 'role' and 'content', False otherwise
        """
        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"Message is not a dict: {type(msg)}")
                return False
            if "role" not in msg or "content" not in msg:
                logger.warning(f"Message missing 'role' or 'content': {msg}")
                return False
            if msg["role"] not in ("system", "user", "assistant"):
                logger.warning(f"Invalid role: {msg['role']}")
                return False

        return True
