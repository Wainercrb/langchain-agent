"""Embeddings providers — Strategy Pattern for pluggable embedding systems.

Each provider implements Embeddings (ABC). Instantiate directly:

    from embeddings import GoogleEmbeddingsWrapper
    
    # Change this line to swap providers
    embeddings = GoogleEmbeddingsWrapper(api_key="...")
    # embeddings = OpenAIEmbeddingsProvider(api_key="...")
    # embeddings = HuggingFaceEmbeddingsProvider(model="...")
"""

from .base import Embeddings
from .google import GoogleEmbeddingsWrapper

__all__ = ["Embeddings", "GoogleEmbeddingsWrapper"]
