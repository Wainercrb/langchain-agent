"""Embeddings providers — Strategy Pattern for pluggable embedding systems."""

from .google import GoogleEmbeddingsWrapper

__all__ = ["GoogleEmbeddingsWrapper"]
