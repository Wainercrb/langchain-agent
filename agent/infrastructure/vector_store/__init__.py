"""Vector store providers — Strategy Pattern for pluggable vector databases.

Each provider implements VectorStoreBase (ABC). Instantiate directly:

    from infrastructure.vector_store import VectorStore
    
    # Change this line to swap providers
    vector_store = VectorStore(supabase_client)
    # vector_store = PineconeVectorStore(api_key="...")
    # vector_store = QdrantVectorStore(url="...")
"""

from .base import VectorStoreBase
from .supabase import VectorStore

__all__ = ["VectorStoreBase", "VectorStore"]
