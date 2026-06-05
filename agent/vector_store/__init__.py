"""Vector store providers — Strategy Pattern for pluggable vector databases.

Each provider implements the VectorStoreOps, IngestionLogger, and HealthCheckable
ABCs (Interface Segregation Principle). Instantiate directly:

    from vector_store import VectorStore

    # Change this line to swap providers
    vector_store = VectorStore(supabase_client)
    # vector_store = PineconeVectorStore(api_key="...")
"""

from .base import VectorStoreOps, IngestionLogger, HealthCheckable, VectorStore as VectorStoreABC, VectorStoreBase
from .supabase import VectorStore

__all__ = [
    "VectorStoreOps",
    "IngestionLogger",
    "HealthCheckable",
    "VectorStoreBase",
    "VectorStore",
]
