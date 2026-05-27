"""Embedding generation with batch processing and retry logic."""

import logging
from typing import List

from config import settings
from utils.exceptions import EmbeddingError

from ..core.base import Embeddings
from ..utils import RateLimiter

logger = logging.getLogger(__name__)


class GoogleEmbeddingsWrapper(Embeddings):
    """Wrapper for Google Embeddings API with batching and retry."""

    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai

            self.api_key = api_key
            genai.configure(api_key=api_key)
            self.model = "gemini-embedding-001"  # Gemini embedding model
            self.genai = genai

            # Initialize rate limiter (100 requests per minute for free tier)
            self.rate_limiter = RateLimiter(requests_per_minute=100)

            # Target dimension for embeddings (database constraint)
            self.target_dimension = 1536

            logger.info(f"GoogleEmbeddingsWrapper initialized with model: {self.model}")
        except ImportError as e:
            logger.error(f"Failed to import google.generativeai: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def _reduce_embedding_dimension(self, embedding: List[float]) -> List[float]:
        """
        Reduce embedding dimensions from 3072 to 1536 using truncation.

        Args:
            embedding: Original embedding vector (3072 dimensions)

        Returns:
            Reduced embedding vector (1536 dimensions)
        """
        if len(embedding) <= self.target_dimension:
            return embedding

        reduced = embedding[: self.target_dimension]
        norm = sum(x * x for x in reduced) ** 0.5
        if norm > 0:
            reduced = [x / norm for x in reduced]

        return reduced

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with rate limiting and dimension reduction."""
        import time

        self.rate_limiter.wait_if_needed()
        response = self.genai.embed_content(model=f"models/{self.model}", content=text)
        embedding = response["embedding"]
        return (
            self._reduce_embedding_dimension(embedding)
            if len(embedding) > self.target_dimension
            else embedding
        )

    def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Embed multiple documents with retry logic and batching.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per batch (default: 10)

        Returns:
            List of embeddings (1536-dimensional vectors)

        Raises:
            Exception: If embedding fails after all retries
        """
        import time

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            retry_count = 0

            while retry_count < settings.embedding_retries:
                try:
                    for text in batch:
                        results.append(self._embed_single(text))

                    batch_num = i // batch_size + 1
                    quota = self.rate_limiter.get_request_count_this_minute()
                    logger.info(
                        f"Embedded batch {batch_num} ({len(batch)} texts) "
                        f"[quota: {quota}/{self.rate_limiter.requests_per_minute} this minute]"
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= settings.embedding_retries:
                        raise EmbeddingError(
                            message=f"Failed to embed batch after {settings.embedding_retries} retries: {str(e)}",
                            error_code="EMBEDDING_MAX_RETRIES",
                            details={"batch_size": len(batch), "retry_count": retry_count},
                        )
                    wait_time = 2**retry_count
                    logger.warning(f"Embedding batch failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)

        logger.info(f"Successfully embedded {len(results)} vectors from {len(texts)} texts")
        return results

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text with retry logic and rate limiting.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (1536-dimensional)

        Raises:
            Exception: If embedding fails after all retries
        """
        import time

        retry_count = 0
        while retry_count < settings.embedding_retries:
            try:
                embedding = self._embed_single(text)
                logger.info("Successfully embedded query")
                return embedding
            except Exception as e:
                retry_count += 1
                if retry_count >= settings.embedding_retries:
                    raise EmbeddingError(
                        message=f"Failed to embed query: {str(e)}",
                        error_code="EMBEDDING_QUERY_FAILED",
                    )
                wait_time = 2**retry_count
                logger.warning(f"Query embedding failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
