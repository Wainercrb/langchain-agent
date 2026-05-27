"""Embedding generation with batch processing and retry logic."""

import logging
import time
from collections import deque
from typing import List

import numpy as np

from .base import Embeddings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API quotas (requests per minute)."""

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = deque()  # Track request times (last 60 seconds)
        self.min_interval = (
            60.0 / requests_per_minute
        )  # Seconds between requests (~0.6s for 100/min)

    def wait_if_needed(self):
        """Check quota and wait if necessary to stay within rate limit."""
        current_time = time.time()

        # Remove timestamps older than 60 seconds
        while self.request_timestamps and self.request_timestamps[0] < current_time - 60:
            self.request_timestamps.popleft()

        # Check if we need to wait for quota reset
        if len(self.request_timestamps) >= self.requests_per_minute:
            oldest_request = self.request_timestamps[0]
            wait_time = (oldest_request + 60) - current_time
            if wait_time > 0:
                logger.warning(
                    f"Quota limit reached ({len(self.request_timestamps)}/{self.requests_per_minute}). "
                    f"Waiting {wait_time:.1f}s for minute to elapse..."
                )
                time.sleep(wait_time)
                # Clear old timestamps after waiting
                current_time = time.time()
                while self.request_timestamps and self.request_timestamps[0] < current_time - 60:
                    self.request_timestamps.popleft()

        # Also implement minimum spacing between requests to distribute evenly
        if self.request_timestamps:
            last_request = self.request_timestamps[-1]
            time_since_last = current_time - last_request
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
                current_time = time.time()

        # Record this request
        self.request_timestamps.append(current_time)

    def get_request_count_this_minute(self) -> int:
        """Get current request count in the last minute."""
        current_time = time.time()
        while self.request_timestamps and self.request_timestamps[0] < current_time - 60:
            self.request_timestamps.popleft()
        return len(self.request_timestamps)


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
        Reduce embedding dimensions from 3072 to 1536 using PCA-like truncation.

        Args:
            embedding: Original embedding vector (3072 dimensions)

        Returns:
            Reduced embedding vector (1536 dimensions)
        """
        if len(embedding) <= self.target_dimension:
            return embedding

        # Simple truncation method: take first 1536 dimensions
        # (More sophisticated: could use PCA, but truncation works well in practice)
        embedding_array = np.array(embedding, dtype=np.float32)
        reduced = embedding_array[: self.target_dimension]

        # Normalize to maintain magnitude
        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced = reduced / norm

        return reduced.tolist()

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
        from config import settings
        from utils.exceptions import EmbeddingError

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            retry_count = 0

            while retry_count < settings.embedding_retries:
                try:
                    # Use genai.embed_content() for batch embeddings
                    for text_idx, text in enumerate(batch):
                        # Check rate limit before each text (since each text = 1 API request)
                        self.rate_limiter.wait_if_needed()

                        response = self.genai.embed_content(
                            model=f"models/{self.model}",
                            content=text,
                        )
                        embedding = response["embedding"]

                        # Reduce dimension if needed (3072 -> 1536)
                        if len(embedding) > self.target_dimension:
                            embedding = self._reduce_embedding_dimension(embedding)

                        results.append(embedding)

                    batch_num = i // batch_size + 1
                    current_count = self.rate_limiter.get_request_count_this_minute()
                    logger.info(
                        f"Embedded batch {batch_num} ({len(batch)} texts) "
                        f"[quota: {current_count}/{self.rate_limiter.requests_per_minute} this minute]"
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= settings.embedding_retries:
                        logger.error(
                            f"Failed to embed batch after {settings.embedding_retries} retries: {str(e)}"
                        )
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
        from config import settings
        from utils.exceptions import EmbeddingError

        retry_count = 0

        while retry_count < settings.embedding_retries:
            try:
                # Check rate limit and wait if needed
                self.rate_limiter.wait_if_needed()

                response = self.genai.embed_content(
                    model=f"models/{self.model}",
                    content=text,
                )
                embedding = response["embedding"]

                # Reduce dimension if needed (3072 -> 1536)
                if len(embedding) > self.target_dimension:
                    embedding = self._reduce_embedding_dimension(embedding)

                logger.info("Successfully embedded query")
                return embedding
            except Exception as e:
                retry_count += 1
                if retry_count >= settings.embedding_retries:
                    logger.error(
                        f"Failed to embed query after {settings.embedding_retries} retries: {str(e)}"
                    )
                    raise EmbeddingError(
                        message=f"Failed to embed query: {str(e)}",
                        error_code="EMBEDDING_QUERY_FAILED",
                    )
                wait_time = 2**retry_count
                logger.warning(f"Query embedding failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
