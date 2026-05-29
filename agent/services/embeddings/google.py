"""Google Embeddings provider — implementa Embeddings usando Google Generative AI."""
import time
from typing import List

from config import settings
from services.embeddings.base import Embeddings
from utils import RateLimiter
from utils.exceptions import EmbeddingError
from services.logging import logger


class GoogleEmbeddingsWrapper(Embeddings):
    """Wrapper for Google Embeddings API with batching and retry."""

    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai

            self.api_key = api_key
            genai.configure(api_key=api_key)
            self.model = "gemini-embedding-001"
            self.genai = genai
            self.rate_limiter = RateLimiter(requests_per_minute=100)
            self.target_dimension = 1536

            logger.info(f"GoogleEmbeddingsWrapper initialized with model: {self.model}")
        except ImportError as e:
            logger.error(f"Failed to import google.generativeai: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def _embed_single(self, text: str) -> List[float]:
        self.rate_limiter.wait_if_needed()
        response = self.genai.embed_content(
            model=f"models/{self.model}",
            content=text,
            output_dimensionality=self.target_dimension,
        )
        embedding = response["embedding"]
        return embedding

    def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
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
