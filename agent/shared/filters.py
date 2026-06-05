"""Filtering utilities for search results."""

from typing import Any, Dict, Generator, List


def filter_by_threshold(
    results: List[Dict[str, Any]], threshold: float
) -> Generator[Dict[str, Any], None, None]:
    """
    Filter search results by similarity threshold.

    Args:
        results: List of search result dictionaries
        threshold: Minimum similarity score (0.0-1.0)

    Yields:
        Result dictionaries that meet or exceed the threshold

    Example:
        filtered = list(filter_by_threshold(results, 0.5))
    """
    for result in results:
        score = result.get("similarity_score", 0.0)
        if score >= threshold:
            yield result
        else:
            from loggers import logger
            logger.debug(
                f"Filtered out document {result.get('document_id')} "
                f"(similarity {score} < {threshold})"
            )
