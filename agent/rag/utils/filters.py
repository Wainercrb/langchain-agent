"""Filtering utilities for search results."""

from datetime import datetime
from typing import Any, Dict, Generator, List

from services.logging import Console

logger = Console()



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
            logger.debug(
                f"Filtered out document {result.get('document_id')} "
                f"(similarity {score} < {threshold})"
            )


def filter_by_version(
    results: List[Dict[str, Any]], min_version: datetime
) -> Generator[Dict[str, Any], None, None]:
    """
    Filter search results by minimum version date.

    Args:
        results: List of search result dictionaries
        min_version: Minimum version date threshold

    Yields:
        Results with version_date >= min_version
    """
    min_date = min_version.date() if isinstance(min_version, datetime) else min_version

    for result in results:
        doc_version = result.get("version_date")
        if doc_version is None:
            continue

        doc_date = doc_version.date() if isinstance(doc_version, datetime) else doc_version

        if doc_date >= min_date:
            yield result
