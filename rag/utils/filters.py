"""Filtering utilities for search results."""

import logging
from datetime import datetime
from typing import Any, Dict, Generator, List

logger = logging.getLogger(__name__)


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
    results: List[Dict[str, Any]], version_date: datetime
) -> Generator[Dict[str, Any], None, None]:
    """
    Filter search results by version date.

    Args:
        results: List of search result dictionaries
        version_date: Minimum version date to include

    Yields:
        Result dictionaries with version_date >= filter date

    Example:
        from datetime import datetime
        min_date = datetime(2025, 1, 1)
        filtered = list(filter_by_version(results, min_date))
    """
    for result in results:
        doc_version = result.get("version_date")
        if doc_version is None:
            logger.debug(
                f"Filtered out document {result.get('document_id')} (version_date is None)"
            )
            continue

        if isinstance(doc_version, datetime):
            doc_version_date = doc_version.date()
        else:
            doc_version_date = doc_version

        if doc_version_date >= version_date.date():
            yield result
        else:
            logger.debug(
                f"Filtered out document {result.get('document_id')} "
                f"(version {doc_version_date} < {version_date.date()})"
            )
