"""Version management for documents with date-based retrieval."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VersionManager:
    """Manage document versioning and version date tracking."""

    @staticmethod
    def generate_version_date(custom_date: Optional[datetime] = None) -> datetime:
        """
        Generate or use custom version date.

        Args:
            custom_date: Optional custom date (default: current UTC time)

        Returns:
            Version date as datetime object
        """
        version_date = custom_date or datetime.utcnow()
        logger.info(f"Generated version date: {version_date.isoformat()}")
        return version_date

    @staticmethod
    def is_newer_version(existing_date: datetime, new_date: datetime) -> bool:
        """
        Check if new date is newer than existing version.

        Args:
            existing_date: Existing version date
            new_date: New version date to compare

        Returns:
            True if new_date > existing_date, False otherwise
        """
        is_newer = new_date > existing_date
        logger.info(f"Version comparison: {new_date.isoformat()} > {existing_date.isoformat()} = {is_newer}")
        return is_newer

    @staticmethod
    def get_version_metadata(
        filename: str,
        version_date: datetime,
        chunk_count: int = 0,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create version metadata dictionary.

        Args:
            filename: Document filename
            version_date: Version date
            chunk_count: Number of chunks in this version
            document_id: Document ID (optional)

        Returns:
            Metadata dictionary with version information
        """
        metadata = {
            "filename": filename,
            "version_date": version_date.isoformat(),
            "chunk_count": chunk_count,
            "created_at": datetime.utcnow().isoformat(),
        }

        if document_id:
            metadata["document_id"] = document_id

        logger.info(f"Created version metadata for {filename}: {chunk_count} chunks")
        return metadata

    @staticmethod
    def extract_version_from_filename(filename: str) -> Optional[datetime]:
        """
        Try to extract version date from filename (e.g., 'doc_2026-05-22.pdf').

        Args:
            filename: Document filename

        Returns:
            Extracted datetime or None if pattern not found
        """
        import re

        # Match patterns like: 2026-05-22, 2026_05_22, 20260522
        date_patterns = [
            r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
            r"(\d{4})_(\d{2})_(\d{2})",  # YYYY_MM_DD
            r"(\d{4})(\d{2})(\d{2})",    # YYYYMMDD
        ]

        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    year, month, day = match.groups()
                    extracted_date = datetime(int(year), int(month), int(day))
                    logger.info(f"Extracted version date from filename: {extracted_date.isoformat()}")
                    return extracted_date
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse extracted date from filename: {filename}")
                    continue

        logger.debug(f"No version date pattern found in filename: {filename}")
        return None
