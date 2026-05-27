"""Track processing state and maintain deduplication state."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StateTracker:
    """Track processed files and maintain deduplication state in JSON."""

    def __init__(self, tracking_file: Optional[Path] = None):
        from config import settings
        from utils.file_utils import load_json_file

        if tracking_file is None:
            tracking_file = settings.knowledge_dir / ".processed_files.json"

        self.tracking_file = tracking_file
        self.state = load_json_file(tracking_file)
        logger.info(
            f"StateTracker initialized with {len(self.state)} tracked files from {tracking_file}"
        )

    def is_processed(self, filename: str, md5_hash: str) -> bool:
        if filename not in self.state:
            logger.debug(f"File not in tracking: {filename}")
            return False

        stored_hash = self.state[filename].get("md5_hash")
        is_same = stored_hash == md5_hash

        logger.info(
            f"Dedup check for {filename}: {is_same} (stored={stored_hash[:8]}..., new={md5_hash[:8]}...)"
        )
        return is_same

    def mark_processed(
        self,
        filename: str,
        md5_hash: str,
        version_date: datetime,
        chunk_count: int,
        document_id: str,
    ) -> None:

        self.state[filename] = {
            "md5_hash": md5_hash,
            "version_date": (
                version_date.isoformat() if isinstance(version_date, datetime) else version_date
            ),
            "chunk_count": chunk_count,
            "document_id": document_id,
            "processed_at": datetime.utcnow().isoformat(),
        }

        self._save()
        logger.info(
            f"Marked as processed: {filename} ({chunk_count} chunks, doc_id={document_id[:8]}...)"
        )

    def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        return self.state.get(filename)

    def remove_file(self, filename: str) -> None:

        if filename in self.state:
            del self.state[filename]
            self._save()
            logger.info(f"Removed from tracking: {filename}")
        else:
            logger.warning(f"File not in tracking, cannot remove: {filename}")

    def list_processed_files(self) -> Dict[str, Any]:
        """
        Get list of all processed files.

        Returns:
            Dictionary of all tracked files with metadata
        """
        logger.info(f"Listing {len(self.state)} processed files")
        return self.state.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tracked files.

        Returns:
            Stats dictionary (total_files, total_chunks, etc.)
        """
        total_chunks = sum(f.get("chunk_count", 0) for f in self.state.values())
        stats = {
            "total_files": len(self.state),
            "total_chunks": total_chunks,
            "tracking_file": str(self.tracking_file),
            "last_updated": datetime.utcnow().isoformat(),
        }
        logger.info(f"State stats: {stats}")
        return stats

    def _save(self) -> None:
        """Save current state to JSON tracking file."""
        from utils.file_utils import save_json_file

        save_json_file(self.tracking_file, self.state)
        logger.debug(f"Saved state to {self.tracking_file}")
