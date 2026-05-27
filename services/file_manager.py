"""File management for scanning, deduplication, and movement."""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FileManager:
    """Manage file scanning, deduplication, and movement."""

    def __init__(self, state_tracker):
        from config import settings
        from utils.file_utils import is_supported_file

        self.settings = settings
        self.state_tracker = state_tracker
        self.is_supported_file = is_supported_file
        logger.info("FileManager initialized")

    def scan_raw_docs(self) -> List[Path]:
        from utils.exceptions import FileManagerError

        try:
            raw_docs_dir = self.settings.knowledge_dir / "raw_docs"

            if not raw_docs_dir.exists():
                logger.warning(f"raw_docs directory does not exist: {raw_docs_dir}")
                return []

            files = [f for f in raw_docs_dir.iterdir() if f.is_file() and self.is_supported_file(f)]

            logger.info(f"Scanned raw_docs: found {len(files)} supported files")
            return files
        except Exception as e:
            logger.error(f"Failed to scan raw_docs: {str(e)}")
            raise FileManagerError(
                message=f"Failed to scan raw_docs directory: {str(e)}",
                error_code="SCAN_ERROR",
            )

    def detect_new_files(self, files: List[Path]) -> Tuple[List[Path], Dict[str, str]]:
        from utils.exceptions import FileManagerError
        from utils.file_utils import calculate_md5

        try:
            new_files = []
            file_hashes = {}

            for file_path in files:
                try:
                    md5_hash = calculate_md5(file_path)
                    file_hashes[file_path.name] = md5_hash

                    # Check if file is new or modified
                    if not self.state_tracker.is_processed(file_path.name, md5_hash):
                        new_files.append(file_path)
                        logger.info(f"Detected new/modified file: {file_path.name}")
                    else:
                        logger.debug(f"File already processed (unchanged): {file_path.name}")
                    continue

            logger.info(f"Detected {len(new_files)} new/modified files out of {len(files)}")
            return new_files, file_hashes
        except Exception as e:
            logger.error(f"Failed to detect new files: {str(e)}")
            raise FileManagerError(
                message=f"Failed to detect new files: {str(e)}",
                error_code="DETECTION_ERROR",
            )

    def move_file(
        self,
        file_path: Path,
        destination_dir: Path,
        add_timestamp: bool = True,
    ) -> Path:
        """
        Move file to destination directory.

        Args:
            file_path: Source file path
            destination_dir: Destination directory
            add_timestamp: Whether to append timestamp to filename

        Returns:
            Path to moved file

        Raises:
            Exception: If move fails
        """
        from utils.exceptions import FileManagerError

        try:
            destination_dir.mkdir(parents=True, exist_ok=True)

            if add_timestamp:
                # Append timestamp to avoid conflicts
                stem = file_path.stem
                suffix = file_path.suffix
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                new_filename = f"{stem}__{timestamp}{suffix}"
                dest_path = destination_dir / new_filename
            else:
                dest_path = destination_dir / file_path.name

            # Move file
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"Moved file: {file_path.name} -> {dest_path.name}")
            return dest_path
        except Exception as e:
            logger.error(f"Failed to move file: {str(e)}")
            raise FileManagerError(
                message=f"Failed to move file {file_path.name}: {str(e)}",
                error_code="MOVE_ERROR",
                details={"source": str(file_path), "destination": str(destination_dir)},
            )

    def move_to_processed(
        self,
        file_path: Path,
        add_timestamp: bool = True,
    ) -> Path:
        """
        Move successfully processed file to processed folder.

        Args:
            file_path: File to move
            add_timestamp: Whether to append timestamp

        Returns:
            Path to moved file
        """
        processed_dir = self.settings.processed_dir
        return self.move_file(file_path, processed_dir, add_timestamp)

    def move_to_failed(
        self,
        file_path: Path,
        error_reason: Optional[str] = None,
        add_timestamp: bool = True,
    ) -> Path:
        """
        Move failed file to failed folder.

        Args:
            file_path: File that failed processing
            error_reason: Brief error description (will be added to filename)
            add_timestamp: Whether to append timestamp

        Returns:
            Path to moved file
        """
        from utils.exceptions import FileManagerError

        try:
            failed_dir = self.settings.failed_dir
            failed_dir.mkdir(parents=True, exist_ok=True)

            # Add error reason to filename if provided
            if error_reason:
                stem = file_path.stem
                suffix = file_path.suffix
                error_code = error_reason.split("_")[0][:10]  # First 10 chars of error
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                new_filename = f"{stem}__{error_code}__{timestamp}{suffix}"
                dest_path = failed_dir / new_filename
            else:
                dest_path = failed_dir / file_path.name

            # Move file
            shutil.move(str(file_path), str(dest_path))
            logger.warning(
                f"Moved to failed: {file_path.name} -> {dest_path.name} (reason: {error_reason})"
            )
            return dest_path
        except Exception as e:
            logger.error(f"Failed to move to failed folder: {str(e)}")
            raise FileManagerError(
                message=f"Failed to move failed file {file_path.name}: {str(e)}",
                error_code="FAILED_MOVE_ERROR",
            )

    def cleanup_raw_docs(self, file_path: Path) -> None:
        """
        Delete a file from raw_docs (after successful move).

        Args:
            file_path: File to delete

        Raises:
            Exception: If deletion fails
        """
        from utils.exceptions import FileManagerError

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted from raw_docs: {file_path.name}")
            else:
                logger.warning(f"File not found for deletion: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            raise FileManagerError(
                message=f"Failed to delete file {file_path.name}: {str(e)}",
                error_code="DELETE_ERROR",
            )

    def get_file_stats(self) -> Dict[str, Any]:
        """
        Get statistics about files in knowledge directories.

        Returns:
            Stats dictionary with file counts and sizes
        """

        def count_files_and_size(directory: Path) -> Tuple[int, int]:
            if not directory.exists():
                return 0, 0
            total_size = 0
            count = 0
            for f in directory.rglob("*"):
                if f.is_file():
                    count += 1
                    total_size += f.stat().st_size
            return count, total_size

        raw_count, raw_size = count_files_and_size(self.settings.knowledge_dir / "raw_docs")
        proc_count, proc_size = count_files_and_size(self.settings.processed_dir)
        fail_count, fail_size = count_files_and_size(self.settings.failed_dir)

        stats = {
            "raw_docs": {"count": raw_count, "size_bytes": raw_size},
            "processed": {"count": proc_count, "size_bytes": proc_size},
            "failed": {"count": fail_count, "size_bytes": fail_size},
            "total_count": raw_count + proc_count + fail_count,
            "total_size_mb": (raw_size + proc_size + fail_size) / (1024 * 1024),
        }

        logger.info(f"File stats: {stats}")
        return stats
