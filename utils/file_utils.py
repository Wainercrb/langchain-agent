"""File utilities for hashing and file operations."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def calculate_md5(file_path: Path) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash as hex string

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not file_path.exists():
        logger.warning(f"JSON file not found: {file_path}, returning empty dict")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {file_path}: {str(e)}")
        raise


def save_json_file(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Save data to JSON file.

    Args:
        file_path: Path to JSON file
        data: Data to save

    Raises:
        IOError: If file cannot be written
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON file: {file_path}")
    except IOError as e:
        logger.error(f"Failed to save JSON file: {file_path}: {str(e)}")
        raise


def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.stat().st_size


def get_supported_extensions() -> list[str]:
    """
    Get list of supported file extensions.

    Returns:
        List of supported extensions (lowercase with dot)
    """
    return [".txt", ".md", ".html", ".pdf", ".docx", ".csv"]


def is_supported_file(file_path: Path) -> bool:
    """
    Check if file extension is supported.

    Args:
        file_path: Path to file

    Returns:
        True if supported, False otherwise
    """
    return file_path.suffix.lower() in get_supported_extensions()
