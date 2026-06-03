"""Backup script for Supabase/pgvector database.

Creates a logical backup using pg_dump (if available) or falls back to
exporting all documents and chunks via the Supabase REST API.

Usage:
    python scripts/backup.py [--retention 7] [--output-dir ./backups]

Environment variables (from .env):
    SUPABASE_URL
    SUPABASE_KEY
    SUPABASE_DIRECT_URL
    DISCORD_WEBHOOK_URL (optional, for alerting)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from infrastructure.logging import logger
from utils.exceptions import Severity


def _send_alert(message: str, error: Exception | None = None) -> None:
    """Send a Discord alert if webhook is configured."""
    if not settings.discord_webhook_url:
        logger.warning(f"Backup alert (no webhook): {message}")
        return

    import asyncio
    from infrastructure.container import alert_service

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(alert_service.send_alert(
            severity=Severity.CRITICAL,
            message=message,
            error=error,
        ))
    except RuntimeError:
        asyncio.run(alert_service.send_alert(
            severity=Severity.CRITICAL,
            message=message,
            error=error,
        ))


def _rotate_backups(output_dir: Path, retention: int) -> None:
    """Delete backups older than the retention count."""
    backups = sorted(output_dir.glob("backup_*.sql.gz"))
    if len(backups) <= retention:
        return

    for old in backups[: len(backups) - retention]:
        old.unlink()
        logger.info(f"Rotated old backup: {old.name}")


def backup_via_pg_dump(output_dir: Path) -> Path | None:
    """Create a backup using pg_dump against the Supabase direct URL.

    Returns the path to the backup file, or None if pg_dump is unavailable.
    """
    pg_dump = shutil.which("pg_dump")
    if not pg_dump:
        logger.info("pg_dump not found in PATH, skipping pg_dump backup")
        return None

    if not settings.supabase_direct_url:
        logger.warning("SUPABASE_DIRECT_URL not set, cannot use pg_dump")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = output_dir / f"backup_{timestamp}.sql.gz"

    # Set PGPASSWORD for auth
    env = os.environ.copy()
    env["PGPASSWORD"] = settings.supabase_key

    cmd = [
        pg_dump,
        settings.supabase_direct_url,
        "--format=custom",
        "--compress=9",
        "--no-owner",
        "--no-privileges",
        f"--file={backup_file}",
    ]

    logger.info(f"Running pg_dump: {' '.join(cmd[:3])}...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error(f"pg_dump failed: {result.stderr}")
        return None

    logger.info(f"pg_dump backup created: {backup_file.name}")
    return backup_file


def backup_via_api(output_dir: Path) -> Path | None:
    """Fallback: export documents and chunks via Supabase REST API.

    Returns the path to the backup file, or None on failure.
    """
    if not settings.supabase_url or not settings.supabase_key:
        logger.warning("Supabase credentials not set, cannot use API backup")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = output_dir / f"backup_api_{timestamp}.json"

    import httpx

    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    backup_data = {
        "backup_timestamp": datetime.now(timezone.utc).isoformat(),
        "supabase_url": settings.supabase_url,
        "tables": {},
    }

    tables = ["documents", "chunks", "ingestion_logs"]

    try:
        with httpx.Client(timeout=60) as client:
            for table in tables:
                logger.info(f"Exporting table: {table}")
                response = client.get(
                    f"{settings.supabase_url}/rest/v1/{table}",
                    headers=headers,
                    params={"select": "*"},
                )
                if response.status_code == 200:
                    rows = response.json()
                    backup_data["tables"][table] = rows
                    logger.info(f"  Exported {len(rows)} rows from {table}")
                else:
                    logger.warning(
                        f"  Failed to export {table}: HTTP {response.status_code}"
                    )
                    backup_data["tables"][table] = {
                        "error": f"HTTP {response.status_code}",
                        "body": response.text[:500],
                    }

        backup_file.write_text(json.dumps(backup_data, indent=2, default=str))
        logger.info(f"API backup created: {backup_file.name}")
        return backup_file

    except Exception as e:
        logger.error(f"API backup failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup Supabase/pgvector database")
    parser.add_argument(
        "--retention",
        type=int,
        default=7,
        help="Number of backups to keep (default: 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./backups",
        help="Directory to store backups (default: ./backups)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    logger.info(f"Backup started (retention={args.retention}, output={output_dir})")

    try:
        # Try pg_dump first, fall back to API export
        backup_path = backup_via_pg_dump(output_dir)
        if backup_path is None:
            logger.info("Falling back to API-based backup")
            backup_path = backup_via_api(output_dir)

        if backup_path is None:
            _send_alert("Backup failed: no backup method succeeded")
            logger.error("Backup failed: all methods exhausted")
            sys.exit(1)

        # Rotate old backups
        _rotate_backups(output_dir, args.retention)

        elapsed = time.time() - start
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Backup complete: {backup_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)"
        )

    except Exception as e:
        logger.error(f"Backup crashed: {e}", exc_info=True)
        _send_alert(f"Backup crashed: {str(e)}", error=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
