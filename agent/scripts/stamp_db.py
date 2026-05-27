"""
Stamp an existing database at the latest Alembic revision.

Use this when you already have tables created by the old schema.sql workflow
and want to mark them as matching the current migration state WITHOUT
re-executing the DDL. After stamping, `alembic upgrade head` will report
"already at latest" and future migrations will apply normally.

Usage:
    python scripts/stamp_db.py
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.container import logger


def main() -> int:
    """Run `alembic stamp head` to mark current schema as migrated."""
    logger.info("=" * 60)
    logger.info("📌 Stamp Database — Alembic Stamp Head")
    logger.info("=" * 60)

    try:
        # Validate SUPABASE_DIRECT_URL
        from config.settings import settings  # noqa: PLC0415

        if not settings.supabase_direct_url:
            logger.error("SUPABASE_DIRECT_URL is not set in .env")
            return 1

        logger.info(f"Using database: {settings.supabase_direct_url[:50]}...")

        ini_path = Path(__file__).parent.parent / "migrations" / "alembic.ini"
        if not ini_path.exists():
            logger.error(f"alembic.ini not found at {ini_path}")
            return 1

        env = os.environ.copy()
        env["SUPABASE_DIRECT_URL"] = settings.supabase_direct_url

        cmd = [
            sys.executable,
            "-m",
            "alembic",
            "--config",
            str(ini_path),
            "stamp",
            "head",
        ]

        logger.info("Running: alembic stamp head")
        result = subprocess.run(cmd, capture_output=False, env=env, timeout=60)

        if result.returncode == 0:
            logger.info("\n✅ Database stamped at head revision")
        else:
            logger.error(f"\n❌ Stamp failed with exit code {result.returncode}")

        return result.returncode

    except Exception as e:
        logger.error(f"\n❌ Stamp failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
