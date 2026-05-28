"""Run pending Alembic migrations."""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from services.container import logger


def main() -> int:
    ini = Path(__file__).parent / "alembic.ini"
    if not ini.exists():
        logger.error(f"alembic.ini not found at {ini}")
        return 1

    env = os.environ.copy()
    env["SUPABASE_DIRECT_URL"] = settings.supabase_direct_url

    logger.info("Running migrations...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "--config", str(ini), "upgrade", "head"],
        env=env, timeout=120,
    )
    if result.returncode != 0:
        logger.error("Migration failed")
        return 1

    logger.info("Database up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
