"""
Database initialization script — runs Alembic migrations and optional seeds.

Replaces the previous SQL-parsing workflow with versioned Alembic migrations.
Alembic handles DDL only (CREATE TABLE, INDEX, TRIGGER, GRANT) via a direct
PostgreSQL connection (SUPABASE_DIRECT_URL). Seed scripts use the Supabase
REST client for application data — they never write through Alembic.

Architecture rule: Alembic = DDL only. Supabase REST = app data CRUD.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.container import logger


def _migrations_dir() -> Path:
    """Return the absolute path to the migrations directory."""
    return Path(__file__).parent.parent / "migrations"


def _alembic_args(subcommand: str) -> list[str]:
    """Build alembic CLI command with config pointing to our migrations dir."""
    ini_path = _migrations_dir() / "alembic.ini"
    return [
        sys.executable,
        "-m",
        "alembic",
        "--config",
        str(ini_path),
        subcommand,
    ]


def run_alembic_upgrade() -> None:
    """Run `alembic upgrade head` via subprocess, inheriting env vars."""
    logger.info("Running: alembic upgrade head")
    env = os.environ.copy()

    # Ensure SUPABASE_DIRECT_URL flows into the subprocess.
    # settings already validated this exists before we get here.
    from config.settings import settings  # noqa: PLC0415

    env["SUPABASE_DIRECT_URL"] = settings.supabase_direct_url

    cmd = _alembic_args("upgrade") + ["head"]
    result = subprocess.run(
        cmd,
        capture_output=False,
        env=env,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Alembic upgrade failed with exit code {result.returncode}. "
            "Check the output above for details."
        )

    logger.info("✅ Alembic upgrade completed successfully")


def run_seeds() -> None:
    """Execute seed scripts if SEED_REFERENCE is enabled.

    Seed scripts are optional and gated by the SEED_REFERENCE env var
    (set to 'true' or '1' to enable). They use the Supabase REST client
    (not the Alembic direct connection) for application data.
    """
    seed_flag = os.environ.get("SEED_REFERENCE", "").lower()
    if seed_flag not in ("true", "1"):
        logger.info("SEED_REFERENCE not set — skipping seeds")
        return

    seed_script = Path(__file__).parent.parent / "seeds" / "seed_reference.py"
    if not seed_script.exists():
        logger.warning(f"Seed script not found: {seed_script}")
        return

    logger.info("Running seed scripts...")
    result = subprocess.run(
        [sys.executable, str(seed_script)],
        capture_output=False,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Seed script failed with exit code {result.returncode}. "
            "Check the output above for details."
        )

    logger.info("✅ Seeds completed successfully")


def setup_db() -> bool:
    """Main database setup entry point.

    Returns:
        True on success, False on failure.
    """
    logger.info("=" * 60)
    logger.info("🗄️  Database Setup — Alembic Migration Pipeline")
    logger.info("=" * 60)

    try:
        # ── Step 1: Validate SUPABASE_DIRECT_URL ──────────────────────
        logger.info("\n[Step 1/3] Validating database connection string...")
        from config.settings import settings  # noqa: PLC0415

        direct_url = settings.supabase_direct_url
        if not direct_url:
            raise ValueError(
                "SUPABASE_DIRECT_URL is required but not set. "
                "Add it to your .env file (postgresql://user:pass@host:port/db)."
            )

        logger.info("  ✓ SUPABASE_DIRECT_URL is set")

        # ── Step 2: Run Alembic migrations ────────────────────────────
        logger.info("\n[Step 2/3] Running database migrations...")
        try:
            run_alembic_upgrade()
        except ImportError as e:
            logger.warning(f"  ⚠  Alembic unavailable: {e}")
            logger.warning("  Install it: pip install alembic")
            logger.warning("  Cannot proceed without Alembic — schema.sql has been removed.")
            return False

        # ── Step 3: Run seed scripts (optional) ───────────────────────
        logger.info("\n[Step 3/3] Running seed scripts...")
        try:
            run_seeds()
        except Exception as e:
            logger.warning(f"  ⚠  Seeds failed (non-fatal): {e}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Database setup completed successfully!")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"\n❌ Database setup failed: {str(e)}", exc_info=True)
        logger.info("\n" + "=" * 60)
        logger.info("TROUBLESHOOTING:")
        logger.info("  1. Verify SUPABASE_DIRECT_URL in .env (postgresql://...)")
        logger.info("  2. Ensure SUPABASE_URL and SUPABASE_KEY are also set")
        logger.info("  3. Run: alembic --config migrations/alembic.ini upgrade head")
        logger.info("  4. For stamping existing DBs: python scripts/stamp_db.py")
        logger.info("=" * 60)
        return False


if __name__ == "__main__":
    success = setup_db()
    sys.exit(0 if success else 1)
