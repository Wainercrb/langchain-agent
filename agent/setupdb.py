"""Run pending Alembic migrations or reset the database.

Usage:
    python setupdb.py              # Run migrations only
    python setupdb.py --reset      # DROP everything and recreate from scratch
"""

import os
import subprocess
import sys
from pathlib import Path

import psycopg2

from config.settings import settings

RESET_MODE = "--reset" in sys.argv


def get_row_counts(cur) -> dict:
    """Get row counts for all tables."""
    counts = {}
    for table in ["documents", "document_chunks", "ingestion_logs"]:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        except psycopg2.Error:
            counts[table] = "N/A"
    return counts


def reset_database() -> None:
    """Drop and recreate the public schema."""
    print("WARNING: This will DELETE all data in the database!")
    confirm = input("Type 'yes' to continue: ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        sys.exit(0)

    try:
        conn = psycopg2.connect(
            settings.supabase_direct_url,
            sslmode="require",
            connect_timeout=10,
        )
    except psycopg2.OperationalError as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    conn.autocommit = True
    cur = conn.cursor()

    before = get_row_counts(cur)
    print(f"Before reset: {before}")

    cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
    cur.execute("CREATE SCHEMA public")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    cur.execute("GRANT ALL ON SCHEMA public TO postgres")
    cur.execute("GRANT ALL ON SCHEMA public TO public")

    after = get_row_counts(cur)
    print(f"After reset: {after}")

    cur.close()
    conn.close()
    print("Database reset complete.")


def run_migrations() -> int:
    """Run alembic migrations."""
    ini = Path(__file__).parent / "alembic.ini"
    if not ini.exists():
        print(f"ERROR: alembic.ini not found at {ini}")
        return 1

    env = os.environ.copy()
    env["SUPABASE_DIRECT_URL"] = settings.supabase_direct_url

    print("Running migrations...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "--config", str(ini), "upgrade", "head"],
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        print("ERROR: Migration failed")
        return 1

    print("Done")
    return 0


def main() -> int:
    if RESET_MODE:
        reset_database()

    return run_migrations()


if __name__ == "__main__":
    sys.exit(main())
