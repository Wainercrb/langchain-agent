"""
Reference data seed script — idempotent, re-runnable.

Populates reference/lookup tables with foundational data using the Supabase
REST client. All inserts use upsert semantics (ON CONFLICT ... DO NOTHING)
to guarantee idempotency across multiple runs.

This script reads SUPABASE_URL and SUPABASE_KEY from .env (via settings)
and NEVER writes to app data tables:
  - documents
  - document_chunks
  - ingestion_logs
  - version_cache

Usage:
    python seeds/seed_reference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.container import logger
from supabase import create_client


def get_supabase_client():
    """Create and return a Supabase admin client."""
    from config.settings import settings  # noqa: PLC0415

    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    logger.info("Creating Supabase client...")
    return create_client(settings.supabase_url, settings.supabase_key)


def seed_document_statuses(client) -> int:
    """Seed document status reference data.

    This is an example reference seed. Replace or extend with actual
    reference/lookup tables your application needs.
    """
    # Example: seed a document_statuses table if it exists
    # Using raw upsert via Supabase REST
    records = [
        {"status": "pending", "description": "Document queued for processing"},
        {"status": "processing", "description": "Document is being processed"},
        {"status": "completed", "description": "Document processing finished"},
        {"status": "failed", "description": "Document processing failed"},
        {"status": "archived", "description": "Document moved to archive"},
    ]

    count = 0
    for record in records:
        try:
            # Upsert: insert if not exists, skip if exists
            # This uses Supabase's .upsert() which maps to INSERT ... ON CONFLICT
            result = client.table("document_statuses").upsert(
                record,
                on_conflict="status",  # conflict target column
                ignore_duplicates=True,
            ).execute()

            if result.data:
                count += 1
                logger.debug(f"  Seeded status: {record['status']}")
        except Exception as e:
            # Table may not exist yet — that's OK for optional reference data
            logger.debug(f"  Skipped status '{record['status']}': {e}")

    return count


def main() -> int:
    """Main seed entry point."""
    logger.info("=" * 60)
    logger.info("🌱 Reference Data Seed Script")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()

        # ── Seed reference data ──────────────────────────────────────
        logger.info("\nSeeding document statuses...")
        seeded = seed_document_statuses(client)
        logger.info(f"  {seeded} status records processed")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Seed completed successfully!")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"\n❌ Seed failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
