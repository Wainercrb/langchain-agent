"""
Alembic environment configuration — reads SUPABASE_DIRECT_URL at runtime.

This module loads the migration configuration, reads the direct database URL
from application settings (which reads from .env / environment), and runs
migrations in both offline and online modes.
"""

import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context

# Ensure the agent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Alembic Config object
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Read SUPABASE_DIRECT_URL from application settings ────────────────
# This is the ONLY way the URL is resolved — no hardcoded values.
# Also supports DIRECT_URL env var directly as fallback.
from config.settings import settings  # noqa: E402

direct_url = settings.supabase_direct_url
if not direct_url:
    # Fallback: try bare env var in case settings failed to load it
    import os
    direct_url = os.environ.get("SUPABASE_DIRECT_URL")

if not direct_url:
    raise ValueError(
        "SUPABASE_DIRECT_URL is required for Alembic migrations. "
        "Set it in .env or as an environment variable."
    )

config.set_main_option("sqlalchemy.url", direct_url)

# ── Metadata ──────────────────────────────────────────────────────────
# No ORM metadata — all migrations are written manually.
# This avoids coupling DDL to application model classes.
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without connecting)."""
    context.configure(
        url=direct_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    from alembic import engine_from_config

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
