"""Alembic environment — reads SUPABASE_DIRECT_URL from settings."""

import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from config import settings
from loggers import logger

db_direct_url = settings.supabase_direct_url

if not db_direct_url:
    logger.error("DB DIRECT URL is not set")
    raise ValueError("DB DIRECT URL is required")

logger.info(f"Using database: {db_direct_url[:30]}...")
config.set_main_option("sqlalchemy.url", db_direct_url)
target_metadata = None


def run_migrations_offline() -> None:
    logger.info("Running migrations offline...")
    context.configure(
        url=db_direct_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

    logger.info("Offline migrations complete")


def run_migrations_online() -> None:
    logger.info("Connecting to database...")
    connectable = create_engine(db_direct_url)

    with connectable.connect() as connection:
        logger.info("Running migrations online...")
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    logger.info("Online migrations complete")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
