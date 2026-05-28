"""Configuration management using Pydantic Settings — centraliza todas las variables de entorno."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Google / Gemini ──────────────────────────────────────────────
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")

    # ── Supabase / pgvector ──────────────────────────────────────────
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_key: str = Field(..., alias="SUPABASE_KEY")
    supabase_direct_url: str = Field(..., alias="SUPABASE_DIRECT_URL")

    # ── Scheduling ───────────────────────────────────────────────────
    cron_interval_minutes: int = Field(default=5, alias="CRON_INTERVAL_MINUTES")

    # ── Paths ────────────────────────────────────────────────────────
    knowledge_dir: Path = Field(default=Path("./knowledge/raw_docs"), alias="KNOWLEDGE_DIR")
    processed_dir: Path = Field(default=Path("./knowledge/processed"), alias="PROCESSED_DIR")
    failed_dir: Path = Field(default=Path("./knowledge/failed"), alias="FAILED_DIR")

    # ── Embeddings ───────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    embedding_retries: int = Field(default=3, alias="EMBEDDING_RETRIES")

    # ── Logging (Strategy Pattern) ───────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    logger_backend: str = Field(default="console", alias="LOGGER_BACKEND")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")

    # ── Alerts ────────────────────────────────────────────────────────
    discord_webhook_url: Optional[str] = Field(default=None, alias="DISCORD_WEBHOOK_URL")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "populate_by_name": True,
        "extra": "ignore",
    }

settings = Settings()
