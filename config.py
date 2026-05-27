"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Google API Configuration
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")

    # Supabase Configuration
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_key: str = Field(..., alias="SUPABASE_KEY")
    supabase_direct_url: Optional[str] = Field(default=None, alias="SUPABASE_DIRECT_URL")

    # Cron Configuration
    cron_interval_minutes: int = Field(default=5, alias="CRON_INTERVAL_MINUTES")

    # Path Configuration
    knowledge_dir: Path = Field(default=Path("./knowledge"), alias="KNOWLEDGE_DIR")
    processed_dir: Optional[Path] = Field(default=None, alias="PROCESSED_DIR")
    failed_dir: Optional[Path] = Field(default=None, alias="FAILED_DIR")

    # Embedding Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    embeddings_batch_size: int = Field(default=10, alias="EMBEDDINGS_BATCH_SIZE")
    embedding_timeout: int = Field(default=30, alias="EMBEDDING_TIMEOUT")
    embedding_retries: int = Field(default=3, alias="EMBEDDING_RETRIES")

    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Alert Configuration (Optional)
    discord_webhook_url: Optional[str] = Field(default=None, alias="DISCORD_WEBHOOK_URL")
    alert_email: Optional[str] = Field(default=None, alias="ALERT_EMAIL")

    # LLM Configuration
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "populate_by_name": True,
        "extra": "ignore",  # Ignore extra fields for backward compatibility
    }

    @field_validator("processed_dir", mode="before")
    @classmethod
    def set_processed_dir(cls, v, info):
        """Set processed_dir if not provided."""
        if v is None:
            knowledge_dir = info.data.get("knowledge_dir", Path("./knowledge"))
            return Path(knowledge_dir) / "processed"
        return Path(v)

    @field_validator("failed_dir", mode="before")
    @classmethod
    def set_failed_dir(cls, v, info):
        """Set failed_dir if not provided."""
        if v is None:
            knowledge_dir = info.data.get("knowledge_dir", Path("./knowledge"))
            return Path(knowledge_dir) / "failed"
        return Path(v)

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v):
        """Validate chunk size."""
        if v < 100:
            raise ValueError("chunk_size must be at least 100")
        if v > 10000:
            raise ValueError("chunk_size must not exceed 10000")
        return v

    @field_validator("embedding_retries")
    @classmethod
    def validate_retries(cls, v):
        """Validate retry count."""
        if v < 1:
            raise ValueError("embedding_retries must be at least 1")
        if v > 10:
            raise ValueError("embedding_retries must not exceed 10")
        return v

    def create_directories(self) -> None:
        """Create required directories."""
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """String representation (hide sensitive data)."""
        return (
            f"Settings(cron_interval={self.cron_interval_minutes}min, "
            f"chunk_size={self.chunk_size}, "
            f"knowledge_dir={self.knowledge_dir})"
        )


# Load settings from environment
settings = Settings()
