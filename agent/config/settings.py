"""Configuration management using Pydantic Settings — centralizes all environment variables."""

from pathlib import Path
from typing import Annotated, Optional

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings


def _parse_list(v: object) -> list[str]:
    """Parse comma-separated string to list."""
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    if isinstance(v, list):
        return [str(p).strip() for p in v if str(p).strip()]
    return v  # type: ignore[return-value]


ListFromEnv = Annotated[list[str], BeforeValidator(_parse_list)]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Google / Gemini ──────────────────────────────────────────────
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=1000, alias="GEMINI_MAX_TOKENS")

    # ── OpenAI ───────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=1000, alias="OPENAI_MAX_TOKENS")

    # ── OpenRouter ───────────────────────────────────────────────────
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openai/gpt-4o", alias="OPENROUTER_MODEL")
    openrouter_temperature: float = Field(default=0.7, alias="OPENROUTER_TEMPERATURE")
    openrouter_max_tokens: int = Field(default=800, alias="OPENROUTER_MAX_TOKENS")

    # ── Supabase / pgvector ──────────────────────────────────────────
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_key: str = Field(..., alias="SUPABASE_KEY")
    supabase_direct_url: str = Field(..., alias="SUPABASE_DIRECT_URL")

    # ── Scheduling ───────────────────────────────────────────────────
    cron_interval_minutes: int = Field(default=5, alias="CRON_INTERVAL_MINUTES")
    ingestion_max_retries: int = Field(default=3, alias="INGESTION_MAX_RETRIES")

    # ── Paths ────────────────────────────────────────────────────────
    knowledge_dir: Path = Field(
        default=Path("./knowledge/raw_docs"), alias="KNOWLEDGE_DIR"
    )
    processed_dir: Path = Field(
        default=Path("./knowledge/processed"), alias="PROCESSED_DIR"
    )
    failed_dir: Path = Field(default=Path("./knowledge/failed"), alias="FAILED_DIR")

    # ── Embeddings ───────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    embedding_retries: int = Field(default=3, alias="EMBEDDING_RETRIES")

    # ── Logging ──────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    logger_backend: str = Field(default="console", alias="LOGGER_BACKEND")

    # ── AWS CloudWatch Logs ──────────────────────────────────────────
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    cloudwatch_log_group: str = Field(default="langchain-agent", alias="CLOUDWATCH_LOG_GROUP")
    cloudwatch_stream_name: str = Field(default="default", alias="CLOUDWATCH_STREAM_NAME")

    # ── CORS ──────────────────────────────────────────────────────────
    cors_origins: ListFromEnv = Field(
        default=[
            "http://localhost:4321",
            "http://localhost:3000",
            "http://127.0.0.1:4321",
            "http://127.0.0.1:3000",
        ],
        alias="CORS_ORIGINS",
        description="Comma-separated string or list of allowed CORS origins",
    )

    # ── Alerts ────────────────────────────────────────────────────────
    discord_webhook_url: Optional[str] = Field(
        default=None, alias="DISCORD_WEBHOOK_URL"
    )
    slack_webhook_url: Optional[str] = Field(
        default=None, alias="SLACK_WEBHOOK_URL"
    )
    alert_rate_limit_per_minute: int = Field(
        default=5, alias="ALERT_RATE_LIMIT_PER_MINUTE"
    )

    # ── LLM ──────────────────────────────────────────────────────────
    llm_timeout_seconds: int = Field(default=60, alias="LLM_TIMEOUT_SECONDS")
    llm_circuit_failure_threshold: int = Field(default=3, alias="LLM_CIRCUIT_FAILURE_THRESHOLD")
    llm_circuit_recovery_timeout: float = Field(default=60.0, alias="LLM_CIRCUIT_RECOVERY_TIMEOUT")
    llm_backoff_base: float = Field(default=1.0, alias="LLM_BACKOFF_BASE")
    llm_backoff_max: float = Field(default=30.0, alias="LLM_BACKOFF_MAX")

    # ── Rate Limiting ─────────────────────────────────────────────────
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(
        default=100, alias="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )

    # ── Traffic Shedding ──────────────────────────────────────────────
    traffic_shedding_enabled: bool = Field(default=False, alias="TRAFFIC_SHEDDING_ENABLED")
    traffic_shedding_retry_after: int = Field(default=60, alias="TRAFFIC_SHEDDING_RETRY_AFTER")

    # ── Observability ──────────────────────────────────────────────────
    # Backend choice: "langsmith" or "none" (no-op fallback)
    observability_backend: str = Field(default="langsmith", alias="OBSERVABILITY_BACKEND")

    # ── LangSmith (used when observability_backend="langsmith") ────────
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: Optional[str] = Field(default=None, alias="LANGSMITH_PROJECT")
    enable_langsmith_tracing: bool = Field(
        default=False, alias="ENABLE_LANGSMITH_TRACING"
    )

    # ── Monitoring ────────────────────────────────────────────────────
    monitoring_enabled: bool = Field(default=True, alias="MONITORING_ENABLED")
    monitoring_interval_seconds: int = Field(default=300, alias="MONITORING_INTERVAL_SECONDS")
    monitoring_memory_threshold_mb: int = Field(default=512, alias="MONITORING_MEMORY_THRESHOLD_MB")
    monitoring_tracing_window_seconds: int = Field(default=300, alias="MONITORING_TRACING_WINDOW_SECONDS")

    # ── Maintenance Schedule (automated runbook items) ───────────────
    # These settings control the 2 automated runbook jobs scheduled by
    # agent/cronjob.py: weekly backup, weekly VACUUM ANALYZE.
    # Log rotation is managed by CloudWatch retention policies.
    # See docs/runbooks/weekly-maintenance.md for the full list.
    maintenance_backup_enabled: bool = Field(default=True, alias="MAINTENANCE_BACKUP_ENABLED")
    maintenance_vacuum_enabled: bool = Field(default=True, alias="MAINTENANCE_VACUUM_ENABLED")
    maintenance_backup_retention: int = Field(default=7, alias="MAINTENANCE_BACKUP_RETENTION")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "populate_by_name": True,
        "extra": "ignore",
    }


settings = Settings()
