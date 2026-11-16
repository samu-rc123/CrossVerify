from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class CanarySettings(BaseSettings):
    """Configuration settings for Automated Canary Controller & SLO Governance."""

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # PostgreSQL Telemetry Store
    DATABASE_URL: str = "postgresql://postgres:postgres@postgres:5432/crossverify"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # Redis Broker & Dynamic Circuit Breaker Flag
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    CANARY_ABORT_KEY: str = "canary:abort_traffic"
    CANARY_ABORT_REASON_KEY: str = "canary:abort_reason"
    CANARY_STATE_KEY: str = "canary:state"

    # Canary SLO & Statistical Guardrails (Days 10-11)
    CANARY_EVAL_WINDOW_MINUTES: int = 5
    CANARY_MIN_SAMPLE_SIZE: int = 10
    CANARY_DRIFT_THRESHOLD: float = 0.05       # 5% maximum acceptable semantic drift rate
    CANARY_LATENCY_SPIKE_MS: float = 1500.0     # 1500ms p95 latency ceiling
    CANARY_MAX_ERROR_RATE: float = 0.05        # 5% maximum acceptable candidate error/timeout rate
    CANARY_POLL_INTERVAL_SECONDS: float = 10.0 # Polling frequency for rolling window
    CANARY_COOLDOWN_CYCLES: int = 3            # Consecutive healthy cycles required to auto-recover

    @property
    def async_database_url(self) -> str:
        """Normalize Postgres connection URL to asyncpg driver scheme."""
        if self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            return self.DATABASE_URL
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        if self.DATABASE_URL.startswith("postgres://"):
            return self.DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
        return self.DATABASE_URL

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_canary_settings() -> CanarySettings:
    """Return cached instance of canary controller settings."""
    return CanarySettings()
