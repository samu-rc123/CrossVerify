import uuid
import socket
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Configuration settings for CrossVerify Background Worker Pool."""

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Redis Stream & Consumer Group Configuration
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_STREAM_KEY: str = "shadow_traffic"
    REDIS_CONSUMER_GROUP: str = "crossverify_workers"
    REDIS_CONSUMER_NAME: str = f"worker_{socket.gethostname()}_{uuid.uuid4().hex[:6]}"
    REDIS_BATCH_SIZE: int = 10
    REDIS_BLOCK_MS: int = 2000
    REDIS_SOCKET_TIMEOUT: float = 5.0
    REDIS_CONNECT_TIMEOUT: float = 5.0

    # PostgreSQL Telemetry Store
    DATABASE_URL: str = "postgresql://postgres:postgres@postgres:5432/crossverify"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # Canary Circuit Breaker Key
    CANARY_ABORT_KEY: str = "canary:abort_traffic"

    # Sandboxed Candidate Model Settings (Day 5)
    CANDIDATE_MODEL_NAME: str = "candidate-gpt-4o-mini"
    CANDIDATE_API_KEY: Optional[str] = None
    CANDIDATE_BASE_URL: Optional[str] = None
    CANDIDATE_TIMEOUT_SECONDS: float = 15.0
    CANDIDATE_SIMULATE_LATENCY_MS: float = 85.0
    CANDIDATE_SIMULATE_DRIFT_RATE: float = 0.0

    # Semantic Discrepancy Engine Settings (Days 7-8)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.80

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
def get_worker_settings() -> WorkerSettings:
    """Return cached instance of worker settings."""
    return WorkerSettings()
