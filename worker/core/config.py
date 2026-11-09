import os
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

    # Canary Circuit Breaker Key
    CANARY_ABORT_KEY: str = "canary:abort_traffic"

    # Candidate Model Name (for Day 5 inference execution)
    CANDIDATE_MODEL_NAME: str = "candidate-gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_worker_settings() -> WorkerSettings:
    """Return cached instance of worker settings."""
    return WorkerSettings()
