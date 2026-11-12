from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for CrossVerify API Gateway."""
    
    # Environment & Service
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 8000
    
    # Champion Model Settings
    CHAMPION_MODEL_NAME: str = "champion-gpt-4o"
    CHAMPION_API_KEY: Optional[str] = None
    CHAMPION_BASE_URL: Optional[str] = None
    CHAMPION_SIMULATE_LATENCY_MS: float = 60.0
    
    # Message Broker (Redis Streams)
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_STREAM_KEY: str = "shadow_traffic"
    REDIS_STREAM_MAXLEN: int = 100000
    REDIS_SOCKET_TIMEOUT: float = 2.0
    REDIS_CONNECT_TIMEOUT: float = 2.0
    REDIS_RETRY_ON_TIMEOUT: bool = True
    REDIS_CONSUMER_GROUP: str = "crossverify_workers"
    REDIS_CONSUMER_NAME: str = "worker_1"
    
    # Telemetry Store (PostgreSQL)
    DATABASE_URL: str = "postgresql://postgres:postgres@postgres:5432/crossverify"
    
    # Canary Circuit Breaker
    CANARY_ABORT_KEY: str = "canary:abort_traffic"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()
