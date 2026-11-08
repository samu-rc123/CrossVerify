import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from gateway.core.config import get_settings

logger = logging.getLogger("crossverify.broker")


class TrafficBroker:
    """
    High-throughput, resilient message broker interface using Redis Streams.
    Asynchronously mirrors live production traffic to 'shadow_traffic'
    without imposing latency or availability risks on the primary response path.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[aioredis.Redis] = None
        self.pool: Optional[aioredis.ConnectionPool] = None
        self.is_connected: bool = False
        self.stats = {
            "published": 0,
            "failed": 0,
            "dropped_circuit_breaker": 0
        }

    async def connect(self) -> bool:
        """
        Establish connection pool to Redis Streams broker.
        Fails gracefully without halting application startup if Redis is temporarily unreachable.
        """
        try:
            self.pool = aioredis.ConnectionPool(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD,
                db=self.settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=self.settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=self.settings.REDIS_CONNECT_TIMEOUT,
                retry_on_timeout=self.settings.REDIS_RETRY_ON_TIMEOUT
            )
            self.client = aioredis.Redis(connection_pool=self.pool)
            await self.client.ping()
            self.is_connected = True
            logger.info(
                f"[TrafficBroker] Connected to Redis Streams at "
                f"{self.settings.REDIS_HOST}:{self.settings.REDIS_PORT} "
                f"(Stream: '{self.settings.REDIS_STREAM_KEY}')"
            )
            return True
        except Exception as e:
            self.is_connected = False
            logger.warning(
                f"[TrafficBroker] Unable to connect to Redis ({e}). "
                f"Shadow traffic will be dropped until broker recovers."
            )
            return False

    async def disconnect(self):
        """Gracefully terminate client connection and connection pool."""
        try:
            if self.client:
                await self.client.aclose()
            if self.pool:
                await self.pool.disconnect()
            self.is_connected = False
            logger.info("[TrafficBroker] Disconnected from Redis Streams broker.")
        except Exception as e:
            logger.error(f"[TrafficBroker] Error during broker disconnect: {e}")

    async def is_circuit_breaker_tripped(self) -> bool:
        """
        Query Redis for the dynamic canary circuit breaker abort flag.
        If tripped ('1' or 'true'), shadowing is temporarily suspended to protect candidate systems.
        """
        if not self.client or not self.is_connected:
            return False

        try:
            abort_flag = await self.client.get(self.settings.CANARY_ABORT_KEY)
            return bool(abort_flag and abort_flag.lower() in ("1", "true", "yes", "tripped"))
        except Exception as e:
            logger.warning(f"[TrafficBroker] Failed to check circuit breaker flag: {e}")
            return False

    async def publish_shadow_traffic(
        self,
        request_id: str,
        request_payload: Dict[str, Any],
        champion_response: str,
        champion_latency_ms: float
    ) -> Optional[str]:
        """
        Mirror an incoming request and its champion baseline response to Redis Streams.
        
        Guarantees:
        1. Strict non-blocking behavior: Never raises exceptions to the calling HTTP thread.
        2. Automatic stream trimming: Caps stream length to prevent unbounded memory growth.
        3. Canary circuit breaker compliance: Drops shadowing if canary thresholds were breached.
        """
        # 1. Circuit breaker gate check
        if await self.is_circuit_breaker_tripped():
            self.stats["dropped_circuit_breaker"] += 1
            logger.warning(
                f"[TrafficBroker] Canary abort flag active. Dropping shadow event for {request_id}."
            )
            return None

        # 2. Broker connection verification / recovery
        if not self.client or not self.is_connected:
            connected = await self.connect()
            if not connected:
                self.stats["failed"] += 1
                logger.warning(f"[TrafficBroker] Redis not connected. Dropped shadow event {request_id}.")
                return None

        # 3. Extract primary prompt content for direct evaluation
        messages = request_payload.get("messages", [])
        prompt_text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break
        if not prompt_text and messages and isinstance(messages[-1], dict):
            prompt_text = messages[-1].get("content", "")

        # 4. Construct stream message dictionary (all values serialized to strings)
        stream_entry = {
            "request_id": str(request_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": str(request_payload.get("model", self.settings.CHAMPION_MODEL_NAME)),
            "prompt": str(prompt_text),
            "messages": json.dumps(messages),
            "temperature": str(request_payload.get("temperature", 0.7)),
            "max_tokens": str(request_payload.get("max_tokens") or ""),
            "champion_response": str(champion_response),
            "champion_latency_ms": f"{champion_latency_ms:.2f}",
            "metadata": json.dumps(request_payload.get("metadata") or {})
        }

        # 5. Append event to stream with approximate capping
        try:
            message_id = await self.client.xadd(
                name=self.settings.REDIS_STREAM_KEY,
                fields=stream_entry,
                maxlen=self.settings.REDIS_STREAM_MAXLEN,
                approximate=True
            )
            self.stats["published"] += 1
            logger.info(
                f"[TrafficBroker] Mirrored event {request_id} -> "
                f"Stream '{self.settings.REDIS_STREAM_KEY}' (ID: {message_id})"
            )
            return message_id
        except RedisError as e:
            self.stats["failed"] += 1
            logger.error(
                f"[TrafficBroker] Redis error while publishing shadow event {request_id}: {e}"
            )
            return None
        except Exception as e:
            self.stats["failed"] += 1
            logger.error(
                f"[TrafficBroker] Unexpected error publishing shadow event {request_id}: {e}"
            )
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Return runtime telemetry for monitoring and health probes."""
        return {
            "is_connected": self.is_connected,
            "stream_key": self.settings.REDIS_STREAM_KEY,
            "published_count": self.stats["published"],
            "failed_count": self.stats["failed"],
            "dropped_circuit_breaker_count": self.stats["dropped_circuit_breaker"]
        }


# Global singleton instance for application use
traffic_broker = TrafficBroker()
