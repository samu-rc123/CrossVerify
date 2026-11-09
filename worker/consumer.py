import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import redis.asyncio as aioredis
from redis.exceptions import RedisError, ResponseError
from pydantic import BaseModel, Field

from worker.core.config import get_worker_settings

logger = logging.getLogger("crossverify.worker.consumer")


class ShadowEvent(BaseModel):
    """Parsed event representing shadowed live production traffic."""
    message_id: str = Field(..., description="Redis Stream message identifier")
    request_id: str = Field(..., description="Unique completion request ID")
    timestamp: str = Field(..., description="UTC ISO timestamp of event creation")
    model: str = Field(..., description="Target model requested by client")
    prompt: str = Field(..., description="Extracted user prompt text")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Full conversation context")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Max generation tokens")
    champion_response: str = Field(..., description="Champion baseline response")
    champion_latency_ms: float = Field(..., description="Champion inference latency in ms")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom tracking metadata")

    @classmethod
    def from_stream_entry(cls, message_id: str, fields: Dict[str, str]) -> "ShadowEvent":
        """Deserialize raw Redis Stream string fields into structured ShadowEvent."""
        try:
            messages = json.loads(fields.get("messages", "[]"))
        except (json.JSONDecodeError, TypeError):
            messages = []

        try:
            metadata = json.loads(fields.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        max_tokens_val = fields.get("max_tokens")
        max_tokens = int(max_tokens_val) if max_tokens_val and max_tokens_val.isdigit() else None

        return cls(
            message_id=message_id,
            request_id=fields.get("request_id", f"unknown-{message_id}"),
            timestamp=fields.get("timestamp", datetime.now(timezone.utc).isoformat()),
            model=fields.get("model", "champion-gpt-4o"),
            prompt=fields.get("prompt", ""),
            messages=messages,
            temperature=float(fields.get("temperature", 0.7)),
            max_tokens=max_tokens,
            champion_response=fields.get("champion_response", ""),
            champion_latency_ms=float(fields.get("champion_latency_ms", 0.0)),
            metadata=metadata
        )


class ShadowConsumer:
    """
    Distributed consumer reading shadowed live requests from Redis Streams
    using persistent Consumer Groups for scalable, parallel evaluation.
    """

    def __init__(self):
        self.settings = get_worker_settings()
        self.client: Optional[aioredis.Redis] = None
        self.pool: Optional[aioredis.ConnectionPool] = None
        self.is_connected: bool = False
        self.is_running: bool = False
        self.stats = {
            "events_received": 0,
            "events_acknowledged": 0,
            "events_failed": 0
        }

    async def connect(self) -> bool:
        """Initialize connection pool to Redis."""
        if self.client and self.is_connected:
            return True

        try:
            self.pool = aioredis.ConnectionPool(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD,
                db=self.settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=self.settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=self.settings.REDIS_CONNECT_TIMEOUT
            )
            self.client = aioredis.Redis(connection_pool=self.pool)
            await self.client.ping()
            self.is_connected = True
            logger.info(
                f"[ShadowConsumer] Connected to Redis at {self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}"
            )
            return True
        except Exception as e:
            self.is_connected = False
            logger.error(f"[ShadowConsumer] Failed to connect to Redis: {e}")
            return False

    async def disconnect(self):
        """Cleanly close Redis connections."""
        self.is_running = False
        if self.client:
            await self.client.aclose()
        if self.pool:
            await self.pool.disconnect()
        self.is_connected = False
        logger.info("[ShadowConsumer] Disconnected from Redis broker.")

    async def ensure_consumer_group(self) -> bool:
        """
        Verify or initialize the consumer group on the shadow stream.
        Uses '$' so workers only consume events arrived after group creation by default,
        or '0' if historical stream replay is desired.
        """
        if not self.client or not self.is_connected:
            return False

        try:
            await self.client.xgroup_create(
                name=self.settings.REDIS_STREAM_KEY,
                groupname=self.settings.REDIS_CONSUMER_GROUP,
                id="0",
                mkstream=True
            )
            logger.info(
                f"[ShadowConsumer] Created consumer group '{self.settings.REDIS_CONSUMER_GROUP}' "
                f"on stream '{self.settings.REDIS_STREAM_KEY}'."
            )
            return True
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(
                    f"[ShadowConsumer] Consumer group '{self.settings.REDIS_CONSUMER_GROUP}' already exists."
                )
                return True
            logger.error(f"[ShadowConsumer] Error creating consumer group: {e}")
            return False
        except Exception as e:
            logger.error(f"[ShadowConsumer] Unexpected error ensuring consumer group: {e}")
            return False

    async def read_events(
        self,
        count: Optional[int] = None,
        block: Optional[int] = None
    ) -> List[ShadowEvent]:
        """
        Read newly assigned messages from the stream using XREADGROUP.
        """
        if not self.client or not self.is_connected:
            return []

        batch_count = count or self.settings.REDIS_BATCH_SIZE
        block_ms = block if block is not None else self.settings.REDIS_BLOCK_MS

        try:
            raw_response = await self.client.xreadgroup(
                groupname=self.settings.REDIS_CONSUMER_GROUP,
                consumername=self.settings.REDIS_CONSUMER_NAME,
                streams={self.settings.REDIS_STREAM_KEY: ">"},
                count=batch_count,
                block=block_ms
            )

            if not raw_response:
                return []

            events = []
            for stream_name, messages in raw_response:
                for message_id, fields in messages:
                    try:
                        event = ShadowEvent.from_stream_entry(message_id, fields)
                        events.append(event)
                        self.stats["events_received"] += 1
                    except Exception as parse_err:
                        self.stats["events_failed"] += 1
                        logger.error(
                            f"[ShadowConsumer] Failed to parse message {message_id}: {parse_err}"
                        )
            return events
        except RedisError as e:
            logger.error(f"[ShadowConsumer] Redis error reading from stream: {e}")
            return []
        except Exception as e:
            logger.error(f"[ShadowConsumer] Unexpected error reading from stream: {e}")
            return []

    async def acknowledge_event(self, message_id: str) -> bool:
        """Acknowledge processed message to remove it from the consumer group's PEL."""
        if not self.client or not self.is_connected:
            return False

        try:
            await self.client.xack(
                self.settings.REDIS_STREAM_KEY,
                self.settings.REDIS_CONSUMER_GROUP,
                message_id
            )
            self.stats["events_acknowledged"] += 1
            logger.debug(f"[ShadowConsumer] Acknowledged message {message_id}")
            return True
        except Exception as e:
            logger.error(f"[ShadowConsumer] Failed to ACK message {message_id}: {e}")
            return False

    async def process_event(self, event: ShadowEvent) -> bool:
        """
        Core event handler for shadowed traffic.
        (Day 4 logs and ACKs; Day 5 triggers candidate inference; Day 6-8 evaluates and saves).
        """
        logger.info(
            f"[ShadowConsumer] Ingested shadow event: req_id={event.request_id}, "
            f"model={event.model}, prompt='{event.prompt[:40]}...', "
            f"champion_latency={event.champion_latency_ms:.2f}ms"
        )
        # Successfully acknowledge processed message
        return await self.acknowledge_event(event.message_id)

    async def run(self, max_iterations: Optional[int] = None):
        """
        Continuous worker loop consuming messages from Redis Streams.
        """
        self.is_running = True
        logger.info(
            f"[ShadowConsumer] Worker '{self.settings.REDIS_CONSUMER_NAME}' starting consumption loop..."
        )

        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                logger.error("[ShadowConsumer] Unable to connect to Redis. Retrying in background...")

        await self.ensure_consumer_group()

        iterations = 0
        while self.is_running:
            if not self.is_connected:
                await asyncio.sleep(2.0)
                await self.connect()
                await self.ensure_consumer_group()
                continue

            events = await self.read_events()
            for event in events:
                try:
                    await self.process_event(event)
                except Exception as proc_err:
                    self.stats["events_failed"] += 1
                    logger.error(f"[ShadowConsumer] Error processing event {event.request_id}: {proc_err}")

            iterations += 1
            if max_iterations and iterations >= max_iterations:
                break

            # Brief non-blocking yield if no events arrived
            if not events:
                await asyncio.sleep(0.05)

        logger.info(f"[ShadowConsumer] Worker '{self.settings.REDIS_CONSUMER_NAME}' stopped.")

    def stop(self):
        """Signal consumer loop to cease execution."""
        self.is_running = False
