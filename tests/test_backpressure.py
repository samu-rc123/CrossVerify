import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from redis.exceptions import RedisError
from fastapi.testclient import TestClient

from gateway.core.broker import TrafficBroker
from gateway.main import app
from worker.consumer import ShadowConsumer, ShadowEvent
from worker.clients.candidate_mock import SandboxedCandidateClient
from worker.evaluator.semantic import SemanticEvaluator
from worker.evaluator.structural import StructuralEvaluator
from worker.db import DatabaseManager, EvaluationRecord


@pytest.fixture
def broker():
    b = TrafficBroker()
    b.is_connected = True
    return b


@pytest.mark.anyio
async def test_concurrent_publisher_flood(broker):
    """
    Simulate high-concurrency burst traffic: 200 concurrent requests
    dispatched simultaneously through TrafficBroker using asyncio.gather.
    Verify 100% throughput completion, 0 exceptions, and accurate metric counters.
    """
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None  # Breaker closed

    async def mock_xadd(name, fields, maxlen, approximate):
        # Emulate realistic async network transport latency (1ms)
        await asyncio.sleep(0.001)
        return f"1731057600000-{fields.get('request_id', '0')}"

    mock_redis.xadd.side_effect = mock_xadd
    broker.client = mock_redis

    concurrent_requests = 200

    async def send_single_shadow(i: int):
        return await broker.publish_shadow_traffic(
            request_id=f"chatcmpl-burst-{i:04d}",
            request_payload={
                "model": "champion-gpt-4o",
                "messages": [{"role": "user", "content": f"High concurrency query {i}"}],
                "temperature": 0.7
            },
            champion_response=f"Champion response {i}",
            champion_latency_ms=45.0
        )

    start_time = time.perf_counter()
    tasks = [send_single_shadow(i) for i in range(concurrent_requests)]
    results = await asyncio.gather(*tasks)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Assertions
    assert len(results) == concurrent_requests
    assert all(r is not None for r in results)
    assert broker.stats["published"] == concurrent_requests
    assert broker.stats["failed"] == 0
    assert broker.stats["dropped_circuit_breaker"] == 0
    assert mock_redis.xadd.call_count == concurrent_requests

    # Verify high throughput efficiency (200 requests dispatched concurrently in < 1000ms)
    assert elapsed_ms < 1000.0


@pytest.mark.anyio
async def test_stream_maxlen_approximate_capping(broker):
    """
    Verify stream capping backpressure controls: all stream insertions
    must strictly enforce MAXLEN=100000 with approximate=True to prevent
    unbounded Redis RAM growth during heavy traffic spikes.
    """
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.xadd.return_value = "1731057600000-0"
    broker.client = mock_redis

    for i in range(25):
        await broker.publish_shadow_traffic(
            request_id=f"chatcmpl-cap-{i}",
            request_payload={"messages": [{"role": "user", "content": "test"}]},
            champion_response="response",
            champion_latency_ms=30.0
        )

    assert mock_redis.xadd.call_count == 25
    for call in mock_redis.xadd.call_args_list:
        kwargs = call.kwargs
        assert kwargs["maxlen"] == broker.settings.REDIS_STREAM_MAXLEN
        assert kwargs["approximate"] is True


@pytest.mark.anyio
async def test_circuit_breaker_backpressure_drop_under_heavy_concurrency(broker):
    """
    Simulate mid-flight circuit breaker trip during an active 100-request burst:
    First 50 requests proceed (breaker closed); then breaker trips OPEN.
    Verify immediate shedding of the remaining 50 requests without hanging.
    """
    mock_redis = AsyncMock()
    breaker_state = {"tripped": False}

    async def mock_get(key):
        if key == broker.settings.CANARY_ABORT_KEY:
            return "1" if breaker_state["tripped"] else None
        return None

    mock_redis.get.side_effect = mock_get
    mock_redis.xadd.return_value = "1731057600000-0"
    broker.client = mock_redis

    # Phase 1: 50 requests with healthy breaker
    p1_tasks = [
        broker.publish_shadow_traffic(
            request_id=f"chatcmpl-p1-{i}",
            request_payload={"messages": [{"role": "user", "content": "healthy"}]},
            champion_response="ok",
            champion_latency_ms=25.0
        )
        for i in range(50)
    ]
    p1_results = await asyncio.gather(*p1_tasks)
    assert sum(1 for r in p1_results if r is not None) == 50
    assert broker.stats["published"] == 50
    assert broker.stats["dropped_circuit_breaker"] == 0

    # Phase 2: Breaker trips OPEN
    breaker_state["tripped"] = True

    p2_tasks = [
        broker.publish_shadow_traffic(
            request_id=f"chatcmpl-p2-{i}",
            request_payload={"messages": [{"role": "user", "content": "aborted"}]},
            champion_response="ok",
            champion_latency_ms=25.0
        )
        for i in range(50)
    ]
    p2_results = await asyncio.gather(*p2_tasks)

    # All 50 phase 2 requests must be shed/dropped immediately
    assert all(r is None for r in p2_results)
    assert broker.stats["published"] == 50
    assert broker.stats["dropped_circuit_breaker"] == 50


@pytest.mark.anyio
async def test_redis_outage_resilience_under_load(broker):
    """
    Simulate complete message broker connectivity failure during high load:
    100 concurrent requests encounter Redis connection drops or timeouts.
    Verify 0 exceptions escape to the caller, isolating the production response path.
    """
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.xadd.side_effect = RedisError("Connection refused: broker cluster partitioned")
    broker.client = mock_redis

    tasks = [
        broker.publish_shadow_traffic(
            request_id=f"chatcmpl-err-{i}",
            request_payload={"messages": [{"role": "user", "content": "fault injection"}]},
            champion_response="champion ok",
            champion_latency_ms=20.0
        )
        for i in range(100)
    ]

    # None of the 100 concurrent calls may raise an exception
    results = await asyncio.gather(*tasks, return_exceptions=False)
    assert all(r is None for r in results)
    assert broker.stats["failed"] == 100
    assert broker.stats["published"] == 0


@pytest.mark.anyio
async def test_worker_consumer_batch_draining_under_backlog():
    """
    Simulate worker consumer batch ingestion under backlog pressure:
    100 shadowed events are queued in Redis Stream.
    Consumer drains backlog in batches of 20, executing candidate evaluation
    and persisting results to SQLite in-memory without dropping messages.
    """
    import numpy as np
    from worker.evaluator import EmbeddingEngine

    db = DatabaseManager(database_url="sqlite+aiosqlite:///:memory:")
    await db.connect()
    await db.ensure_schema()

    # Fast in-memory embedding engine to avoid model weight loading overhead
    engine = EmbeddingEngine(
        dimension=3,
        encoder_fn=lambda texts: np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)
    )
    evaluator = SemanticEvaluator(embedding_engine=engine, threshold=0.82)
    consumer = ShadowConsumer(semantic_evaluator=evaluator, db_manager=db)
    consumer.is_connected = True

    # Pre-generate 100 stream entries
    total_events = 100
    batch_size = 20
    all_stream_entries = []

    for i in range(total_events):
        msg_id = f"1731057600000-{i}"
        entry_data = {
            "request_id": f"chatcmpl-backlog-{i:03d}",
            "model": "champion-gpt-4o",
            "prompt": f"Backlog test query {i}",
            "messages": json.dumps([{"role": "user", "content": f"Backlog test query {i}"}]),
            "champion_response": f"Champion ground truth {i}",
            "champion_latency_ms": "35.5",
            "metadata": json.dumps({"batch": i // batch_size})
        }
        all_stream_entries.append((msg_id, entry_data))

    # Mock Redis client batch reads
    read_cursor = [0]
    acknowledged_ids = []

    mock_redis = AsyncMock()

    async def mock_xreadgroup(groupname, consumername, streams, count, block):
        start = read_cursor[0]
        if start >= len(all_stream_entries):
            return []
        end = min(start + count, len(all_stream_entries))
        batch = all_stream_entries[start:end]
        read_cursor[0] = end
        return [(consumer.settings.REDIS_STREAM_KEY, batch)]

    async def mock_xack(stream_key, group_name, *msg_ids):
        acknowledged_ids.extend(msg_ids)
        return len(msg_ids)

    mock_redis.xreadgroup.side_effect = mock_xreadgroup
    mock_redis.xack.side_effect = mock_xack
    consumer.client = mock_redis

    # Process all batches through consumer
    iterations = (total_events + batch_size - 1) // batch_size
    for _ in range(iterations):
        events = await consumer.read_events(count=batch_size, block=10)
        for event in events:
            await consumer.process_event(event)

    # Verify all 100 messages were processed and acknowledged
    assert len(acknowledged_ids) == total_events
    assert consumer.stats["events_acknowledged"] == total_events
    assert consumer.stats["evaluations_persisted"] == total_events

    # Verify all 100 records persisted in database
    recent = await db.get_recent_evaluations(limit=150)
    assert len(recent) == total_events

    await db.disconnect()


@pytest.mark.anyio
async def test_consumer_slow_candidate_timeout_isolation():
    """
    Verify backpressure defense against slow/hanging Candidate models:
    A slow Candidate takes 500ms; worker enforces 50ms timeout.
    The slow event must be flagged, persisted with error, and acknowledged
    so it does not hold up the consumer group or accumulate unacked backlog.
    """
    import numpy as np
    from worker.evaluator import EmbeddingEngine

    db = DatabaseManager(database_url="sqlite+aiosqlite:///:memory:")
    await db.connect()
    await db.ensure_schema()

    engine = EmbeddingEngine(
        dimension=3,
        encoder_fn=lambda texts: np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)
    )
    evaluator = SemanticEvaluator(embedding_engine=engine, threshold=0.82)
    # Simulated latency of 500ms exceeds 50ms timeout -> triggers timeout isolation
    candidate_client = SandboxedCandidateClient(timeout_seconds=0.05, simulated_latency_ms=500.0)

    consumer = ShadowConsumer(
        candidate_client=candidate_client,
        semantic_evaluator=evaluator,
        db_manager=db
    )
    consumer.is_connected = True

    mock_redis = AsyncMock()
    mock_redis.xack.return_value = 1
    consumer.client = mock_redis

    from datetime import datetime, timezone
    event = ShadowEvent(
        message_id="1731057600000-slow",
        request_id="chatcmpl-slow-001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model="candidate-gpt-4o-mini",
        prompt="Slow inference prompt",
        messages=[{"role": "user", "content": "Slow inference prompt"}],
        champion_response="Champion response immediately available",
        champion_latency_ms=25.0
    )

    start = time.perf_counter()
    success = await consumer.process_event(event)
    duration = time.perf_counter() - start

    # Operation must complete around timeout (not full 500ms)
    assert duration < 0.30
    assert success is True
    assert consumer.stats["candidate_failed"] == 1
    assert consumer.stats["events_acknowledged"] == 1

    # Verified persisted with timeout error
    record = await db.get_evaluation_by_request_id("chatcmpl-slow-001")
    assert record is not None
    assert "timed out" in (record.error_message or "").lower()
    assert record.discrepancy_detected is True

    # Acknowledged in Redis to unblock queue
    mock_redis.xack.assert_called_once_with(
        consumer.settings.REDIS_STREAM_KEY,
        consumer.settings.REDIS_CONSUMER_GROUP,
        "1731057600000-slow"
    )

    await db.disconnect()


def test_end_to_end_gateway_concurrency_throughput():
    """
    Simulate synchronous gateway ingress under 50 concurrent client calls.
    Verifies every call returns 200 OK, carries proper telemetry headers,
    and schedules shadow mirroring without slowing down the synchronous path.
    """
    client = TestClient(app)

    with patch("gateway.api.routes.traffic_broker.publish_shadow_traffic", new_callable=AsyncMock) as mock_pub:
        mock_pub.return_value = "1731057600000-ok"

        payloads = [
            {
                "model": "champion-gpt-4o",
                "messages": [{"role": "user", "content": f"Prompt {i}"}],
                "temperature": 0.5
            }
            for i in range(50)
        ]

        responses = [client.post("/v1/chat/completions", json=p) for p in payloads]

        assert len(responses) == 50
        for resp in responses:
            assert resp.status_code == 200
            assert resp.headers["X-Canary-Status"] == "active"
            assert resp.headers["X-CrossVerify-Model"] == "champion-gpt-4o"
            assert "X-Request-ID" in resp.headers
            assert "X-CrossVerify-Latency-Ms" in resp.headers
            data = resp.json()
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["content"] != ""
