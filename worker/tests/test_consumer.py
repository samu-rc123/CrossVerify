import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from redis.exceptions import ResponseError, RedisError

from worker.consumer import ShadowConsumer, ShadowEvent


@pytest.fixture
def consumer():
    """Create fresh instance of ShadowConsumer for testing."""
    return ShadowConsumer()


def test_shadow_event_from_stream_entry():
    """Verify deserialization of raw Redis Stream fields into ShadowEvent."""
    message_id = "1731140000000-0"
    fields = {
        "request_id": "chatcmpl-test-123",
        "timestamp": "2026-11-09T04:00:00Z",
        "model": "champion-gpt-4o",
        "prompt": "Explain Byzantine Fault Tolerance",
        "messages": json.dumps([
            {"role": "user", "content": "Explain Byzantine Fault Tolerance"}
        ]),
        "temperature": "0.7",
        "max_tokens": "250",
        "champion_response": "BFT is a condition of a distributed computing system...",
        "champion_latency_ms": "55.40",
        "metadata": json.dumps({"source": "unit_test"})
    }

    event = ShadowEvent.from_stream_entry(message_id, fields)

    assert event.message_id == message_id
    assert event.request_id == "chatcmpl-test-123"
    assert event.model == "champion-gpt-4o"
    assert event.prompt == "Explain Byzantine Fault Tolerance"
    assert len(event.messages) == 1
    assert event.temperature == 0.7
    assert event.max_tokens == 250
    assert event.champion_latency_ms == 55.40
    assert event.metadata == {"source": "unit_test"}


def test_shadow_event_corrupt_json_fallback():
    """Verify resilience when stream payload has malformed JSON fields."""
    message_id = "1731140000001-0"
    fields = {
        "request_id": "chatcmpl-corrupt",
        "prompt": "Test prompt",
        "messages": "INVALID_JSON{",
        "metadata": "{NOT_VALID_JSON}",
        "champion_response": "Response",
        "champion_latency_ms": "40.0"
    }

    event = ShadowEvent.from_stream_entry(message_id, fields)
    assert event.messages == []
    assert event.metadata == {}
    assert event.champion_response == "Response"


@pytest.mark.anyio
async def test_ensure_consumer_group_creates_new(consumer):
    """Verify consumer group creation with MKSTREAM=True."""
    mock_redis = AsyncMock()
    mock_redis.xgroup_create.return_value = True
    consumer.client = mock_redis
    consumer.is_connected = True

    result = await consumer.ensure_consumer_group()
    assert result is True
    mock_redis.xgroup_create.assert_called_once_with(
        name=consumer.settings.REDIS_STREAM_KEY,
        groupname=consumer.settings.REDIS_CONSUMER_GROUP,
        id="0",
        mkstream=True
    )


@pytest.mark.anyio
async def test_ensure_consumer_group_handles_busygroup(consumer):
    """Verify BUSYGROUP error is caught and treated as success."""
    mock_redis = AsyncMock()
    mock_redis.xgroup_create.side_effect = ResponseError("BUSYGROUP Consumer Group name already exists")
    consumer.client = mock_redis
    consumer.is_connected = True

    result = await consumer.ensure_consumer_group()
    assert result is True


@pytest.mark.anyio
async def test_read_events_success(consumer):
    """Verify reading and parsing events from XREADGROUP."""
    mock_redis = AsyncMock()
    raw_stream_data = [
        (
            "shadow_traffic",
            [
                (
                    "1731140000000-0",
                    {
                        "request_id": "chatcmpl-001",
                        "model": "champion-gpt-4o",
                        "prompt": "Hello",
                        "messages": json.dumps([{"role": "user", "content": "Hello"}]),
                        "champion_response": "Hi there!",
                        "champion_latency_ms": "35.0"
                    }
                )
            ]
        )
    ]
    mock_redis.xreadgroup.return_value = raw_stream_data
    consumer.client = mock_redis
    consumer.is_connected = True

    events = await consumer.read_events(count=5, block=100)

    assert len(events) == 1
    assert events[0].request_id == "chatcmpl-001"
    assert events[0].champion_response == "Hi there!"
    assert consumer.stats["events_received"] == 1


@pytest.mark.anyio
async def test_read_events_empty(consumer):
    """Verify empty result when no new events exist in stream."""
    mock_redis = AsyncMock()
    mock_redis.xreadgroup.return_value = []
    consumer.client = mock_redis
    consumer.is_connected = True

    events = await consumer.read_events(count=5, block=50)
    assert events == []


@pytest.mark.anyio
async def test_acknowledge_event(consumer):
    """Verify XACK is dispatched with correct stream key and message ID."""
    mock_redis = AsyncMock()
    mock_redis.xack.return_value = 1
    consumer.client = mock_redis
    consumer.is_connected = True

    success = await consumer.acknowledge_event("1731140000000-0")
    assert success is True
    assert consumer.stats["events_acknowledged"] == 1
    mock_redis.xack.assert_called_once_with(
        consumer.settings.REDIS_STREAM_KEY,
        consumer.settings.REDIS_CONSUMER_GROUP,
        "1731140000000-0"
    )


@pytest.mark.anyio
async def test_process_event(consumer):
    """Verify processing an event acknowledges it and returns True."""
    mock_redis = AsyncMock()
    mock_redis.xack.return_value = 1
    consumer.client = mock_redis
    consumer.is_connected = True

    event = ShadowEvent(
        message_id="1731140000000-0",
        request_id="chatcmpl-proc-test",
        timestamp="2026-11-09T04:00:00Z",
        model="champion-gpt-4o",
        prompt="Process test",
        messages=[{"role": "user", "content": "Process test"}],
        champion_response="Acknowledged response",
        champion_latency_ms=45.0
    )

    success = await consumer.process_event(event)
    assert success is True
    assert consumer.stats["events_acknowledged"] == 1
    assert consumer.stats["candidate_success"] == 1


@pytest.mark.anyio
async def test_process_event_candidate_timeout_still_acknowledges():
    """Verify sandboxed candidate timeout is isolated and event is still ACKed."""
    from worker.clients import SandboxedCandidateClient
    slow_candidate = SandboxedCandidateClient(timeout_seconds=0.02, simulated_latency_ms=250.0)
    consumer = ShadowConsumer(candidate_client=slow_candidate)

    mock_redis = AsyncMock()
    mock_redis.xack.return_value = 1
    consumer.client = mock_redis
    consumer.is_connected = True

    event = ShadowEvent(
        message_id="1731140000002-0",
        request_id="chatcmpl-timeout-test",
        timestamp="2026-11-12T04:00:00Z",
        model="champion-gpt-4o",
        prompt="Slow prompt",
        messages=[{"role": "user", "content": "Slow prompt"}],
        champion_response="Fast champion response",
        champion_latency_ms=30.0
    )

    success = await consumer.process_event(event)
    assert success is True
    assert consumer.stats["candidate_failed"] == 1
    assert consumer.stats["events_acknowledged"] == 1


@pytest.mark.anyio
async def test_worker_run_bounded_iterations(consumer):
    """Verify bounded run loop terminates gracefully without infinite blocking."""
    mock_redis = AsyncMock()
    consumer.client = mock_redis
    consumer.is_connected = True

    with patch.object(consumer, "ensure_consumer_group", new_callable=AsyncMock) as mock_group, \
         patch.object(consumer, "read_events", new_callable=AsyncMock) as mock_read:
        mock_group.return_value = True
        mock_read.return_value = []

        # Run for 2 iterations and verify it exits cleanly
        await consumer.run(max_iterations=2)

        assert consumer.is_running is False
        assert mock_group.call_count == 1
        assert mock_read.call_count == 2
