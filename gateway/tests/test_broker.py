import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from redis.exceptions import RedisError

from gateway.core.broker import TrafficBroker


@pytest.fixture
def broker():
    """Create a fresh TrafficBroker instance for testing."""
    instance = TrafficBroker()
    return instance


@pytest.mark.anyio
async def test_broker_initial_state(broker):
    """Verify default initial state of the broker."""
    assert not broker.is_connected
    assert broker.client is None
    assert broker.stats["published"] == 0
    assert broker.stats["failed"] == 0
    assert broker.stats["dropped_circuit_breaker"] == 0


@pytest.mark.anyio
async def test_publish_shadow_traffic_success(broker):
    """Verify successful XADD stream publishing with proper payload formatting."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None  # Circuit breaker not tripped
    mock_redis.xadd.return_value = "1731057600000-0"

    broker.client = mock_redis
    broker.is_connected = True

    payload = {
        "model": "champion-gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "Analyze system telemetry"}
        ],
        "temperature": 0.5,
        "max_tokens": 200,
        "metadata": {"user_id": "usr_test"}
    }

    msg_id = await broker.publish_shadow_traffic(
        request_id="chatcmpl-test-001",
        request_payload=payload,
        champion_response="System telemetry is optimal.",
        champion_latency_ms=42.50
    )

    assert msg_id == "1731057600000-0"
    assert broker.stats["published"] == 1
    assert broker.stats["failed"] == 0

    # Verify xadd call arguments
    mock_redis.xadd.assert_called_once()
    call_kwargs = mock_redis.xadd.call_args.kwargs
    assert call_kwargs["name"] == broker.settings.REDIS_STREAM_KEY
    assert call_kwargs["maxlen"] == broker.settings.REDIS_STREAM_MAXLEN
    assert call_kwargs["approximate"] is True

    fields = call_kwargs["fields"]
    assert fields["request_id"] == "chatcmpl-test-001"
    assert fields["model"] == "champion-gpt-4o"
    assert fields["prompt"] == "Analyze system telemetry"
    assert fields["temperature"] == "0.5"
    assert fields["max_tokens"] == "200"
    assert fields["champion_response"] == "System telemetry is optimal."
    assert fields["champion_latency_ms"] == "42.50"
    assert json.loads(fields["messages"]) == payload["messages"]
    assert json.loads(fields["metadata"]) == payload["metadata"]


@pytest.mark.anyio
async def test_publish_shadow_traffic_circuit_breaker_tripped(broker):
    """Verify shadow publishing is aborted when canary circuit breaker flag is active."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = "1"  # Circuit breaker is tripped!

    broker.client = mock_redis
    broker.is_connected = True

    payload = {
        "model": "champion-gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}]
    }

    msg_id = await broker.publish_shadow_traffic(
        request_id="chatcmpl-test-cb",
        request_payload=payload,
        champion_response="Hello!",
        champion_latency_ms=30.0
    )

    assert msg_id is None
    assert broker.stats["dropped_circuit_breaker"] == 1
    assert broker.stats["published"] == 0
    mock_redis.xadd.assert_not_called()


@pytest.mark.anyio
async def test_publish_shadow_traffic_redis_error_resilience(broker):
    """Verify exceptions in Redis do not propagate and increment failed stats."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.xadd.side_effect = RedisError("Stream buffer full / Connection lost")

    broker.client = mock_redis
    broker.is_connected = True

    payload = {
        "model": "champion-gpt-4o",
        "messages": [{"role": "user", "content": "Will this crash the user?"}]
    }

    # Must NOT raise exception to preserve caller reliability
    msg_id = await broker.publish_shadow_traffic(
        request_id="chatcmpl-test-err",
        request_payload=payload,
        champion_response="User should still get their response.",
        champion_latency_ms=25.0
    )

    assert msg_id is None
    assert broker.stats["failed"] == 1
    assert broker.stats["published"] == 0


@pytest.mark.anyio
async def test_broker_get_metrics(broker):
    """Verify telemetry dictionary contains all expected health and traffic fields."""
    metrics = broker.get_metrics()
    assert "is_connected" in metrics
    assert "stream_key" in metrics
    assert "published_count" in metrics
    assert "failed_count" in metrics
    assert "dropped_circuit_breaker_count" in metrics
