import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from gateway.main import app
from gateway.core.broker import traffic_broker


@pytest.fixture
def client():
    return TestClient(app)


def test_canary_status_header_when_healthy(client):
    """Verify completions return 'X-Canary-Status: active' when circuit breaker is closed."""
    with patch.object(traffic_broker, "is_circuit_breaker_tripped", new=AsyncMock(return_value=False)):
        payload = {
            "model": "champion-gpt-4o",
            "messages": [{"role": "user", "content": "What is continuous verification?"}]
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers.get("X-Canary-Status") == "active"
        assert "X-Request-ID" in response.headers
        assert "X-CrossVerify-Latency-Ms" in response.headers


def test_canary_status_header_and_drop_when_aborted(client):
    """Verify completions return 'X-Canary-Status: aborted' and drop shadow routing when breaker is tripped."""
    initial_dropped = traffic_broker.stats["dropped_circuit_breaker"]

    with patch.object(traffic_broker, "is_circuit_breaker_tripped", new=AsyncMock(return_value=True)):
        payload = {
            "model": "champion-gpt-4o",
            "messages": [{"role": "user", "content": "Should shadow traffic run?"}]
        }
        response = client.post("/v1/chat/completions", json=payload)

        # Champion still serves user seamlessly!
        assert response.status_code == 200
        assert response.headers.get("X-Canary-Status") == "aborted"
        data = response.json()
        assert len(data["choices"]) > 0

        # Dropped counter incremented
        assert traffic_broker.stats["dropped_circuit_breaker"] == initial_dropped + 1


def test_get_canary_status_endpoint_closed(client):
    """Verify GET /v1/canary/status returns live status when circuit breaker is closed."""
    mock_status = {
        "state": "CLOSED",
        "abort_active": False,
        "abort_reason": None,
        "broker_connected": True,
        "broker_stats": {"published": 10, "failed": 0, "dropped_circuit_breaker": 0}
    }

    with patch.object(traffic_broker, "get_circuit_breaker_status", new=AsyncMock(return_value=mock_status)):
        response = client.get("/v1/canary/status")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "CLOSED"
        assert data["abort_active"] is False
        assert data["abort_reason"] is None
        assert data["broker_connected"] is True


def test_get_canary_status_endpoint_open_with_diagnostics(client):
    """Verify GET /v1/canary/status returns diagnostics when circuit breaker is tripped OPEN."""
    mock_status = {
        "state": "OPEN",
        "abort_active": True,
        "abort_reason": {
            "drift_rate": 0.082,
            "violations": ["Drift rate 8.2% breached threshold 5.0%"]
        },
        "broker_connected": True,
        "broker_stats": {"published": 50, "failed": 0, "dropped_circuit_breaker": 5}
    }

    with patch.object(traffic_broker, "get_circuit_breaker_status", new=AsyncMock(return_value=mock_status)):
        response = client.get("/v1/canary/status")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "OPEN"
        assert data["abort_active"] is True
        assert data["abort_reason"]["drift_rate"] == 0.082
        assert len(data["abort_reason"]["violations"]) == 1


def test_post_canary_reset_endpoint(client):
    """Verify POST /v1/canary/reset executes reset and returns confirmation."""
    with patch.object(traffic_broker, "reset_circuit_breaker", new=AsyncMock(return_value=True)):
        response = client.post("/v1/canary/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset_successful"
        assert "reset to CLOSED" in data["message"]


@pytest.mark.anyio
async def test_broker_get_circuit_breaker_status_redis_integration():
    """Verify broker get_circuit_breaker_status queries Redis keys and parses JSON reason."""
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = lambda k: {
        traffic_broker.settings.CANARY_ABORT_KEY: "1",
        traffic_broker.settings.CANARY_STATE_KEY: "OPEN",
        traffic_broker.settings.CANARY_ABORT_REASON_KEY: json.dumps({"reason": "p95 latency spike", "p95": 1600.0})
    }.get(k)

    traffic_broker.client = mock_redis
    traffic_broker.is_connected = True

    status_data = await traffic_broker.get_circuit_breaker_status()
    assert status_data["state"] == "OPEN"
    assert status_data["abort_active"] is True
    assert status_data["abort_reason"]["reason"] == "p95 latency spike"
    assert status_data["abort_reason"]["p95"] == 1600.0


@pytest.mark.anyio
async def test_broker_reset_circuit_breaker_redis_integration():
    """Verify broker reset_circuit_breaker clears Redis keys and sets state to CLOSED."""
    deleted_keys = []
    set_records = {}

    mock_redis = AsyncMock()
    mock_redis.delete.side_effect = lambda *keys: deleted_keys.extend(keys) or len(keys)
    mock_redis.set.side_effect = lambda k, v: set_records.update({k: v})

    traffic_broker.client = mock_redis
    traffic_broker.is_connected = True

    success = await traffic_broker.reset_circuit_breaker()
    assert success is True
    assert traffic_broker.settings.CANARY_ABORT_KEY in deleted_keys
    assert traffic_broker.settings.CANARY_ABORT_REASON_KEY in deleted_keys
    assert set_records.get(traffic_broker.settings.CANARY_STATE_KEY) == "CLOSED"
