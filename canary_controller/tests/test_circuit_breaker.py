import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from canary_controller.circuit_breaker import (
    CanaryCircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerDecision
)
from canary_controller.aggregator import WindowMetrics


def create_mock_redis():
    """Create an in-memory dictionary-backed mock Redis client for testing."""
    storage = {}
    mock = AsyncMock()

    async def mock_get(key):
        return storage.get(key)

    async def mock_set(key, value):
        storage[key] = str(value)
        return True

    async def mock_delete(*keys):
        count = 0
        for k in keys:
            if k in storage:
                del storage[k]
                count += 1
        return count

    mock.get.side_effect = mock_get
    mock.set.side_effect = mock_set
    mock.delete.side_effect = mock_delete
    mock._storage = storage
    return mock


def make_metrics(
    total_samples=50,
    drift_rate=0.02,
    p95_latency=350.0,
    error_rate=0.01,
    structural_pass_rate=1.0
) -> WindowMetrics:
    """Helper to generate WindowMetrics for testing."""
    now = datetime.now(timezone.utc)
    return WindowMetrics(
        window_start=now,
        window_end=now,
        window_minutes=5,
        total_samples=total_samples,
        discrepancy_count=int(total_samples * drift_rate),
        drift_rate=drift_rate,
        error_count=int(total_samples * error_rate),
        error_rate=error_rate,
        avg_champion_latency_ms=50.0,
        avg_candidate_latency_ms=p95_latency * 0.7,
        p50_candidate_latency_ms=p95_latency * 0.7,
        p95_candidate_latency_ms=p95_latency,
        p99_candidate_latency_ms=p95_latency * 1.1,
        avg_semantic_similarity=0.95,
        structural_discrepancy_count=0,
        structural_pass_rate=structural_pass_rate
    )


@pytest.fixture
def mock_redis():
    return create_mock_redis()


@pytest.fixture
def breaker(mock_redis):
    return CanaryCircuitBreaker(redis_client=mock_redis)


@pytest.mark.anyio
async def test_circuit_breaker_initial_state(breaker):
    """Verify breaker starts in CLOSED state with 0 healthy cycles."""
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.healthy_consecutive_cycles == 0
    assert breaker.last_decision is None


@pytest.mark.anyio
async def test_insufficient_samples_guard(breaker):
    """Verify evaluations with < MIN_SAMPLE_SIZE maintain state without tripping."""
    metrics = make_metrics(total_samples=4, drift_rate=0.50)  # Drift is high but samples too low
    decision = await breaker.evaluate(metrics)

    assert decision.state == CircuitBreakerState.CLOSED
    assert decision.tripped is False
    assert len(decision.violations) == 0
    assert "Insufficient sample count" in decision.reason


@pytest.mark.anyio
async def test_healthy_metrics_maintain_closed(breaker):
    """Verify safe metrics maintain CLOSED state and don't trip."""
    metrics = make_metrics(total_samples=30, drift_rate=0.02, p95_latency=400.0, error_rate=0.01)
    decision = await breaker.evaluate(metrics)

    assert decision.state == CircuitBreakerState.CLOSED
    assert decision.tripped is False
    assert len(decision.violations) == 0
    assert decision.reason == "All canary SLOs satisfied"


@pytest.mark.anyio
async def test_trip_on_drift_rate_breach(breaker, mock_redis):
    """Verify breaker trips OPEN and writes Redis abort flag when drift rate > 5%."""
    metrics = make_metrics(total_samples=40, drift_rate=0.08)  # 8% > 5%
    decision = await breaker.evaluate(metrics)

    assert decision.state == CircuitBreakerState.OPEN
    assert decision.tripped is True
    assert any("Drift rate" in v for v in decision.violations)

    # Check Redis storage
    assert mock_redis._storage.get(breaker.settings.CANARY_ABORT_KEY) == "1"
    assert mock_redis._storage.get(breaker.settings.CANARY_STATE_KEY) == "OPEN"
    reason_raw = mock_redis._storage.get(breaker.settings.CANARY_ABORT_REASON_KEY)
    assert reason_raw is not None
    reason_json = json.loads(reason_raw)
    assert reason_json["drift_rate"] == 0.08
    assert "Drift rate" in reason_json["reason"]


@pytest.mark.anyio
async def test_trip_on_latency_spike(breaker, mock_redis):
    """Verify breaker trips OPEN when candidate p95 latency > 1500ms."""
    metrics = make_metrics(total_samples=25, p95_latency=1850.0)  # 1850ms > 1500ms
    decision = await breaker.evaluate(metrics)

    assert decision.state == CircuitBreakerState.OPEN
    assert decision.tripped is True
    assert any("p95 latency" in v for v in decision.violations)
    assert mock_redis._storage.get(breaker.settings.CANARY_ABORT_KEY) == "1"


@pytest.mark.anyio
async def test_trip_on_error_rate(breaker, mock_redis):
    """Verify breaker trips OPEN when candidate error rate > 5%."""
    metrics = make_metrics(total_samples=30, error_rate=0.10)  # 10% > 5%
    decision = await breaker.evaluate(metrics)

    assert decision.state == CircuitBreakerState.OPEN
    assert decision.tripped is True
    assert any("error rate" in v for v in decision.violations)
    assert mock_redis._storage.get(breaker.settings.CANARY_ABORT_KEY) == "1"


@pytest.mark.anyio
async def test_multiple_simultaneous_violations(breaker):
    """Verify all breached SLOs are captured in violations list."""
    metrics = make_metrics(total_samples=50, drift_rate=0.12, p95_latency=2200.0, error_rate=0.15)
    decision = await breaker.evaluate(metrics)

    assert decision.state == CircuitBreakerState.OPEN
    assert decision.tripped is True
    assert len(decision.violations) == 3


@pytest.mark.anyio
async def test_cooldown_recovery_to_closed(breaker, mock_redis):
    """Verify auto-recovery transitions from OPEN -> HALF_OPEN -> CLOSED across cooldown cycles."""
    bad_metrics = make_metrics(total_samples=20, drift_rate=0.10)
    good_metrics = make_metrics(total_samples=20, drift_rate=0.01)

    # 1. Trip breaker OPEN
    await breaker.evaluate(bad_metrics)
    assert breaker.state == CircuitBreakerState.OPEN
    assert mock_redis._storage.get(breaker.settings.CANARY_ABORT_KEY) == "1"

    # 2. Cycle 1 of healthy metrics -> transitions to HALF_OPEN
    dec1 = await breaker.evaluate(good_metrics)
    assert dec1.state == CircuitBreakerState.HALF_OPEN
    assert dec1.tripped is True
    assert breaker.healthy_consecutive_cycles == 1
    # Abort flag still active during cooldown
    assert mock_redis._storage.get(breaker.settings.CANARY_ABORT_KEY) == "1"

    # 3. Cycle 2 of healthy metrics -> still HALF_OPEN
    dec2 = await breaker.evaluate(good_metrics)
    assert dec2.state == CircuitBreakerState.HALF_OPEN
    assert breaker.healthy_consecutive_cycles == 2

    # 4. Cycle 3 of healthy metrics -> auto-recovers to CLOSED!
    dec3 = await breaker.evaluate(good_metrics)
    assert dec3.state == CircuitBreakerState.CLOSED
    assert dec3.tripped is False
    assert breaker.healthy_consecutive_cycles == 0
    # Abort flag cleared!
    assert breaker.settings.CANARY_ABORT_KEY not in mock_redis._storage
    assert breaker.settings.CANARY_ABORT_REASON_KEY not in mock_redis._storage
    assert mock_redis._storage.get(breaker.settings.CANARY_STATE_KEY) == "CLOSED"


@pytest.mark.anyio
async def test_manual_reset(breaker, mock_redis):
    """Verify manual reset clears OPEN state and deletes Redis abort flag immediately."""
    bad_metrics = make_metrics(total_samples=20, drift_rate=0.10)
    await breaker.evaluate(bad_metrics)
    assert breaker.state == CircuitBreakerState.OPEN

    # Perform manual reset
    await breaker.manual_reset()

    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.healthy_consecutive_cycles == 0
    assert breaker.settings.CANARY_ABORT_KEY not in mock_redis._storage
    assert mock_redis._storage.get(breaker.settings.CANARY_STATE_KEY) == "CLOSED"


@pytest.mark.anyio
async def test_get_live_status(breaker, mock_redis):
    """Verify live status retrieval formats state and diagnostic reason."""
    bad_metrics = make_metrics(total_samples=30, drift_rate=0.09)
    await breaker.evaluate(bad_metrics)

    status = await breaker.get_live_status()
    assert status["state"] == "OPEN"
    assert status["abort_active"] is True
    assert status["abort_reason"]["drift_rate"] == 0.09
    assert "checked_at" in status