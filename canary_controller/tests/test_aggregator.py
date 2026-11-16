import pytest
from datetime import datetime, timedelta, timezone

from canary_controller.aggregator import TelemetryAggregator
from worker.db import DatabaseManager, EvaluationRecord


@pytest.fixture
async def setup_db():
    """Provision isolated in-memory DB for telemetry aggregation tests."""
    db = DatabaseManager(database_url="sqlite+aiosqlite:///:memory:")
    await db.connect()
    await db.ensure_schema()
    aggregator = TelemetryAggregator(engine=db.engine)
    yield db, aggregator
    await aggregator.close()
    await db.disconnect()


@pytest.mark.anyio
async def test_empty_window_metrics(setup_db):
    """Verify empty evaluations table returns safe zeroed metrics without crashing."""
    _, aggregator = setup_db
    metrics = await aggregator.compute_window_metrics(window_minutes=5)

    assert metrics.total_samples == 0
    assert metrics.discrepancy_count == 0
    assert metrics.drift_rate == 0.0
    assert metrics.error_count == 0
    assert metrics.error_rate == 0.0
    assert metrics.avg_candidate_latency_ms == 0.0
    assert metrics.p95_candidate_latency_ms == 0.0
    assert metrics.structural_pass_rate == 1.0


@pytest.mark.anyio
async def test_sliding_window_computations(setup_db):
    """Verify statistical rollups: drift rate, error rate, p50/p95/p99 latencies, similarities."""
    db, aggregator = setup_db
    now = datetime.now(timezone.utc)

    # Insert 20 records inside window
    for i in range(20):
        # Introduce 2 discrepancies (10% drift rate)
        has_disc = (i < 2)
        # Introduce 1 error (5% error rate)
        err = "Timeout" if i == 0 else None
        # Introduce 1 structural defect
        struct_ok = False if i == 1 else True
        # Set controlled latency: 100ms + i * 20ms -> range [100ms, 480ms]
        lat = 100.0 + (i * 20.0)

        record = EvaluationRecord(
            request_id=f"chatcmpl-win-{i}",
            champion_model="champion-gpt-4o",
            candidate_model="candidate-gpt-4o-mini",
            prompt=f"Test prompt {i}",
            champion_response="Champion",
            candidate_response="Candidate",
            champion_latency_ms=50.0,
            candidate_latency_ms=lat,
            semantic_similarity=0.95 if not has_disc else 0.40,
            structural_valid=struct_ok,
            discrepancy_detected=has_disc,
            error_message=err,
            created_at=now - timedelta(minutes=1)
        )
        async with db.session_factory() as session:
            session.add(record)
            await session.commit()

    metrics = await aggregator.compute_window_metrics(window_minutes=5, as_of=now)

    assert metrics.total_samples == 20
    assert metrics.discrepancy_count == 2
    assert metrics.drift_rate == 0.10
    assert metrics.error_count == 1
    assert metrics.error_rate == 0.05
    assert metrics.structural_discrepancy_count == 1
    assert metrics.structural_pass_rate == 0.95

    # Latencies range from 100 to 480, mean is 290
    assert metrics.avg_champion_latency_ms == 50.0
    assert metrics.avg_candidate_latency_ms == 290.0
    assert metrics.p50_candidate_latency_ms == 290.0
    assert metrics.p95_candidate_latency_ms > 450.0
    assert metrics.p99_candidate_latency_ms > 470.0


@pytest.mark.anyio
async def test_sliding_window_time_cutoff_filtering(setup_db):
    """Verify records older than window cutoff are excluded from aggregation."""
    db, aggregator = setup_db
    now = datetime.now(timezone.utc)

    # 3 old records (10 minutes ago, outside 5-min window)
    for i in range(3):
        rec_old = EvaluationRecord(
            request_id=f"chatcmpl-old-{i}",
            champion_model="champion-gpt-4o",
            candidate_model="candidate-gpt-4o-mini",
            prompt="Old prompt",
            champion_response="Old",
            candidate_response="Old",
            champion_latency_ms=50.0,
            candidate_latency_ms=100.0,
            discrepancy_detected=True,
            created_at=now - timedelta(minutes=10)
        )
        async with db.session_factory() as session:
            session.add(rec_old)
            await session.commit()

    # 5 fresh records (2 minutes ago, inside 5-min window)
    for i in range(5):
        rec_fresh = EvaluationRecord(
            request_id=f"chatcmpl-fresh-{i}",
            champion_model="champion-gpt-4o",
            candidate_model="candidate-gpt-4o-mini",
            prompt="Fresh prompt",
            champion_response="Fresh",
            candidate_response="Fresh",
            champion_latency_ms=50.0,
            candidate_latency_ms=80.0,
            discrepancy_detected=False,
            created_at=now - timedelta(minutes=2)
        )
        async with db.session_factory() as session:
            session.add(rec_fresh)
            await session.commit()

    metrics = await aggregator.compute_window_metrics(window_minutes=5, as_of=now)

    # Only the 5 fresh records should be counted!
    assert metrics.total_samples == 5
    assert metrics.discrepancy_count == 0
    assert metrics.drift_rate == 0.0
    assert metrics.avg_candidate_latency_ms == 80.0
