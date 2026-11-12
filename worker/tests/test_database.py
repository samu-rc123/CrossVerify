import pytest
from worker.db import DatabaseManager


@pytest.fixture
async def db_manager():
    """Provision an isolated in-memory async database for schema & persistence tests."""
    manager = DatabaseManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.connect()
    await manager.ensure_schema()
    yield manager
    await manager.disconnect()


@pytest.mark.anyio
async def test_ensure_schema_and_save_evaluation(db_manager):
    """Verify ORM schema creation and telemetry record persistence."""
    record = await db_manager.save_evaluation(
        request_id="chatcmpl-db-001",
        champion_model="champion-gpt-4o",
        candidate_model="candidate-gpt-4o-mini",
        prompt="Explain event-driven architecture",
        champion_response="Event-driven architecture uses events to trigger communication...",
        candidate_response="EDA communicates across decoupled services via events...",
        champion_latency_ms=62.5,
        candidate_latency_ms=88.1,
        semantic_similarity=0.91,
        structural_valid=True,
        discrepancy_detected=False,
        metadata={"env": "test", "stream_id": "1731400000-0"}
    )

    assert record is not None
    assert record.id is not None
    assert record.request_id == "chatcmpl-db-001"
    assert record.champion_model == "champion-gpt-4o"
    assert record.candidate_model == "candidate-gpt-4o-mini"
    assert record.semantic_similarity == 0.91
    assert record.discrepancy_detected is False
    assert record.extra_metadata == {"env": "test", "stream_id": "1731400000-0"}
    assert record.created_at is not None
    assert db_manager.stats["records_saved"] == 1


@pytest.mark.anyio
async def test_get_evaluation_by_request_id(db_manager):
    """Verify querying telemetry records by request_id."""
    await db_manager.save_evaluation(
        request_id="chatcmpl-lookup-42",
        prompt="Test query",
        champion_response="Champion output",
        candidate_response="Candidate output",
        semantic_similarity=0.84,
        discrepancy_detected=False
    )

    found = await db_manager.get_evaluation_by_request_id("chatcmpl-lookup-42")
    assert found is not None
    assert found.request_id == "chatcmpl-lookup-42"
    assert found.semantic_similarity == 0.84

    missing = await db_manager.get_evaluation_by_request_id("nonexistent-id")
    assert missing is None


@pytest.mark.anyio
async def test_get_recent_evaluations_ordering(db_manager):
    """Verify recent evaluations query returns records in descending chronological order."""
    for i in range(5):
        await db_manager.save_evaluation(
            request_id=f"chatcmpl-batch-{i}",
            prompt=f"Prompt {i}",
            champion_response=f"Champion {i}",
            candidate_response=f"Candidate {i}",
            semantic_similarity=0.80 + (i * 0.02),
            discrepancy_detected=(i == 0)
        )

    recent = await db_manager.get_recent_evaluations(limit=3)
    assert len(recent) == 3
    assert recent[0].request_id == "chatcmpl-batch-4"
    assert recent[1].request_id == "chatcmpl-batch-3"
    assert recent[2].request_id == "chatcmpl-batch-2"


@pytest.mark.anyio
async def test_database_graceful_failure_when_unreachable():
    """Verify save_evaluation does not raise exceptions when database is unreachable."""
    manager = DatabaseManager(database_url="postgresql+asyncpg://bad:bad@127.0.0.1:59999/none")
    result = await manager.save_evaluation(
        request_id="chatcmpl-offline",
        prompt="Offline test",
        champion_response="A",
        candidate_response="B"
    )
    assert result is None
    assert manager.stats["save_failures"] == 1
