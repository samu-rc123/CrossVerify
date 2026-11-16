import pytest
from unittest.mock import AsyncMock

from worker.evaluator.structural import StructuralEvaluator, StructuralComparisonResult
from worker.consumer import ShadowConsumer, ShadowEvent
from worker.clients import SandboxedCandidateClient
from worker.db import DatabaseManager


@pytest.fixture
def evaluator():
    return StructuralEvaluator()


def test_extract_json_direct(evaluator):
    """Verify direct raw JSON string parsing."""
    raw = '{"name": "CrossVerify", "active": true, "score": 98.5}'
    data, ok = evaluator.extract_json(raw)
    assert ok is True
    assert data["name"] == "CrossVerify"
    assert data["active"] is True
    assert data["score"] == 98.5


def test_extract_json_markdown_fence(evaluator):
    """Verify extraction of JSON wrapped in markdown code fences."""
    fenced = (
        "Here is the function call payload:\n"
        "```json\n"
        "{\n"
        '  "tool": "search_db",\n'
        '  "parameters": {"query": "latency spikes", "limit": 10}\n'
        "}\n"
        "```\n"
        "End of response."
    )
    data, ok = evaluator.extract_json(fenced)
    assert ok is True
    assert data["tool"] == "search_db"
    assert data["parameters"]["limit"] == 10


def test_extract_json_malformed(evaluator):
    """Verify malformed JSON fails cleanly without raising exceptions."""
    malformed = '{"action": "search", "broken": [1, 2, }'
    data, ok = evaluator.extract_json(malformed)
    assert ok is False
    assert data is None


def test_validate_against_schema_valid_and_invalid(evaluator):
    """Verify jsonschema validation against required properties and types."""
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"}
        },
        "required": ["query", "max_results"]
    }

    valid_payload = {"query": "LLM verification", "max_results": 5}
    ok, err = evaluator.validate_against_schema(valid_payload, schema)
    assert ok is True
    assert err is None

    invalid_payload = {"query": "LLM verification", "max_results": "NOT_AN_INT"}
    ok_inv, err_inv = evaluator.validate_against_schema(invalid_payload, schema)
    assert ok_inv is False
    assert "Schema validation error" in err_inv


def test_evaluate_explicit_schema_success(evaluator):
    """Verify evaluation passes when candidate conforms to required schema."""
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
        },
        "required": ["sentiment"]
    }
    champ = '{"sentiment": "positive", "confidence": 0.99}'
    cand = '{"sentiment": "positive"}'

    res = evaluator.evaluate(champ, cand, expected_schema=schema)
    assert res.is_structured is True
    assert res.syntax_valid is True
    assert res.schema_matched is True
    assert res.structural_valid is True


def test_evaluate_explicit_schema_failure(evaluator):
    """Verify evaluation fails when candidate violates required schema."""
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]}
        },
        "required": ["status"]
    }
    champ = '{"status": "ok"}'
    cand = '{"status": "unknown_status_val"}'

    res = evaluator.evaluate(champ, cand, expected_schema=schema)
    assert res.is_structured is True
    assert res.syntax_valid is True
    assert res.schema_matched is False
    assert res.structural_valid is False
    assert "Schema validation error" in res.error_message


def test_evaluate_implicit_champion_json_detected_as_broken(evaluator):
    """Verify discrepancy flagged when champion outputs structured JSON but candidate returns unstructured prose."""
    champ = '{"action": "execute_query", "table": "users"}'
    cand = "I will go ahead and look up the users table for you."

    res = evaluator.evaluate(champ, cand)
    assert res.is_structured is True
    assert res.syntax_valid is False
    assert res.structural_valid is False
    assert "Candidate returned unparseable prose" in res.error_message


def test_evaluate_implicit_champion_json_missing_keys(evaluator):
    """Verify discrepancy flagged when candidate JSON misses expected keys found in champion output."""
    champ = '{"action": "create_order", "user_id": 123, "sku": "ITEM_99"}'
    cand = '{"action": "create_order"}'

    res = evaluator.evaluate(champ, cand)
    assert res.is_structured is True
    assert res.syntax_valid is True
    assert res.structural_valid is False
    assert "missing expected keys" in res.error_message


def test_evaluate_unstructured_prose(evaluator):
    """Verify freeform natural language is treated as structurally valid."""
    champ = "Paris is the capital of France."
    cand = "The capital city of France is Paris."

    res = evaluator.evaluate(champ, cand)
    assert res.is_structured is False
    assert res.structural_valid is True


@pytest.mark.anyio
async def test_consumer_structural_discrepancy_persisted_to_db():
    """Verify structural discrepancy is persisted in PostgreSQL telemetry when candidate breaks JSON."""
    db = DatabaseManager(database_url="sqlite+aiosqlite:///:memory:")
    await db.ensure_schema()

    # Candidate returns freeform text instead of JSON
    candidate = SandboxedCandidateClient(simulated_latency_ms=5.0)
    consumer = ShadowConsumer(candidate_client=candidate, db_manager=db)

    mock_redis = AsyncMock()
    mock_redis.xack.return_value = 1
    consumer.client = mock_redis
    consumer.is_connected = True

    event = ShadowEvent(
        message_id="1731140000003-0",
        request_id="chatcmpl-struct-discrepancy",
        timestamp="2026-11-16T10:00:00Z",
        model="champion-gpt-4o",
        prompt="Output JSON tool call",
        messages=[{"role": "user", "content": "Output JSON tool call"}],
        champion_response='{"function": "get_weather", "location": "Tokyo"}',
        champion_latency_ms=45.0
    )

    success = await consumer.process_event(event)
    assert success is True
    assert consumer.stats["structural_discrepancies"] == 1
    assert consumer.stats["discrepancies_detected"] == 1

    record = await db.get_evaluation_by_request_id("chatcmpl-struct-discrepancy")
    assert record is not None
    assert record.structural_valid is False
    assert record.discrepancy_detected is True
    assert "Candidate returned unparseable prose" in record.error_message
    assert record.extra_metadata["is_structured"] is True
    await db.disconnect()
