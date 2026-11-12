import pytest
from unittest.mock import patch

from worker.clients import ChampionMockClient, SandboxedCandidateClient


@pytest.mark.anyio
async def test_champion_mock_generate():
    """Verify ChampionMockClient returns expected completion structure and latency."""
    client = ChampionMockClient(simulated_latency_ms=10.0)
    messages = [{"role": "user", "content": "Hello champion"}]

    result = await client.generate(messages=messages)

    assert result.success is True
    assert result.model == "champion-gpt-4o"
    assert "Hello!" in result.content
    assert result.latency_ms >= 5.0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens


@pytest.mark.anyio
async def test_candidate_mock_normal_generation():
    """Verify SandboxedCandidateClient produces semantically aligned output under normal mode."""
    client = SandboxedCandidateClient(
        model_name="candidate-test",
        timeout_seconds=2.0,
        simulated_latency_ms=10.0,
        drift_rate=0.0
    )
    messages = [{"role": "user", "content": "Write python code for filtering"}]

    result = await client.generate(messages=messages)

    assert result.success is True
    assert result.timed_out is False
    assert result.error_message is None
    assert result.model == "candidate-test"
    assert "def solution" in result.content
    assert result.latency_ms >= 5.0


@pytest.mark.anyio
async def test_candidate_mock_forced_drift():
    """Verify SandboxedCandidateClient triggers divergent output when force_drift=True."""
    client = SandboxedCandidateClient(simulated_latency_ms=5.0)
    messages = [{"role": "user", "content": "Hello there"}]

    result = await client.generate(messages=messages, force_drift=True)

    assert result.success is True
    assert "ERROR_DRIFT" in result.content


@pytest.mark.anyio
async def test_candidate_sandbox_timeout_isolation():
    """Verify strict asyncio timeout catches slow inference without crashing worker."""
    client = SandboxedCandidateClient(
        model_name="slow-candidate",
        timeout_seconds=0.05,
        simulated_latency_ms=500.0
    )
    messages = [{"role": "user", "content": "Take too long"}]

    result = await client.generate(messages=messages)

    assert result.success is False
    assert result.timed_out is True
    assert result.content == ""
    assert "timed out" in result.error_message.lower()
    assert result.latency_ms >= 40.0


@pytest.mark.anyio
async def test_candidate_sandbox_exception_isolation():
    """Verify unexpected runtime exceptions are isolated inside InferenceResult."""
    client = SandboxedCandidateClient(timeout_seconds=2.0)
    messages = [{"role": "user", "content": "Crash test"}]

    with patch.object(client, "_execute_inference", side_effect=RuntimeError("Upstream GPU OOM")):
        result = await client.generate(messages=messages)

    assert result.success is False
    assert result.timed_out is False
    assert "Upstream GPU OOM" in result.error_message
