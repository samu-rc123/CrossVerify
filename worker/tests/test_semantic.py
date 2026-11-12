import numpy as np
import pytest

from worker.evaluator.semantic import EmbeddingEngine, SemanticEvaluator


@pytest.fixture(scope="module")
def shared_evaluator():
    """Shared SemanticEvaluator backed by real all-MiniLM-L6-v2 and FAISS IndexFlatIP."""
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    return SemanticEvaluator(embedding_engine=engine, threshold=0.75)


def test_faiss_cosine_similarity_geometry():
    """Verify FAISS IndexFlatIP computes exact cosine scores for identical, orthogonal, opposite, and zero vectors."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2_same = np.array([5.0, 0.0, 0.0], dtype=np.float32)  # Unnormalized same direction
    v3_ortho = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v4_opp = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    v5_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    assert np.isclose(SemanticEvaluator.compute_cosine_similarity(v1, v2_same), 1.0, atol=1e-4)
    assert np.isclose(SemanticEvaluator.compute_cosine_similarity(v1, v3_ortho), 0.0, atol=1e-4)
    assert np.isclose(SemanticEvaluator.compute_cosine_similarity(v1, v4_opp), -1.0, atol=1e-4)
    assert SemanticEvaluator.compute_cosine_similarity(v1, v5_zero) == 0.0


def test_faiss_cosine_similarity_dimension_mismatch():
    """Verify dimension mismatch raises explicit ValueError."""
    with pytest.raises(ValueError):
        SemanticEvaluator.compute_cosine_similarity(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )


@pytest.mark.anyio
async def test_semantic_evaluator_equivalent_paraphrase(shared_evaluator):
    """Verify semantically equivalent paraphrases score high similarity without flagging discrepancy."""
    await shared_evaluator.warm_up()

    champion_text = "Hello! How can I assist you with your tasks today?"
    candidate_text = "Hi there! What can I help you accomplish today?"

    result = await shared_evaluator.evaluate(champion_text, candidate_text)

    assert result.similarity_score >= 0.75
    assert result.discrepancy_detected is False
    assert result.embedding_latency_ms >= 0.0


@pytest.mark.anyio
async def test_semantic_evaluator_detects_drift(shared_evaluator):
    """Verify completely divergent outputs yield low cosine similarity and trigger discrepancy_detected."""
    await shared_evaluator.warm_up()

    champion_text = (
        "Here is an idiomatic Python solution:\n"
        "def solution(items: list) -> list:\n"
        "    return [item for item in items if item is not None]"
    )
    candidate_text = (
        "ERROR_DRIFT: Unrelated weather forecast for Antarctica — "
        "penguins are experiencing -40C winds with zero visibility."
    )

    result = await shared_evaluator.evaluate(champion_text, candidate_text)

    assert result.similarity_score < 0.35
    assert result.discrepancy_detected is True


@pytest.mark.anyio
async def test_semantic_evaluator_empty_text_flags_discrepancy():
    """Verify empty candidate or champion output short-circuits cleanly to 0.0 and flags discrepancy."""
    engine = EmbeddingEngine(dimension=3, encoder_fn=lambda texts: np.ones((len(texts), 3), dtype=np.float32))
    evaluator = SemanticEvaluator(embedding_engine=engine, threshold=0.80)

    res_empty_candidate = await evaluator.evaluate("Valid champion response", "   ")
    assert res_empty_candidate.similarity_score == 0.0
    assert res_empty_candidate.discrepancy_detected is True

    res_empty_champion = await evaluator.evaluate("", "Valid candidate response")
    assert res_empty_champion.similarity_score == 0.0
    assert res_empty_champion.discrepancy_detected is True
