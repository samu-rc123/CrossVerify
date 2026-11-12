import numpy as np
import pytest

from worker.evaluator.semantic import EmbeddingEngine


@pytest.fixture(scope="module")
def real_embedding_engine():
    """Shared local SentenceTransformer engine across embedding tests to avoid reloading weights."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.mark.anyio
async def test_embedding_engine_real_model_single_and_batch(real_embedding_engine):
    """Verify real local all-MiniLM-L6-v2 embedding generation produces unit-norm 384-d vectors."""
    await real_embedding_engine.warm_up()

    single_vec = await real_embedding_engine.embed_text("Verify distributed system reliability.")
    assert single_vec.shape == (384,)
    assert single_vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(single_vec), 1.0, atol=1e-4)

    batch_vecs = await real_embedding_engine.embed_batch([
        "First sentence for semantic embedding.",
        "Second sentence with different meaning."
    ])
    assert batch_vecs.shape == (2, 384)
    assert batch_vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(batch_vecs, axis=1), 1.0, atol=1e-4)


@pytest.mark.anyio
async def test_embedding_engine_empty_and_mixed_batch(real_embedding_engine):
    """Verify edge cases: empty input list and whitespace strings produce safe zero vectors."""
    empty_batch = await real_embedding_engine.embed_batch([])
    assert empty_batch.shape == (0, 384)

    mixed_batch = await real_embedding_engine.embed_batch(["   ", "Non-empty prompt response", ""])
    assert mixed_batch.shape == (3, 384)
    assert np.allclose(mixed_batch[0], 0.0)
    assert np.isclose(np.linalg.norm(mixed_batch[1]), 1.0, atol=1e-4)
    assert np.allclose(mixed_batch[2], 0.0)


@pytest.mark.anyio
async def test_embedding_engine_custom_encoder_normalization():
    """Verify custom encoder output is automatically L2-normalized to float32."""
    def fake_encoder(texts):
        return np.array([[3.0, 4.0, 0.0] for _ in texts], dtype=np.float32)

    engine = EmbeddingEngine(dimension=3, encoder_fn=fake_encoder)
    vec = await engine.embed_text("unit test")

    assert vec.shape == (3,)
    assert np.allclose(vec, [0.6, 0.8, 0.0], atol=1e-5)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
