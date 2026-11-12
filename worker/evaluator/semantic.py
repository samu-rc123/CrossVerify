import asyncio
import logging
import threading
from typing import List, Optional, Callable

import numpy as np

from worker.core.config import get_worker_settings

logger = logging.getLogger("crossverify.worker.evaluator.semantic")


class EmbeddingEngine:
    """
    Local dense vector embedding generator backed by HuggingFace SentenceTransformers.
    Default model: 'all-MiniLM-L6-v2' (384 dimensions, fast CPU/GPU inference).
    Uses lazy thread-safe initialization and offloads forward passes to worker threads
    so the asyncio Redis consumer loop remains non-blocking.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: int = 384,
        encoder_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        self.settings = get_worker_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension
        self._encoder_fn = encoder_fn
        self._model = None
        self._lock = threading.Lock()

    def _get_or_load_model(self):
        """Thread-safe lazy loader for SentenceTransformer model weights."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info(f"[EmbeddingEngine] Loading SentenceTransformer model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
                if hasattr(self._model, "get_embedding_dimension"):
                    detected_dim = self._model.get_embedding_dimension()
                else:
                    detected_dim = self._model.get_sentence_embedding_dimension()
                if detected_dim:
                    self.dimension = int(detected_dim)
                logger.info(
                    f"[EmbeddingEngine] Model '{self.model_name}' ready (dim={self.dimension})."
                )
        return self._model

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize embedding rows safely, preserving zero vectors without NaN."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        normalized = vectors / safe_norms
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def _encode_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous embedding encoding executed inside a worker thread."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if self._encoder_fn is not None:
            raw = self._encoder_fn(texts)
            return self._normalize_rows(raw)

        # Preserve zero vectors for empty/whitespace-only texts
        non_empty_indices = [idx for idx, text in enumerate(texts) if text and text.strip()]
        if not non_empty_indices:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

        model = self._get_or_load_model()
        non_empty_texts = [texts[idx] for idx in non_empty_indices]
        encoded = model.encode(
            non_empty_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        encoded = self._normalize_rows(encoded)

        if len(non_empty_indices) == len(texts):
            return encoded

        output = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
        for out_pos, orig_idx in enumerate(non_empty_indices):
            output[orig_idx] = encoded[out_pos]
        return output

    async def warm_up(self) -> None:
        """Pre-load the embedding model in a background thread at worker startup."""
        if self._encoder_fn is None:
            await asyncio.to_thread(self._get_or_load_model)

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Asynchronously encode a batch of strings into L2-normalized dense float32 vectors."""
        return await asyncio.to_thread(self._encode_sync, texts)

    async def embed_text(self, text: str) -> np.ndarray:
        """Asynchronously encode a single string into a 1D L2-normalized float32 vector."""
        batch = await self.embed_batch([text])
        return batch[0]
