"""
Embedding Manager powered by sentence-transformers.
Computes dense vector representations with L2 normalization for exact cosine similarity.
"""

import logging
from typing import List, Union
import numpy as np

from core.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages sentence embedding generation with automatic normalization."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self.model = None

        self._load_model()

    def _load_model(self):
        """Attempts to load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = self.model.get_embedding_dimension()
            else:
                self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info("SentenceTransformer loaded successfully (dim=%d).", self.dimension)
        except Exception as e:
            logger.warning(
                "Could not load SentenceTransformer '%s': %s. Using high-dimensional hash fallback.",
                self.model_name,
                e,
            )
            self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts and returns L2-normalized float32 numpy array [N, dim].
        L2 normalization ensures inner product equals cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.error("SentenceTransformer inference failed: %s. Falling back.", e)

        # Fallback deterministic pseudo-dense semantic projection
        return self._fallback_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string [1, dim]."""
        return self.embed_texts([query])

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Deterministic fallback embedding for network-isolated environments."""
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h = hash(w) % self.dimension
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


# Cached singleton instance
_EMBEDDING_INSTANCE = None


def get_embedding_manager() -> EmbeddingManager:
    """Returns the shared EmbeddingManager singleton."""
    global _EMBEDDING_INSTANCE
    if _EMBEDDING_INSTANCE is None:
        _EMBEDDING_INSTANCE = EmbeddingManager()
    return _EMBEDDING_INSTANCE
