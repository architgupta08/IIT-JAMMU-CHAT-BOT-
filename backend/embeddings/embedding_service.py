"""
embeddings/embedding_service.py — Shared Embedding Service
===========================================================
Centralizes embedding generation with LRU caching to avoid
redundant model calls. Used by ChromaDB store, BM25 indexer,
and autocomplete builder.
"""

import logging
from functools import lru_cache
from typing import List, Optional

from config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Shared embedding model with caching."""

    def __init__(self):
        settings = get_settings()
        self.model_name = settings.embedding_model
        self._model = None
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")

    def _load_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encode a list of texts into embedding vectors."""
        model = self._load_model()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text string."""
        return self.encode([text])[0]


# ── Singleton ─────────────────────────────────────────────────────
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the singleton EmbeddingService."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
