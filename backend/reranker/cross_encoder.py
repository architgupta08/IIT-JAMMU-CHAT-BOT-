"""
reranker/cross_encoder.py — Cross-Encoder Reranking
====================================================
Re-ranks retrieved documents using a cross-encoder model for
more accurate relevance scoring. The cross-encoder sees both
the query and document together, producing better scores than
bi-encoder (embedding) similarity alone.

Uses: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
"""

import logging
from typing import List, Dict, Any, Optional

from config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder based document reranker."""

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._enabled = self.settings.reranker_enabled

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None and self._enabled:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.settings.reranker_model)
                logger.info(f"Cross-encoder loaded: {self.settings.reranker_model}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed for cross-encoder. "
                    "Reranking disabled."
                )
                self._enabled = False
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}. Reranking disabled.")
                self._enabled = False

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using the cross-encoder.

        Args:
            query: The user's search query
            documents: List of retrieved document dicts (must have 'text' key)
            top_k: Number of top documents to return after reranking

        Returns:
            Re-ranked list of documents with updated 'score' values
        """
        if not self._enabled or not documents:
            return documents[:top_k] if top_k else documents

        self._load_model()
        if not self._model:
            return documents[:top_k] if top_k else documents

        top_k = top_k or self.settings.reranker_top_k

        try:
            # Create query-document pairs for the cross-encoder
            pairs = []
            for doc in documents:
                text = doc.get("text", "")[:2048]  # Truncate to ~500 words (fits 512 token limit)
                pairs.append([query, text])

            # Score all pairs
            scores = self._model.predict(pairs)

            # Attach scores and sort
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])

            # Sort by rerank score (higher = more relevant)
            reranked = sorted(documents, key=lambda d: d.get("rerank_score", 0), reverse=True)

            # Normalize rerank scores to 0-1
            if reranked:
                max_score = max(d.get("rerank_score", 0) for d in reranked)
                min_score = min(d.get("rerank_score", 0) for d in reranked)
                score_range = max_score - min_score if max_score != min_score else 1.0
                for d in reranked:
                    d["score"] = (d.get("rerank_score", 0) - min_score) / score_range

            logger.debug(
                f"Reranked {len(documents)} docs → top {top_k}. "
                f"Best: {reranked[0].get('title', '')[:50] if reranked else 'N/A'}"
            )

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Cross-encoder reranking error: {e}")
            return documents[:top_k]


# ── Singleton ─────────────────────────────────────────────────────
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get or create the singleton CrossEncoderReranker."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
