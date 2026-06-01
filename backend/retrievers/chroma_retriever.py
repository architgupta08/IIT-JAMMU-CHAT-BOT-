"""
retrievers/chroma_retriever.py — ChromaDB Semantic Retriever
============================================================
Wraps ChromaStore with recency-weighted scoring for the RAG engine.
"""

import logging
from typing import List, Dict, Any, Optional

from config import get_settings
from utils.scoring import compute_recency_score, compute_final_score, normalize_chroma_distance

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """Semantic retriever powered by ChromaDB with recency scoring."""

    def __init__(self, chroma_store):
        self.store = chroma_store
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar documents with recency-weighted scores.

        Returns list of dicts with keys:
            title, text, source_url, source_type, score, similarity,
            year, department, doc_type, crawl_date
        """
        if not self.store:
            return []

        top_k = top_k or self.settings.top_k_chroma

        try:
            results = self.store.search(query, top_k=top_k, where_filter=where_filter)
        except Exception as e:
            logger.error(f"ChromaDB retrieval error: {e}")
            return []

        scored_results = []
        for r in results:
            recency = compute_recency_score(r.year)
            final = compute_final_score(
                semantic_score=r.similarity,
                recency_score=recency,
            )

            scored_results.append({
                "title": r.title,
                "text": r.text,
                "source_url": r.source_url,
                "source_type": "Vector DB",
                "score": final,
                "similarity": r.similarity,
                "recency_score": recency,
                "year": r.year,
                "department": r.department,
                "doc_type": r.doc_type,
                "crawl_date": r.crawl_date,
            })

        # Sort by final score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results
