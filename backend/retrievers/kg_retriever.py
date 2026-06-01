"""
retrievers/kg_retriever.py — Knowledge Graph Retriever
======================================================
Searches the NetworkX knowledge graph for related entities.
Wraps the existing KnowledgeGraph with the standard retriever interface.
"""

import logging
from typing import List, Dict, Any, Optional

from config import get_settings

logger = logging.getLogger(__name__)


class KGRetriever:
    """Knowledge Graph entity/relationship retriever."""

    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search Knowledge Graph for related entities.

        Returns list of dicts matching the retriever interface.
        """
        if not self.kg:
            return []

        top_k = top_k or self.settings.top_k_kg

        try:
            results = self.kg.search_relevant(query, top_k=top_k)
            return [
                {
                    "title": r.get("entity", ""),
                    "text": r.get("context", ""),
                    "source_url": r.get("attributes", {}).get("source_url", ""),
                    "source_type": "Knowledge Graph",
                    "score": r.get("score", 0) / max(1, max(
                        (x.get("score", 0) for x in results), default=1
                    )),  # normalize to 0-1
                    "similarity": 0.5,  # KG scores aren't directly comparable
                    "year": None,
                    "department": r.get("attributes", {}).get("department", "General"),
                    "doc_type": r.get("attributes", {}).get("entity_type", "General"),
                    "crawl_date": r.get("attributes", {}).get("updated_at", ""),
                }
                for r in results
                if r.get("context")
            ]
        except Exception as e:
            logger.error(f"KG retrieval error: {e}")
            return []
