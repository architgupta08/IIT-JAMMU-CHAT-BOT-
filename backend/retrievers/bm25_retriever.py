"""
retrievers/bm25_retriever.py — BM25 Keyword Retriever
======================================================
Classic BM25 (Okapi BM25) keyword search over the document corpus.
Complements ChromaDB's semantic search by catching exact keyword
matches that embeddings may miss (e.g. specific professor names,
abbreviations, course codes).

The BM25 index is built lazily from ChromaDB documents and
rebuilt when the scraper adds new content.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from config import get_settings

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 keyword retriever over the document corpus."""

    def __init__(self, chroma_store=None):
        self.store = chroma_store
        self.settings = get_settings()
        self._bm25 = None
        self._corpus_docs: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._is_built = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r"\b\w{2,}\b", text.lower())

    def build_index(self):
        """Build or rebuild the BM25 index from ChromaDB documents."""
        if not self.store:
            logger.warning("BM25: No ChromaDB store available, skipping index build")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. Run: pip install rank-bm25")
            return

        try:
            count = self.store.count()
            if count == 0:
                logger.info("BM25: ChromaDB is empty, nothing to index")
                return

            # Fetch all documents with metadata
            result = self.store._collection.get(
                limit=min(count, 10000),
                include=["documents", "metadatas"]
            )

            self._corpus_docs = []
            self._tokenized_corpus = []

            docs = result.get("documents", [])
            metas = result.get("metadatas", [])

            for i, text in enumerate(docs):
                if not text or len(text.strip()) < 50:
                    continue
                meta = metas[i] if i < len(metas) else {}
                self._corpus_docs.append({
                    "text": text,
                    "title": meta.get("title", ""),
                    "source_url": meta.get("source_url", ""),
                    "topic": meta.get("topic", ""),
                    "year": meta.get("year"),
                    "department": meta.get("department", "General"),
                    "doc_type": meta.get("doc_type", "General"),
                    "crawl_date": meta.get("crawl_date", ""),
                })
                tokens = self._tokenize(f"{meta.get('title', '')} {text}")
                self._tokenized_corpus.append(tokens)

            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(
                    self._tokenized_corpus,
                    k1=self.settings.bm25_k1,
                    b=self.settings.bm25_b,
                )
                self._is_built = True
                logger.info(f"BM25 index built: {len(self._corpus_docs)} documents")
            else:
                logger.warning("BM25: No valid documents to index")

        except Exception as e:
            logger.error(f"BM25 index build error: {e}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents by BM25 keyword matching.

        Returns list of dicts matching the retriever interface.
        """
        if not self._is_built:
            self.build_index()

        if not self._bm25 or not self._corpus_docs:
            return []

        top_k = top_k or self.settings.top_k_bm25

        try:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            scores = self._bm25.get_scores(query_tokens)

            # Get top-k indices
            scored_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            results = []
            max_score = max(scores) if max(scores) > 0 else 1.0

            for idx in scored_indices:
                if scores[idx] <= 0:
                    continue
                doc = self._corpus_docs[idx]
                normalized_score = scores[idx] / max_score  # 0-1 range

                results.append({
                    "title": doc["title"],
                    "text": doc["text"],
                    "source_url": doc.get("source_url", ""),
                    "source_type": "BM25 Keyword",
                    "score": normalized_score,
                    "similarity": normalized_score,
                    "year": int(doc["year"]) if doc.get("year") and str(doc["year"]).isdigit() else None,
                    "department": doc.get("department", "General"),
                    "doc_type": doc.get("doc_type", "General"),
                    "crawl_date": doc.get("crawl_date", ""),
                })

            return results

        except Exception as e:
            logger.error(f"BM25 retrieval error: {e}")
            return []
