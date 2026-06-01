"""
vectorstore/chroma_store.py — Enhanced ChromaDB Vector Store
=============================================================
Persistent ChromaDB collection with rich metadata, recency scoring,
and metadata-based filtering. Drop-in replacement for the old chroma_store.py.
"""

import os
import hashlib
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from config import get_settings
from utils.text import build_document_metadata

logger = logging.getLogger(__name__)


def _resolve_chroma_path() -> str:
    """Resolve ChromaDB path relative to project root."""
    settings = get_settings()
    path = settings.chroma_db_path
    if os.path.isabs(path):
        os.makedirs(path, exist_ok=True)
        return path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(backend_dir))
    from_root = os.path.join(project_root, path)
    os.makedirs(from_root, exist_ok=True)
    return from_root


@dataclass
class ChromaSearchResult:
    """A single search result from ChromaDB."""
    doc_id: str
    text: str
    title: str
    source_url: str
    topic: str
    score: float          # distance (lower = more similar)
    similarity: float     # normalized similarity (higher = better)
    year: Optional[int]
    department: str
    doc_type: str
    crawl_date: str
    metadata: Dict[str, Any]


class ChromaStore:
    """
    Persistent ChromaDB vector store with rich metadata.

    Enhancements over original:
      - Rich metadata: year, department, doc_type, crawl_date
      - Metadata filtering support
      - Normalized similarity scores
      - Batch upsert with deduplication
    """

    COLLECTION_NAME = "iitj_knowledge"

    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions

        settings = get_settings()
        self._db_path = _resolve_chroma_path()
        logger.info(f"ChromaDB path: {self._db_path}")

        self._client = chromadb.PersistentClient(path=self._db_path)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"description": "IIT Jammu knowledge base for RAG"}
        )
        logger.info(
            f"ChromaDB ready — collection '{self.COLLECTION_NAME}' "
            f"has {self._collection.count()} documents"
        )

    @staticmethod
    def _content_hash(text: str) -> str:
        """SHA-256 hash of text content for deduplication."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _hash_exists(self, content_hash: str) -> bool:
        """Check if a document with this content hash already exists."""
        try:
            results = self._collection.get(
                where={"content_hash": content_hash},
                limit=1
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    def add_documents(
        self,
        documents: List[Dict[str, str]],
        batch_size: int = 50
    ) -> int:
        """
        Add documents with rich metadata to ChromaDB. Skips duplicates.

        Each document dict should have:
          - text: str (required)
          - title: str
          - source_url: str
          - topic: str
          - year: str (optional)
          - department: str (optional)
          - doc_type: str (optional)

        Returns: number of new documents actually inserted.
        """
        inserted = 0
        ids_batch = []
        texts_batch = []
        metas_batch = []
        seen_ids = set()

        for doc in documents:
            text = doc.get("text", "").strip()
            if not text or len(text) < 50:
                continue

            content_hash = self._content_hash(text)
            doc_id = f"doc_{content_hash[:16]}"
            if doc_id in seen_ids:
                continue

            if self._hash_exists(content_hash):
                continue

            seen_ids.add(doc_id)
            title = doc.get("title", "Untitled")
            source_url = doc.get("source_url", "")

            # Build rich metadata
            metadata = build_document_metadata(title, text, source_url)
            metadata["topic"] = doc.get("topic", metadata.get("doc_type", "General"))[:100]
            metadata["content_hash"] = content_hash

            # Override with explicitly provided metadata
            if doc.get("year"):
                metadata["year"] = str(doc["year"])
            if doc.get("department"):
                metadata["department"] = doc["department"]
            if doc.get("doc_type"):
                metadata["doc_type"] = doc["doc_type"]
            if doc.get("document_type"):
                metadata["document_type"] = doc["document_type"]
            if doc.get("last_updated"):
                metadata["last_updated"] = doc["last_updated"]
            if doc.get("target_audience"):
                metadata["target_audience"] = doc["target_audience"]

            ids_batch.append(doc_id)
            texts_batch.append(text)
            metas_batch.append(metadata)

            if len(ids_batch) >= batch_size:
                try:
                    self._collection.add(
                        ids=ids_batch,
                        documents=texts_batch,
                        metadatas=metas_batch
                    )
                    inserted += len(ids_batch)
                except Exception as e:
                    logger.error(f"ChromaDB batch insert error: {e}")
                ids_batch, texts_batch, metas_batch = [], [], []

        # Flush remaining
        if ids_batch:
            try:
                self._collection.add(
                    ids=ids_batch,
                    documents=texts_batch,
                    metadatas=metas_batch
                )
                inserted += len(ids_batch)
            except Exception as e:
                logger.error(f"ChromaDB final batch insert error: {e}")


        if inserted > 0:
            logger.info(
                f"ChromaDB: inserted {inserted} new documents "
                f"(skipped {len(documents) - inserted} duplicates)"
            )
        return inserted

    def search(
        self,
        query: str,
        top_k: int = 8,
        where_filter: Optional[Dict] = None,
    ) -> List[ChromaSearchResult]:
        """
        Semantic search with optional metadata filtering.

        Args:
            query: Search query text
            top_k: Number of results to return
            where_filter: ChromaDB where clause, e.g. {"year": "2025"}

        Returns: List of ChromaSearchResult ordered by relevance.
        """
        if not query.strip():
            return []

        try:
            n_results = min(top_k, max(1, self._collection.count()))
            query_params = {
                "query_texts": [query],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_params["where"] = where_filter

            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return []

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 1.0
                # Normalize distance to similarity (0-1, higher = better)
                similarity = max(0.0, 1.0 - (dist / 2.0))

                year = None
                try:
                    year = int(meta.get("year", 0))
                    if year < 2000:
                        year = None
                except (ValueError, TypeError):
                    pass

                search_results.append(ChromaSearchResult(
                    doc_id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    title=meta.get("title", ""),
                    source_url=meta.get("source_url", ""),
                    topic=meta.get("topic", ""),
                    score=dist,
                    similarity=similarity,
                    year=year,
                    department=meta.get("department", "General"),
                    doc_type=meta.get("doc_type", "General"),
                    crawl_date=meta.get("crawl_date", ""),
                    metadata=meta,
                ))

        return search_results

    def count(self) -> int:
        """Return total number of documents in the collection."""
        return self._collection.count()

    def get_all_texts(self) -> List[str]:
        """Retrieve all document texts (used for BM25 index building)."""
        try:
            count = self._collection.count()
            if count == 0:
                return []
            result = self._collection.get(
                limit=count,
                include=["documents"]
            )
            return result.get("documents", [])
        except Exception as e:
            logger.error(f"Failed to get all texts: {e}")
            return []

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Retrieve all document metadata (used for autocomplete index)."""
        try:
            count = self._collection.count()
            if count == 0:
                return []
            result = self._collection.get(
                limit=count,
                include=["metadatas"]
            )
            return result.get("metadatas", [])
        except Exception as e:
            logger.error(f"Failed to get all metadata: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        settings = get_settings()
        return {
            "total_documents": self._collection.count(),
            "collection_name": self.COLLECTION_NAME,
            "db_path": self._db_path,
            "embedding_model": settings.embedding_model,
        }


# ── Singleton ─────────────────────────────────────────────────────
_store: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """Get or create the singleton ChromaStore instance."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
