"""
chroma_store.py — ChromaDB Vector Store for IIT Jammu Chatbot
==============================================================
Manages a persistent ChromaDB collection for semantic document search.

Key design:
  - SHA-256 content dedup: never inserts duplicate chunks
  - Append-only: never deletes existing documents
  - Persistent on disk: survives server restarts
  - Uses sentence-transformers for local embeddings (no API key)
"""

import os
import hashlib
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/processed/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _resolve_chroma_path() -> str:
    """Resolve ChromaDB path relative to project root."""
    path = CHROMA_DB_PATH
    if path.startswith("../"):
        path = path[3:]

    # Try as-is
    if os.path.isabs(path):
        return path

    # Try relative to backend/
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    from_backend = os.path.join(backend_dir, path)

    # Try from project root
    project_root = os.path.dirname(backend_dir)
    from_root = os.path.join(project_root, path)

    # Prefer project root location
    if os.path.exists(from_root) or not os.path.exists(from_backend):
        os.makedirs(from_root, exist_ok=True)
        return from_root

    os.makedirs(from_backend, exist_ok=True)
    return from_backend


@dataclass
class ChromaSearchResult:
    """A single search result from ChromaDB."""
    doc_id: str
    text: str
    title: str
    source_url: str
    topic: str
    score: float  # distance (lower = more similar)
    metadata: Dict[str, Any]


class ChromaStore:
    """
    Persistent ChromaDB vector store for IIT Jammu knowledge.

    Features:
      - Automatic embedding via sentence-transformers
      - SHA-256 dedup prevents duplicate insertions
      - Append-only — never deletes existing documents
      - Persistent storage on disk
    """

    COLLECTION_NAME = "iitj_knowledge"

    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions

        self._db_path = _resolve_chroma_path()
        logger.info(f"ChromaDB path: {self._db_path}")

        # Create persistent client
        self._client = chromadb.PersistentClient(path=self._db_path)

        # Set up embedding function
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Get or create collection
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
        Add documents to ChromaDB. Skips duplicates by content hash.

        Each document dict should have:
          - text: str (required)
          - title: str
          - source_url: str
          - topic: str

        Returns: number of new documents actually inserted.
        """
        inserted = 0
        ids_batch = []
        texts_batch = []
        metas_batch = []

        for doc in documents:
            text = doc.get("text", "").strip()
            if not text or len(text) < 50:
                continue

            content_hash = self._content_hash(text)

            # Skip if already exists
            if self._hash_exists(content_hash):
                continue

            doc_id = f"doc_{content_hash[:16]}"
            metadata = {
                "title": doc.get("title", "Untitled")[:200],
                "source_url": doc.get("source_url", "")[:500],
                "topic": doc.get("topic", "General")[:100],
                "content_hash": content_hash,
                "scraped_at": datetime.utcnow().isoformat(),
                "char_count": len(text),
            }

            ids_batch.append(doc_id)
            texts_batch.append(text)
            metas_batch.append(metadata)

            # Flush batch
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
            logger.info(f"ChromaDB: inserted {inserted} new documents (skipped {len(documents) - inserted} duplicates)")

        return inserted

    def search(self, query: str, top_k: int = 5) -> List[ChromaSearchResult]:
        """
        Semantic search for documents matching the query.
        Returns top_k results ordered by relevance (lowest distance = best match).
        """
        if not query.strip():
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return []

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 1.0
                search_results.append(ChromaSearchResult(
                    doc_id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    title=meta.get("title", ""),
                    source_url=meta.get("source_url", ""),
                    topic=meta.get("topic", ""),
                    score=dist,
                    metadata=meta,
                ))

        return search_results

    def count(self) -> int:
        """Return total number of documents in the collection."""
        return self._collection.count()

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        return {
            "total_documents": self._collection.count(),
            "collection_name": self.COLLECTION_NAME,
            "db_path": self._db_path,
            "embedding_model": EMBEDDING_MODEL,
        }


# ── Singleton ─────────────────────────────────────────────────────
_store: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """Get or create the singleton ChromaStore instance."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
