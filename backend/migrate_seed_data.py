"""
migrate_seed_data.py — Migrate existing seed index to ChromaDB + KG
=====================================================================
One-time migration script that reads the existing iitj_index.json
(2,147 nodes) and inserts all content into ChromaDB and the Knowledge Graph.

Safe to run multiple times — SHA-256 dedup prevents duplicates.

USAGE:
  cd backend
  python migrate_seed_data.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Resolve index path ─────────────────────────────────────────────

def _find_index_file() -> str:
    """Find iitj_index.json from multiple possible locations."""
    candidates = [
        os.getenv("INDEX_FILE", "data/processed/iitj_index.json"),
        "data/processed/iitj_index.json",
        "../data/processed/iitj_index.json",
    ]

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)

    for c in candidates:
        # Try as-is
        if os.path.exists(c):
            return c
        # Try from project root
        from_root = os.path.join(project_root, c.lstrip("../"))
        if os.path.exists(from_root):
            return from_root
        # Try from backend dir
        from_backend = os.path.join(backend_dir, c)
        if os.path.exists(from_backend):
            return from_backend

    raise FileNotFoundError(
        "Cannot find iitj_index.json. Tried: " + str(candidates)
    )


def _flatten_tree(nodes: list, parent_path: str = "") -> List[Dict]:
    """Recursively flatten the tree into a list of documents."""
    docs = []
    for node in nodes:
        title = node.get("title", "Untitled")
        path = f"{parent_path} > {title}" if parent_path else title
        text = node.get("text", "")
        summary = node.get("summary", "")
        source = node.get("source", "")

        # Combine summary and text for richer content
        combined = ""
        if summary:
            combined += summary + "\n"
        if text:
            combined += text

        if combined.strip() and len(combined.strip()) >= 50:
            docs.append({
                "title": title,
                "text": combined.strip(),
                "source_url": source if source.startswith("http") else f"seed:{source}",
                "topic": _infer_topic(path, title),
                "path": path,
            })

        # Recurse into children
        children = node.get("nodes", [])
        if children:
            docs.extend(_flatten_tree(children, path))

    return docs


def _infer_topic(path: str, title: str) -> str:
    """Infer topic from path and title."""
    combined = f"{path} {title}".lower()

    if any(w in combined for w in ["fee", "charge", "tuition", "hostel charge"]):
        return "Fee Structure"
    if any(w in combined for w in ["placement", "ctc", "lpa", "recruit"]):
        return "Placements"
    if any(w in combined for w in ["admission", "jee", "gate", "josaa"]):
        return "Admissions"
    if any(w in combined for w in ["scholarship", "mcm", "pmrf", "fellowship"]):
        return "Scholarships"
    if any(w in combined for w in ["department", "cse", "electrical", "mechanical"]):
        return "Departments"
    if any(w in combined for w in ["faculty", "professor"]):
        return "Faculty"
    if any(w in combined for w in ["research", "lab", "publication"]):
        return "Research"
    if any(w in combined for w in ["campus", "hostel", "library", "sports", "medical"]):
        return "Campus & Facilities"
    if any(w in combined for w in ["program", "btech", "mtech", "phd", "course"]):
        return "Academic Programs"
    if any(w in combined for w in ["contact", "address", "phone", "email"]):
        return "Contact"
    if any(w in combined for w in ["about", "director", "history", "vision"]):
        return "About IIT Jammu"

    return "General"


def _chunk_documents(docs: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Chunk long documents into smaller pieces for better retrieval."""
    chunked = []
    for doc in docs:
        text = doc["text"]
        if len(text) <= chunk_size:
            chunked.append(doc)
            continue

        # Split into chunks
        start = 0
        idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= 50:
                idx += 1
                chunked.append({
                    "title": f"{doc['title']} (part {idx})",
                    "text": chunk,
                    "source_url": doc.get("source_url", ""),
                    "topic": doc.get("topic", "General"),
                })
            start = end - overlap

    return chunked


def migrate():
    """Main migration function."""
    logger.info("=" * 60)
    logger.info("  IIT Jammu Seed Data Migration")
    logger.info("  iitj_index.json → ChromaDB + Knowledge Graph")
    logger.info("=" * 60)

    # 1. Find and load the index
    index_path = _find_index_file()
    logger.info(f"\n📂 Loading index from: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    total_nodes = index_data.get("total_nodes", "unknown")
    logger.info(f"   Index has {total_nodes} declared nodes")

    # 2. Flatten tree into documents
    tree = index_data.get("structure", [])
    docs = _flatten_tree(tree)
    logger.info(f"   Flattened to {len(docs)} documents with content")

    # 3. Chunk documents
    chunked_docs = _chunk_documents(docs)
    logger.info(f"   After chunking: {len(chunked_docs)} chunks")

    # 4. Initialize ChromaDB
    logger.info("\n📦 Initializing ChromaDB...")
    from chroma_store import get_chroma_store
    chroma = get_chroma_store()
    initial_count = chroma.count()
    logger.info(f"   ChromaDB has {initial_count} existing documents")

    # 5. Insert into ChromaDB
    logger.info("\n📥 Inserting into ChromaDB...")
    inserted = chroma.add_documents(chunked_docs)
    final_count = chroma.count()
    logger.info(f"   Inserted: {inserted} new documents")
    logger.info(f"   ChromaDB total: {final_count} documents")

    # 6. Initialize Knowledge Graph
    logger.info("\n🕸️  Initializing Knowledge Graph...")
    from knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    initial_nodes = kg.node_count()
    logger.info(f"   KG has {initial_nodes} existing nodes")

    # 7. Extract entities and add to KG
    logger.info("\n🔍 Extracting entities into Knowledge Graph...")
    for doc in docs:  # Use original (non-chunked) docs for better entity extraction
        kg.extract_and_add_from_text(
            text=doc["text"],
            title=doc["title"],
            source_url=doc.get("source_url", ""),
        )

    kg.save()
    final_nodes = kg.node_count()
    final_edges = kg.edge_count()
    logger.info(f"   KG nodes: {initial_nodes} → {final_nodes} (+{final_nodes - initial_nodes})")
    logger.info(f"   KG edges: {final_edges}")

    # 8. Summary
    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Migration complete!")
    logger.info(f"  ChromaDB: {final_count} documents")
    logger.info(f"  Knowledge Graph: {final_nodes} nodes, {final_edges} edges")
    logger.info("=" * 60)

    return {
        "chroma_docs": final_count,
        "kg_nodes": final_nodes,
        "kg_edges": final_edges,
        "new_docs_inserted": inserted,
    }


if __name__ == "__main__":
    migrate()
