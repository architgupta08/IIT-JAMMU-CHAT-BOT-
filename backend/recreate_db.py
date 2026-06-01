"""
backend/recreate_db.py — Recreate and populate ChromaDB and Knowledge Graph.
=============================================================================
This script clears the ChromaDB 'iitj_knowledge' collection and resets the 
'knowledge_graph.graphml' file, then executes the complete ingestion pipeline:
1. Seeds canonical HOD relationships to the Knowledge Graph.
2. Ingests raw crawled markdown files into ChromaDB and auto-builds KG relationships.
3. Ingests curated IIT Jammu FAQs.
4. Ingests LLM-extracted FAQ pairs from auto_generated_faqs.json.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recreate_db")

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def clear_chromadb(store):
    """Deletes and recreates the ChromaDB collection."""
    logger.info("Clearing ChromaDB collection...")
    try:
        store._client.delete_collection(store.COLLECTION_NAME)
        logger.info(f"Collection '{store.COLLECTION_NAME}' deleted successfully.")
    except Exception as e:
        logger.warning(f"Failed to delete collection (might not exist): {e}")

    # Recreate the collection
    store._collection = store._client.get_or_create_collection(
        name=store.COLLECTION_NAME,
        embedding_function=store._embed_fn,
        metadata={"description": "IIT Jammu knowledge base for RAG"}
    )
    logger.info(f"Collection '{store.COLLECTION_NAME}' recreated.")

def clear_knowledge_graph():
    """Deletes the existing GraphML file if it exists."""
    logger.info("Resetting Knowledge Graph file...")
    try:
        from knowledge_graph import _resolve_kg_path
        kg_path = _resolve_kg_path()
        if os.path.exists(kg_path):
            os.remove(kg_path)
            logger.info(f"Deleted existing Knowledge Graph file at: {kg_path}")
        else:
            logger.info("No existing Knowledge Graph file found to delete.")
    except Exception as e:
        logger.error(f"Error resetting Knowledge Graph file: {e}")

def inject_auto_generated_faqs(store):
    """Reads auto_generated_faqs.json, flattens, and injects FAQs to ChromaDB."""
    logger.info("Injecting automated LLM-extracted FAQs...")
    project_root = Path(backend_dir).parent
    faq_file = project_root / "data" / "processed" / "auto_generated_faqs.json"
    
    if not faq_file.exists():
        logger.warning(f"No auto-generated FAQs file found at {faq_file}. Skipping this step.")
        return

    try:
        state = json.loads(faq_file.read_text(encoding="utf-8"))
        all_faqs = []
        for filename, faqs in state.items():
            if isinstance(faqs, list):
                all_faqs.extend(faqs)
        
        logger.info(f"Found {len(all_faqs)} auto-generated FAQs. Preparing for ChromaDB injection...")
        
        docs = []
        for idx, faq in enumerate(all_faqs, 1):
            question = faq.get("q", "").strip()
            answer = faq.get("a", "").strip()
            if not question or not answer:
                continue
                
            text = (
                f"IIT Jammu FAQ\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Department: {faq.get('department', 'General')}\n"
            )
            docs.append({
                "text": text,
                "title": f"Auto FAQ {idx:04d}: {question[:90]}",
                "topic": "FAQ",
                "source_url": faq.get("source_url", "https://www.iitjammu.ac.in"),
                "department": faq.get("department", "General"),
                "doc_type": "faq",
                "document_type": "faq",
                "last_updated": "2026-05-22",
                "target_audience": "General",
                "year": "2026",
            })
            
        inserted = store.add_documents(docs, batch_size=100)
        logger.info(f"Successfully injected {inserted} auto-generated FAQs into ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to inject auto-generated FAQs: {e}")

def main():
    print("=" * 70)
    print("  IIT JAMMU DATABASE RECREATION PIPELINE")
    print("=" * 70)

    # 1. Initialize vector store
    from vectorstore.chroma_store import get_chroma_store
    store = get_chroma_store()

    # 2. Reset database and knowledge graph
    clear_chromadb(store)
    clear_knowledge_graph()

    # 3. Seed HOD relationships to KG first (starting fresh)
    logger.info("Step 1: Seeding canonical HOD relationships to Knowledge Graph...")
    from knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    kg.seed_hods()
    kg.save()
    logger.info(f"KG seeded: {kg.node_count()} nodes, {kg.edge_count()} edges.")

    # 4. Ingest raw crawled pages (this runs ingest_raw_md.py main)
    logger.info("Step 2: Ingesting raw crawled Markdown files...")
    import ingest_raw_md
    ingest_raw_md.main()

    # 4.5. Ingest detailed faculty profiles from API (this runs ingest_faculty_api.py main)
    logger.info("Step 2.5: Ingesting structured faculty API profiles...")
    import ingest_faculty_api
    ingest_faculty_api.main()

    # 5. Inject curated FAQs (this runs inject_curated_iitj_faqs.py main)
    logger.info("Step 3: Injecting curated FAQs...")
    import inject_curated_iitj_faqs
    inject_curated_iitj_faqs.main()

    # 6. Inject automated LLM FAQs
    logger.info("Step 4: Injecting LLM-extracted FAQs...")
    inject_auto_generated_faqs(store)

    print("\n" + "=" * 70)
    print("  DATABASE AND KNOWLEDGE GRAPH RECREATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
