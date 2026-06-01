import os
import json
import logging
from typing import List, Dict
from pydantic_settings import BaseSettings

# Simple logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("manage_db")

def get_chroma():
    """Helper to get ChromaDB collection safely."""
    try:
        from vectorstore.chroma_store import get_chroma_store
        return get_chroma_store()
    except Exception as e:
        logger.error(f"Failed to load ChromaDB. Are you in the backend folder? Error: {e}")
        return None

def purge_old_documents():
    """Deletes any document from ChromaDB that has 2021, 2022, or 2023 in its title."""
    chroma = get_chroma()
    if not chroma:
        return

    logger.info("Scanning ChromaDB for old documents (2021, 2022, 2023)...")
    try:
        # Get all documents
        all_docs = chroma._collection.get(include=["metadatas"])
        metadatas = all_docs.get("metadatas", [])
        ids = all_docs.get("ids", [])
        
        to_delete = []
        old_years = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]
        
        for i, meta in enumerate(metadatas):
            title = meta.get("title", "")
            if any(year in title for year in old_years):
                to_delete.append(ids[i])
                logger.info(f"Found old document: {title}")
                
        if to_delete:
            logger.info(f"Deleting {len(to_delete)} old chunks...")
            chroma._collection.delete(ids=to_delete)
            logger.info("✅ Successfully purged old documents!")
        else:
            logger.info("✅ No old documents found in the database.")
            
    except Exception as e:
        logger.error(f"Failed to purge documents: {e}")

def load_custom_faqs(json_path: str):
    """Loads a JSON file of Q&A pairs and adds them to ChromaDB as high-priority FAQs."""
    chroma = get_chroma()
    if not chroma:
        return
        
    if not os.path.exists(json_path):
        logger.error(f"FAQ file not found: {json_path}")
        logger.info("Please create a JSON file like this:")
        logger.info('[\n  {"q": "What is the BTech fee?", "a": "The fee is..."}\n]')
        return
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
            
        logger.info(f"Found {len(faqs)} FAQs in {json_path}. Ingesting into ChromaDB...")
        
        from utils.text import chunk_text
        
        total_added = 0
        for i, faq in enumerate(faqs):
            question = faq.get("q", faq.get("question", ""))
            answer = faq.get("a", faq.get("answer", ""))
            
            if not question or not answer:
                continue
                
            # Combine Q and A into a single searchable document
            text = f"Q: {question}\nA: {answer}"
            
            chunks = chunk_text(
                text=text,
                title=f"Custom FAQ: {question}",
                source_url="custom_faq",
                topic="FAQ",
            )
            
            # Modify source_type explicitly so RAG engine knows it's an FAQ
            for chunk in chunks:
                chunk["source_type"] = "FAQ"
                
            added = chroma.add_documents(chunks)
            total_added += added
            
        logger.info(f"✅ Successfully injected {total_added} FAQ chunks into the database!")
        
    except Exception as e:
        logger.error(f"Failed to load FAQs: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IIT Jammu Database Manager")
    parser.add_argument("--purge-old", action="store_true", help="Delete 2021-2023 documents")
    parser.add_argument("--load-faqs", type=str, metavar="FILE", help="Path to JSON file with FAQs")
    
    args = parser.parse_args()
    
    if args.purge_old:
        purge_old_documents()
        
    if args.load_faqs:
        load_custom_faqs(args.load_faqs)
        
    if not args.purge_old and not args.load_faqs:
        parser.print_help()
