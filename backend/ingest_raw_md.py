"""
ingest_raw_md.py — Ingest all raw crawled .md files into ChromaDB + KG
=======================================================================
Reads all .md files from data/raw/, chunks them, and adds to ChromaDB
and Knowledge Graph. This covers all the JS-rendered IIT Jammu pages
that were previously crawled by the scraper.

Run from backend/:
    python ingest_raw_md.py
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_raw_md")

for noisy in ['httpx', 'httpcore', 'huggingface_hub', 'sentence_transformers',
              'transformers', 'urllib3', 'filelock', 'chromadb']:
    logging.getLogger(noisy).setLevel(logging.ERROR)


def filename_to_url(filename: str) -> str:
    """Convert .md filename back to approximate source URL."""
    name = filename.replace(".md", "")
    # Replace __ with /
    name = name.replace("__", "/")
    # Add https:// if it starts with www. or iitjammu or sites.google
    if name.startswith("www.") or name.startswith("iitjammu") or name.startswith("sites."):
        return f"https://{name}"
    return f"https://{name}"


def filename_to_title(filename: str) -> str:
    """Convert .md filename to a readable title."""
    name = filename.replace(".md", "")
    # Replace __ with " > "
    name = name.replace("__", " > ")
    # Remove domain prefix
    name = re.sub(r"^(www\.)?iitjammu\.ac\.in\s*>\s*", "IIT Jammu - ", name)
    name = re.sub(r"^sites\.google\.com\s*>\s*", "Google Sites - ", name)
    # Clean up
    name = name.replace("_", " ").replace("-", " ").strip()
    return name[:120]


from typing import Dict

def infer_department(text: str, title: str, filename: str = "") -> str:
    """Infer department using filename, title and content."""
    combined = f"{filename} {title} {text[:500]}".lower()
    
    # Precise filename/path matching first
    if "computer_science" in combined or "cse" in combined:
        return "Computer Science and Engineering"
    if "electrical" in combined or " ee " in combined or "__ee" in combined:
        return "Electrical Engineering"
    if "mechanical" in combined or " me " in combined or "__me" in combined:
        return "Mechanical Engineering"
    if "civil" in combined or " ce " in combined or "__civil" in combined:
        return "Civil Engineering"
    if "chemical" in combined or " che " in combined or "__chemical" in combined:
        return "Chemical Engineering"
    if "materials" in combined or "__materials" in combined or "mme" in combined:
        return "Materials Engineering"
    if "biosciences" in combined or "bsbe" in combined or "bioengineering" in combined:
        return "Biosciences and Bioengineering"
    if "mathematics" in combined or " math " in combined:
        return "Mathematics"
    if "physics" in combined:
        return "Physics"
    if "chemistry" in combined:
        return "Chemistry"
    if "humanities" in combined or "hss" in combined or "social sciences" in combined:
        return "Humanities and Social Sciences"
        
    return "General"


def infer_metadata(text: str, title: str, filename: str = "") -> Dict[str, str]:
    """Extract document_type, department, last_updated, and target_audience."""
    dept = infer_department(text, title, filename)
    
    combined = f"{filename} {title} {text[:1000]}".lower()
    
    # 1. Document Type Classification
    doc_type = "General_Page"
    if any(w in combined for w in ["hod-message", "hod_message", "message-from-head", "message-from-department-hod"]) or "message from the head" in combined:
        doc_type = "HOD_Message"
    elif any(w in combined for w in ["faculty-list", "faculty_profile", "faculty-search", "staff-page"]) or "designation" in combined or "research interest" in combined:
        doc_type = "Faculty_Profile"
    elif "placement-report" in combined or "placement-highlights" in combined or "placement-academia" in combined or "placement-industry" in combined or "placement record" in combined:
        doc_type = "Placement_Record"
    elif "fee-structure" in combined or "mess-fee" in combined or "tuition fee" in combined:
        doc_type = "Fee_Structure"
    elif "admission-policy" in combined or "admissions" in combined or "eligibility-criteria" in combined:
        doc_type = "Admission_Policy"
    elif "circular" in combined or "notification" in combined or "notice" in combined:
        doc_type = "Academic_Circular"
        
    # 2. Target Audience Classification
    audiences = []
    if any(w in combined for w in ["btech", "b.tech", "jee", "josaa", "undergraduate", "ug"]):
        audiences.append("UG")
    if any(w in combined for w in ["mtech", "m.tech", "gate", "ccmt", "postgraduate", "pg", "msc", "m.sc"]):
        audiences.append("PG")
    if any(w in combined for w in ["phd", "ph.d", "doctoral", "research scholar", "pmrf"]):
        audiences.append("PhD")
    if any(w in combined for w in ["faculty", "professor", "prof", "dr.", "recruitment", "careers", "jobs"]):
        audiences.append("Faculty")
        
    target_audience = " / ".join(audiences) if audiences else "General"
    
    # 3. Last Updated Date extraction
    last_updated = "2026-05-22"  # fallback
    # Search for YYYY-MM-DD
    date_match = re.search(r"\b(202\d-\d{2}-\d{2})\b", text)
    if date_match:
        last_updated = date_match.group(1)
    else:
        # Try DD-MM-YYYY
        date_match2 = re.search(r"\b(\d{2}-\d{2}-202\d)\b", text)
        if date_match2:
            try:
                parts = date_match2.group(1).split("-")
                last_updated = f"{parts[2]}-{parts[1]}-{parts[0]}"
            except Exception:
                pass
                
    return {
        "document_type": doc_type,
        "department": dept,
        "last_updated": last_updated,
        "target_audience": target_audience
    }


def infer_topic(text: str, title: str) -> str:
    """Infer topic category."""
    combined = f"{title} {text[:300]}".lower()
    if any(w in combined for w in ["faculty", "professor", "prof.", "dr."]):
        return "Faculty"
    if any(w in combined for w in ["admission", "jee", "gate", "josaa"]):
        return "Admissions"
    if any(w in combined for w in ["fee", "tuition", "charges"]):
        return "Fee Structure"
    if any(w in combined for w in ["placement", "ctc", "recruit"]):
        return "Placements"
    if any(w in combined for w in ["research", "lab", "publication"]):
        return "Research"
    if any(w in combined for w in ["hostel", "mess", "campus", "library"]):
        return "Campus & Facilities"
    if any(w in combined for w in ["program", "course", "curriculum"]):
        return "Academic Programs"
    return "General"


def chunk_text(text: str, max_chunk: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chunk:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk
        chunk = text[start:end].strip()
        if chunk and len(chunk) >= 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def main():
    import warnings
    warnings.filterwarnings("ignore")

    project_root = Path(backend_dir).parent
    raw_dir = project_root / "data" / "raw"

    print("=" * 60)
    print("  INGEST RAW .MD FILES → ChromaDB + Knowledge Graph")
    print("=" * 60)

    # Find all .md files
    md_files = sorted(raw_dir.glob("*.md"))
    print(f"\n📂 Found {len(md_files)} .md files in {raw_dir}")

    if not md_files:
        print("❌ No .md files found!")
        return

    # Initialize ChromaDB
    print("\n📦 Loading ChromaDB...")
    from vectorstore.chroma_store import get_chroma_store
    chroma = get_chroma_store()
    initial_count = chroma.count()
    print(f"   ChromaDB initial count: {initial_count} documents")

    # Initialize Knowledge Graph
    print("🕸️  Loading Knowledge Graph...")
    from knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    initial_nodes = kg.node_count()
    initial_edges = kg.edge_count()
    print(f"   KG initial: {initial_nodes} nodes, {initial_edges} edges")

    print(f"\n🔄 Processing {len(md_files)} markdown files...\n")

    total_docs_added = 0
    total_processed = 0
    total_skipped = 0

    for i, md_file in enumerate(md_files, 1):
        filename = md_file.name
        title = filename_to_title(filename)
        source_url = filename_to_url(filename)

        # Read file
        try:
            text = md_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"  [{i:3d}/{len(md_files)}] {title[:50]:50s} ❌ READ ERROR: {e}")
            total_skipped += 1
            continue

        if len(text) < 50:
            print(f"  [{i:3d}/{len(md_files)}] {title[:50]:50s} ⏩ TOO SHORT ({len(text)} chars)")
            total_skipped += 1
            continue

        total_processed += 1
        meta = infer_metadata(text, title, filename)
        department = meta["department"]
        topic = infer_topic(text, title)

        # Chunk
        chunks = chunk_text(text)

        # Prepare docs for ChromaDB
        docs = []
        for ci, chunk in enumerate(chunks):
            doc_title = title if len(chunks) == 1 else f"{title} (part {ci+1})"
            docs.append({
                "text": f"IIT Jammu Information\nTitle: {doc_title}\nDepartment: {department}\nSource: {source_url}\n\n{chunk}",
                "title": doc_title,
                "topic": topic,
                "source_url": source_url,
                "department": department,
                "doc_type": "crawled_page",
                "document_type": meta["document_type"],
                "last_updated": meta["last_updated"],
                "target_audience": meta["target_audience"],
                "year": "2026",
            })

        # Add to ChromaDB
        added = chroma.add_documents(docs, batch_size=100)
        total_docs_added += added

        # Add to Knowledge Graph
        kg.extract_and_add_from_text(text=text[:5000], title=title, source_url=source_url)

        status = f"✅ {added} docs" if added > 0 else "⏩ (dedup)"
        print(f"  [{i:3d}/{len(md_files)}] {title[:50]:50s} {status} ({len(chunks)} chunks)")

    # Save Knowledge Graph
    print("\n💾 Saving Knowledge Graph...")
    kg.save()

    # Final stats
    final_count = chroma.count()
    final_nodes = kg.node_count()
    final_edges = kg.edge_count()

    print("\n" + "=" * 60)
    print("  ✅ RAW MD INGESTION COMPLETE!")
    print("=" * 60)
    print(f"  Files processed: {total_processed}/{len(md_files)}")
    print(f"  Files skipped:   {total_skipped}")
    print(f"  ChromaDB: {initial_count} → {final_count} documents (+{final_count - initial_count})")
    print(f"  Knowledge Graph: {initial_nodes} → {final_nodes} nodes (+{final_nodes - initial_nodes})")
    print(f"  KG Edges: {initial_edges} → {final_edges} edges (+{final_edges - initial_edges})")
    print("=" * 60)


if __name__ == "__main__":
    main()
