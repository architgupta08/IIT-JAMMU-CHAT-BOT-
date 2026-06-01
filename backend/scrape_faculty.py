"""
scrape_faculty.py — Official IIT Jammu Faculty Ingestion
=========================================================
Uses the official https://iitjammu.ac.in/api/faculty endpoint
to extract real, structured faculty data:
  - name, designation, department
  - research / academic interests
  - publications
  - education / bio

Then injects into:
  1. ChromaDB vector store (for RAG retrieval)
  2. Knowledge Graph (for relationship-based queries)

Run:
    cd backend
    python scrape_faculty.py
"""

import sys, os, re, json, logging, time
sys.path.append(".")

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────
API_URL   = "https://iitjammu.ac.in/api/faculty"
BASE_URL  = "https://iitjammu.ac.in"
HEADERS   = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.iitjammu.ac.in/",
}

# Only these designations are real teaching/research faculty
REAL_DESIGNATIONS = {
    "professor", "associate professor", "assistant professor",
    "adjunct professor", "visiting professor", "honorary professor",
    "lecturer", "research scientist", "senior research scientist",
}

# Departments we care about
TARGET_DEPARTMENTS = {
    "Computer Science and Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Mathematics",
    "Mathematics and Computing",
    "Physics",
    "Chemistry",
    "Humanities and Social Sciences",
    "Biosciences and Bioengineering",
    "Materials Engineering",
    "Interdisciplinary",
}


def clean_html(html_str: str) -> str:
    """Strip HTML tags and clean up whitespace."""
    if not html_str:
        return ""
    text = BeautifulSoup(html_str, "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_real_faculty(rec: dict) -> bool:
    """Return True only for real teaching/research faculty."""
    designation = (rec.get("designation") or "").strip().lower()
    ftype = (rec.get("facultytype") or "").lower()

    # Skip alumni, students, staff, etc.
    skip_types = {"alumni", "student", "phd", "mtech", "btech", "staff", "project"}
    if any(s in ftype for s in skip_types):
        return False
    if any(s in designation for s in skip_types):
        return False

    # Must have a real faculty designation
    return any(d in designation for d in REAL_DESIGNATIONS)


def get_faculty_profile_url(rec: dict) -> str:
    url_part = rec.get("faculty_url", "")
    if url_part:
        slug = url_part.lstrip("~")
        return f"https://iitjammu.ac.in/faculty/{slug}"
    return BASE_URL


def build_document_text(rec: dict) -> str:
    """Build rich structured text for ChromaDB from one faculty record."""
    name        = rec.get("faculty_name", "").strip()
    salutation  = rec.get("salutation", "").strip()
    designation = rec.get("designation", "").strip()
    depts       = rec.get("department", [])
    dept        = ", ".join(depts) if isinstance(depts, list) else str(depts)
    email       = rec.get("email", "").strip()
    profile_url = get_faculty_profile_url(rec)

    research    = clean_html(rec.get("academicinterests") or "")
    experience  = clean_html(rec.get("research_experience") or "")
    publications= clean_html(rec.get("publications") or "")
    education   = clean_html(rec.get("education") or "")
    biosketch   = clean_html(rec.get("biosketch") or "")
    otherinfo   = clean_html(rec.get("otherinfo") or "")

    full_name = f"{salutation} {name}".strip()

    lines = [
        f"Faculty Profile: {full_name}",
        f"Name: {full_name}",
        f"Designation: {designation}",
        f"Department: {dept}",
    ]
    if email and "@iitjammu" in email:
        lines.append(f"Email: {email}")
    if research:
        lines.append(f"\nResearch / Academic Interests:\n{research}")
    if experience:
        lines.append(f"\nResearch Experience:\n{experience[:1200]}")
    if publications:
        lines.append(f"\nSelected Publications:\n{publications[:1500]}")
    if education:
        lines.append(f"\nEducation:\n{education[:600]}")
    if biosketch and biosketch.lower() not in ("no", "none", ""):
        lines.append(f"\nBio:\n{biosketch[:400]}")
    if otherinfo:
        lines.append(f"\nOther Info:\n{otherinfo[:300]}")
    lines.append(f"\nProfile URL: {profile_url}")

    return "\n".join(lines)


def build_chroma_doc(rec: dict) -> dict:
    """Return a dict ready for chroma_store.add_documents()."""
    name   = rec.get("faculty_name", "Unknown").strip()
    depts  = rec.get("department", [])
    dept   = depts[0] if depts else "IIT Jammu"

    return {
        "text":       build_document_text(rec),
        "title":      f"{rec.get('salutation','')} {name}".strip(),
        "source_url": get_faculty_profile_url(rec),
        "topic":      "Faculty Profile",
        "doc_type":   "faculty",
        "department": dept,
        "year":       "2025",
    }


def update_knowledge_graph(kg_graph, rec: dict):
    """Add faculty → department and faculty → research area edges to the KG."""
    name        = rec.get("faculty_name", "").strip()
    designation = rec.get("designation", "").strip()
    depts       = rec.get("department", [])
    research_raw= clean_html(rec.get("academicinterests") or "")
    email       = rec.get("email", "").strip()

    if not name or len(name) < 3:
        return

    full_name = f"{rec.get('salutation','')} {name}".strip()

    # Faculty node
    kg_graph.add_node(
        full_name,
        type="faculty",
        designation=designation,
        email=email,
        profile_url=get_faculty_profile_url(rec),
    )

    # Department edges
    for dept in (depts if isinstance(depts, list) else [depts]):
        dept = dept.strip()
        if not dept:
            continue
        if not kg_graph.has_node(dept):
            kg_graph.add_node(dept, type="department")
        kg_graph.add_edge(full_name, dept, relation="belongs_to")

    # Research area edges
    if research_raw:
        areas = re.split(r"[,;|•\n\r\t]+", research_raw)
        for area in areas:
            area = re.sub(r"\s+", " ", area).strip()
            if 5 < len(area) < 100:
                if not kg_graph.has_node(area):
                    kg_graph.add_node(area, type="research_area")
                kg_graph.add_edge(full_name, area, relation="researches")


# ── Main ───────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("IIT Jammu Official Faculty Ingestion")
    logger.info(f"API: {API_URL}")
    logger.info("=" * 60)

    # ── Step 1: Fetch from official API ──────────────────────────
    logger.info("\n[1/4] Fetching faculty data from official API...")
    try:
        resp = requests.get(API_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        all_records = resp.json()
        logger.info(f"    Total records received: {len(all_records)}")
    except Exception as e:
        logger.error(f"Failed to fetch API: {e}")
        sys.exit(1)

    # ── Step 2: Filter real faculty only ─────────────────────────
    logger.info("\n[2/4] Filtering real teaching/research faculty...")
    real_faculty = [r for r in all_records if is_real_faculty(r)]
    logger.info(f"    Real faculty found: {len(real_faculty)}")

    # Log department breakdown
    dept_counts = {}
    for f in real_faculty:
        for d in (f.get("department") or ["Unknown"]):
            dept_counts[d] = dept_counts.get(d, 0) + 1
    for dept, cnt in sorted(dept_counts.items(), key=lambda x: -x[1]):
        logger.info(f"      {dept}: {cnt} faculty")

    # ── Step 3: Save raw JSON ─────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "faculty_data.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(real_faculty, fp, indent=2, ensure_ascii=False)
    logger.info(f"\n    Saved raw data → {json_path}")

    # ── Step 4: Inject into ChromaDB ─────────────────────────────
    logger.info("\n[3/4] Injecting into ChromaDB...")
    try:
        from vectorstore.chroma_store import get_chroma_store
        chroma = get_chroma_store()

        # First delete old faculty documents to avoid duplication
        logger.info("    Purging old faculty docs from ChromaDB...")
        try:
            result = chroma._collection.get(where={"doc_type": "faculty"})
            old_ids = result.get("ids", [])
            if old_ids:
                chroma._collection.delete(ids=old_ids)
                logger.info(f"    Deleted {len(old_ids)} old faculty docs")
        except Exception as e:
            logger.warning(f"    Could not purge old faculty docs: {e}")

        # Build and insert new documents
        docs = []
        for rec in real_faculty:
            doc = build_chroma_doc(rec)
            if len(doc["text"]) > 100:  # skip stubs
                docs.append(doc)

        logger.info(f"    Inserting {len(docs)} faculty documents...")
        if docs:
            # Insert in batches of 50
            batch_size = 50
            inserted = 0
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                chroma.add_documents(batch)
                inserted += len(batch)
                logger.info(f"    ... {inserted}/{len(docs)} inserted")
            logger.info(f"✅ ChromaDB: {inserted} faculty documents inserted")

    except Exception as e:
        logger.error(f"ChromaDB injection failed: {e}")
        import traceback; traceback.print_exc()

    # ── Step 5: Inject into Knowledge Graph ──────────────────────
    logger.info("\n[4/4] Updating Knowledge Graph...")
    try:
        import networkx as nx
        from services.knowledge_graph import get_knowledge_graph
        from config.settings import get_settings

        kg_service = get_knowledge_graph()
        kg_graph = kg_service._graph

        nodes_before = kg_graph.number_of_nodes()
        edges_before = kg_graph.number_of_edges()

        for rec in real_faculty:
            update_knowledge_graph(kg_graph, rec)

        nodes_after = kg_graph.number_of_nodes()
        edges_after = kg_graph.number_of_edges()
        logger.info(f"    Nodes: {nodes_before} → {nodes_after} (+{nodes_after-nodes_before})")
        logger.info(f"    Edges: {edges_before} → {edges_after} (+{edges_after-edges_before})")

        # Save updated graph
        settings = get_settings()
        kg_path = getattr(settings, "kg_file",
                          os.path.join(out_dir, "knowledge_graph.graphml"))
        if not os.path.isabs(kg_path):
            kg_path = os.path.join(os.path.dirname(__file__), kg_path)
        nx.write_graphml(kg_graph, kg_path)
        logger.info(f"✅ Knowledge Graph saved → {kg_path}")

    except Exception as e:
        logger.error(f"Knowledge Graph update failed: {e}")
        import traceback; traceback.print_exc()

    # ── Also update autocomplete service hints ────────────────────
    logger.info("\n[BONUS] Updating autocomplete hints from faculty data...")
    try:
        from autocomplete.service import get_autocomplete_service
        svc = get_autocomplete_service()
        added = 0
        for rec in real_faculty:
            name = rec.get("faculty_name", "").strip()
            sal  = rec.get("salutation", "").strip()
            depts= rec.get("department", [])
            dept = depts[0] if depts else ""
            if name and len(name) > 3:
                full = f"{sal} {name}".strip()
                svc._trie.insert(full, frequency=88, category="faculty")
                svc._all_terms.append({"text": full, "category": "faculty", "frequency": 88})
                svc._trie.insert(f"Who is {full}?", frequency=85, category="faculty")
                svc._all_terms.append({"text": f"Who is {full}?", "category": "faculty", "frequency": 85})
                if dept:
                    svc._trie.insert(f"What are the research interests of {full}?", frequency=80, category="faculty")
                    svc._all_terms.append({"text": f"What are the research interests of {full}?", "category": "faculty", "frequency": 80})
                added += 1
        logger.info(f"✅ Autocomplete: added {added} faculty name hints")
    except Exception as e:
        logger.warning(f"Autocomplete update skipped: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Faculty ingestion complete!")
    logger.info(f"   {len(real_faculty)} faculty records processed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
