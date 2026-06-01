"""
ingest_faculty_api.py — Fetch faculty data directly from IIT Jammu's API
=========================================================================
The EE/ME/CE/CSE department pages load faculty data from:
    https://iitjammu.ac.in/api/faculty2/~facultyname

This script calls the API directly for every faculty member,
gets structured JSON, and injects into ChromaDB + Knowledge Graph.

Run from backend/:
    python ingest_faculty_api.py
"""

import os
import sys
import re
import time
import logging
import warnings
from typing import Optional, Dict, Any
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("faculty_api")

for noisy in ['httpx', 'httpcore', 'huggingface_hub', 'sentence_transformers',
              'transformers', 'urllib3', 'filelock', 'chromadb']:
    logging.getLogger(noisy).setLevel(logging.ERROR)


# ── All faculty usernames to fetch via API ──────────────────────────
# Format: (username_slug, department)
FACULTY_LIST = [
    # ── EE Department ──────────────────────────────────────────────
    ("~ajaysingh", "Electrical Engineering"),
    ("~alokkumarsaxena", "Electrical Engineering"),
    ("~ambikaprasadshah", "Electrical Engineering"),
    ("~ankitdubey", "Electrical Engineering"),
    ("~ankurbansal", "Electrical Engineering"),
    ("~anupshukla", "Electrical Engineering"),
    ("~archanarajput", "Electrical Engineering"),
    ("~arunkumarverma", "Electrical Engineering"),
    ("~badrinsubudhi", "Electrical Engineering"),
    ("~chandanyadav", "Electrical Engineering"),
    ("~ibhanchandrath", "Electrical Engineering"),
    ("~kankatghosh", "Electrical Engineering"),
    ("~karannathwani", "Electrical Engineering"),
    ("~nalinkumarsharma", "Electrical Engineering"),
    ("~padminisingh", "Electrical Engineering"),
    ("~priyankamishra", "Electrical Engineering"),
    ("~priyatoshjena", "Electrical Engineering"),
    ("~ravikantsaini", "Electrical Engineering"),
    ("~rohitbuddhiramchaurasiya", "Electrical Engineering"),
    ("~satyadevahlawat", "Electrical Engineering"),
    ("~shikhabaghel", "Electrical Engineering"),
    ("~somasdhavala", "Electrical Engineering"),
    ("~sudhakarmodem", "Electrical Engineering"),

    # ── CSE Department ─────────────────────────────────────────────
    ("~aroofaimen", "Computer Science and Engineering"),
    ("~gauravvarshney", "Computer Science and Engineering"),
    ("~harkeeratkaur", "Computer Science and Engineering"),
    ("~manojsinghgaur", "Computer Science and Engineering"),
    ("~mrinmoybhattacharjee", "Computer Science and Engineering"),
    ("~samareshbera", "Computer Science and Engineering"),
    ("~saradaprasadgochhayat", "Computer Science and Engineering"),
    ("~sayantanmukherjee", "Computer Science and Engineering"),
    ("~shaifugupta", "Computer Science and Engineering"),
    ("~sidharthmaheshwari", "Computer Science and Engineering"),
    ("~subhasisbhattacharjee", "Computer Science and Engineering"),
    ("~sumanbanerjee", "Computer Science and Engineering"),
    ("~sumitkpandey", "Computer Science and Engineering"),
    ("~vinitjakhetiya", "Computer Science and Engineering"),
    ("~yamunaprasad", "Computer Science and Engineering"),

    # ── ME Department ──────────────────────────────────────────────
    ("~abhaysharma", "Mechanical Engineering"),
    ("~akashsubhashawale", "Mechanical Engineering"),
    ("~angshuman", "Mechanical Engineering"),
    ("~arvindkrajput", "Mechanical Engineering"),
    ("~ashutoshbijalwan", "Mechanical Engineering"),
    ("~goutamdutta", "Mechanical Engineering"),
    ("~navneetkumar", "Mechanical Engineering"),
    ("~pothukuchiharish", "Mechanical Engineering"),
    ("~vrajkumar", "Mechanical Engineering"),
    ("~roshanudarampatil", "Mechanical Engineering"),
    ("~sahilkalra", "Mechanical Engineering"),
    ("~samratrao", "Mechanical Engineering"),
    ("~bsatyasekhar", "Mechanical Engineering"),
    ("~saurabhbiswas", "Mechanical Engineering"),
    ("~shanmugadaskp", "Mechanical Engineering"),
    ("~shivas", "Mechanical Engineering"),
    ("~vijaykumarpal", "Mechanical Engineering"),

    # ── CE Department ──────────────────────────────────────────────
    ("~ankitkathuria", "Civil Engineering"),
    ("~anuragmisra", "Civil Engineering"),
    ("~chemboluvinay", "Civil Engineering"),
    ("~deepakyadav", "Civil Engineering"),
    ("~dhanendrakumar", "Civil Engineering"),
    ("~divyeshvarade", "Civil Engineering"),
    ("~mirfaizanulhaq", "Civil Engineering"),
    ("~nitinjoshi", "Civil Engineering"),
    ("~pervaizfathimakhatoonm", "Civil Engineering"),
    ("~prasunhalder", "Civil Engineering"),
    ("~pratikkumar", "Civil Engineering"),
    ("~rajendrakvarma", "Civil Engineering"),
    ("~rimenjamatia", "Civil Engineering"),
    ("~riyabhowmik", "Civil Engineering"),
    ("~sameerksarmap", "Civil Engineering"),
    ("~sarahmariamabraham", "Civil Engineering"),
    ("~sivakumarg", "Civil Engineering"),
    ("~srishtisingh", "Civil Engineering"),
    ("~surendrabeniwal", "Civil Engineering"),
    ("~vedprakashranjan", "Civil Engineering"),
]

# API base URL (discovered from the JS source code)
API_BASE = "https://iitjammu.ac.in/api/faculty2/"


def strip_html(html_text: str) -> str:
    """Remove HTML tags from a string."""
    if not html_text:
        return ""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _is_url_only(text: str) -> bool:
    """Return True if text is just a URL or 'Website: URL' with no real content."""
    t = text.strip()
    if not t:
        return True
    # Strip leading "Website:" or "Visit" etc.
    t = re.sub(r'^(?:Website|Visit|See|Click|Link|For more details please visit my Website)\s*:?\s*', '', t, flags=re.IGNORECASE).strip()
    # Check if what remains is just a URL (possibly with trailing punctuation)
    if re.match(r'^https?://\S+$', t):
        return True
    # Check if it's just "Website: URL" pattern
    if re.match(r'^https?://\S+\s*$', t):
        return True
    return False


def fetch_faculty_data(slug: str) -> Optional[Dict[str, Any]]:
    """Fetch faculty JSON from IIT Jammu API with retry logic and longer timeout."""
    import requests
    url = f"{API_BASE}{slug}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30, verify=False)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.warning(f"  [Attempt {attempt}/{max_retries}] Failed to fetch {url}: {e}")
            if attempt < max_retries:
                time.sleep(2.0)
    return None



def format_faculty_document(data: Dict[str, Any], department: str) -> str:
    """Format structured faculty JSON into a readable text document."""
    sections = []

    name = data.get("faculty_name", "Unknown")
    designation = data.get("designation", "")
    email = data.get("email", "")

    sections.append(f"IIT Jammu Faculty Profile")
    sections.append(f"Name: {name}")
    sections.append(f"Department: {department}")
    if designation:
        sections.append(f"Designation: {strip_html(designation)}")
    if email:
        sections.append(f"Email: {strip_html(email)}")

    # All possible data fields from the API
    field_map = {
        "briefinfo": "Brief Information",
        "researchinterest": "Research Interests",
        "education": "Education Qualifications",
        "professionalbackground": "Professional Background",
        "academicinterests": "Academic Interests",
        "teachingengagements": "Teaching Engagements",
        "research_experience": "Research Experience",
        "jointresearchwork": "Joint Research Work",
        "researchsupervised": "Research Supervised",
        "researchprojects": "Research Projects",
        "consultancyprojects": "Consultancy Projects",
        "publications": "Publications",
        "patents": "Patents",
        "bookchapter": "Books / Book Chapters",
        "editorialservices": "Editorial Services",
        "administrativeresponsibilities": "Administrative Responsibilities",
        "awardsandhonours": "Awards and Honours",
        "currentopenings": "Current Openings",
        "otherinfo": "Other Information",
        "personalwebsite": "Personal Website",
    }

    # Track the personal website URL so we can add it once at the end
    personal_website_url = None

    for field, label in field_map.items():
        val = data.get(field)
        if val and str(val).strip():
            clean = strip_html(str(val))

            # Apply publications override FIRST (before URL-only check)
            # so that custom publication lists for Badri/Shaifu don't get filtered
            if field == "publications":
                if "Badri" in name or "Subudhi" in name:
                    clean = (
                        "Representative Publications:\n"
                        "1. Meghna, B. N. Subudhi, V. Jakhetiya, A. Bansal, T. Veerakumar, A. Ghosh, \"G-Unet: A Deep Learning Model for Image Inhomogeneity Correction.\"\n"
                        "2. A. Ghosh, H. Singh, S. Suman, B. N. Subudhi, Vinit Jakhetiya, T. Veerakumar, \"Transformer-based change detection model for remote sensing.\"\n"
                        "3. B. N. Subudhi, T. Veerakumar, V. Jakhetiya, \"Change Detection and Color Attention for Video Surveillance.\"\n"
                        "4. Selected research papers in medical image analysis and action recognition using deep learning.\n"
                        "For the full list of publications, please refer to the official website: https://sites.google.com/view/badrisubudhi/home"
                    )
                elif "Shaifu" in name or "Gupta" in name:
                    clean = (
                        "Representative Publications:\n"
                        "1. Shaifu Gupta et al., \"Learning to detect PII: Tabular vs. Document classification models for network traffic analysis\" (Journal of Information Security and Applications, 2025).\n"
                        "2. Shaifu Gupta et al., \"Long range dependence in cloud servers: a statistical analysis based on Google workload trace\" (Computing, 2020).\n"
                        "3. Shaifu Gupta et al., \"Relevance feedback based online learning model for resource bottleneck prediction in cloud servers\" (Neurocomputing, 2020).\n"
                        "4. Shaifu Gupta et al., \"Data Poisoning Attack by Label Flipping on SplitFed Learning\" (Communications in Computer and Information Science, 2023).\n"
                        "For the full list of publications, please refer to the Google Scholar profile: https://scholar.google.co.in/citations?hl=en&user=XGMQWb8AAAAJ"
                    )

            # Skip sections whose content is just a URL/website link
            # (many faculty like Badri Subudhi have "Website: URL" for every field,
            #  which pollutes the document with 12+ repetitions of the same link)
            if field != "personalwebsite" and _is_url_only(clean):
                # Capture the URL for the personal website section
                url_match = re.search(r'https?://\S+', clean)
                if url_match and not personal_website_url:
                    personal_website_url = url_match.group(0)
                continue

            if field == "personalwebsite":
                # Capture and store; we'll add it at the end
                url_match = re.search(r'https?://\S+', clean)
                if url_match:
                    personal_website_url = url_match.group(0)
                continue  # Will be appended at the end
            if clean and len(clean) > 3:
                sections.append(f"\n{label}:\n{clean}")

    # Append personal website once at the end (instead of repeating it in every section)
    if personal_website_url:
        sections.append(f"\nPersonal Website: {personal_website_url}")

    return "\n".join(sections)



def main():
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("  FACULTY API INGESTION — Direct from IIT Jammu API")
    print(f"  Total faculty to fetch: {len(FACULTY_LIST)}")
    print(f"  API endpoint: {API_BASE}")
    print("=" * 60)

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

    print(f"\n🔄 Fetching {len(FACULTY_LIST)} faculty profiles from API...\n")

    total_docs_added = 0
    total_fetched = 0
    total_failed = 0

    for i, (slug, department) in enumerate(FACULTY_LIST, 1):
        display_name = slug.replace("~", "").replace("_", " ").title()
        print(f"  [{i:3d}/{len(FACULTY_LIST)}] {display_name:40s} ", end="", flush=True)

        data = fetch_faculty_data(slug)
        if not data or not data.get("faculty_name"):
            total_failed += 1
            print("❌ NO DATA")
            continue

        total_fetched += 1
        faculty_name = data.get("faculty_name", display_name)

        # Format as document
        doc_text = format_faculty_document(data, department)

        if len(doc_text) < 100:
            print(f"⏩ TOO SHORT ({len(doc_text)} chars)")
            continue

        # Chunk if very long
        chunks = []
        if len(doc_text) <= 1500:
            chunks = [doc_text]
        else:
            # Split into manageable sections
            lines = doc_text.split("\n")
            current_chunk = []
            current_len = 0
            for line in lines:
                if current_len + len(line) > 1200 and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [f"IIT Jammu Faculty Profile (continued)\nName: {faculty_name}\nDepartment: {department}\n"]
                    current_len = len(current_chunk[0])
                current_chunk.append(line)
                current_len += len(line)
            if current_chunk:
                chunks.append("\n".join(current_chunk))

        # Add to ChromaDB
        docs = []
        for ci, chunk in enumerate(chunks):
            title = f"{faculty_name} - {department}" if len(chunks) == 1 else f"{faculty_name} - {department} (part {ci+1})"
            docs.append({
                "text": chunk,
                "title": title,
                "topic": "Faculty",
                "source_url": f"https://iitjammu.ac.in/api/faculty2/{slug}",
                "department": department,
                "doc_type": "faculty_profile_api",
                "year": "2026",
            })

        added = chroma.add_documents(docs, batch_size=50)
        total_docs_added += added

        # Add to Knowledge Graph
        kg.extract_and_add_from_text(text=doc_text[:5000], title=f"{faculty_name} - {department}")

        # Explicitly add person + department entities
        kg.add_entity(faculty_name, "Person", {
            "department": department,
            "designation": strip_html(data.get("designation", "")),
            "email": strip_html(data.get("email", "")),
            "role": "Faculty",
        })
        kg.add_relationship("IIT Jammu", faculty_name, "RELATED_TO")
        kg.add_relationship(department, faculty_name, "HAS_FACULTY")

        # Extract research interests as entities
        ri = data.get("researchinterest", "")
        if ri:
            clean_ri = strip_html(ri)
            # Try to link research keywords
            for kw in re.split(r'[,;•\n]', clean_ri):
                kw = kw.strip()
                if 5 < len(kw) < 80:
                    kg.add_entity(kw, "ResearchArea", {"department": department})
                    kg.add_relationship(faculty_name, kw, "RESEARCHES")

        status = f"✅ {added} docs ({len(chunks)} chunks, {len(doc_text)} chars)"
        print(status)

        time.sleep(0.2)  # Be polite to the API

    # Save Knowledge Graph
    print("\n💾 Saving Knowledge Graph...")
    kg.save()

    # Final stats
    final_count = chroma.count()
    final_nodes = kg.node_count()
    final_edges = kg.edge_count()

    print("\n" + "=" * 60)
    print("  ✅ FACULTY API INGESTION COMPLETE!")
    print("=" * 60)
    print(f"  Faculty fetched successfully: {total_fetched}/{len(FACULTY_LIST)}")
    print(f"  Faculty failed/no data:       {total_failed}")
    print(f"  ChromaDB: {initial_count} → {final_count} documents (+{final_count - initial_count})")
    print(f"  Knowledge Graph: {initial_nodes} → {final_nodes} nodes (+{final_nodes - initial_nodes})")
    print(f"  KG Edges: {initial_edges} → {final_edges} edges (+{final_edges - initial_edges})")
    print("=" * 60)


if __name__ == "__main__":
    main()
