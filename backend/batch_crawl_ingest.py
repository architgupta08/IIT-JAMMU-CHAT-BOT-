"""
batch_crawl_ingest.py — Crawl professor profile URLs and ingest into ChromaDB + KG
==================================================================================
Fetches each URL, extracts text, adds to ChromaDB as searchable documents,
and extracts entities into the Knowledge Graph.

Run from backend/:
    python batch_crawl_ingest.py
"""

import os
import sys
import re
import time
import logging
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend is on path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("batch_crawl")

# Suppress noisy loggers
for noisy in ['httpx', 'httpcore', 'huggingface_hub', 'sentence_transformers',
              'transformers', 'urllib3', 'filelock', 'chromadb']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ── All URLs to crawl ──────────────────────────────────────────────
URLS = [
    # ── EE Department ──────────────────────────────────────────────
    ("https://www.iitjammu.ac.in/ee/faculty-list.html", "EE Faculty List", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~ajaysingh", "Prof. Ajay Singh - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~alokkumarsaxena", "Prof. Alok Kumar Saxena - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~ambikaprasadshah", "Prof. Ambika Prasad Shah - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~ankitdubey", "Prof. Ankit Dubey - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~ankurbansal", "Prof. Ankur Bansal - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~anupshukla", "Prof. Anup Shukla - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~archanarajput", "Prof. Archana Rajput - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~arunkumarverma", "Prof. Arun Kumar Verma - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~badrinsubudhi", "Prof. Badri N Subudhi - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~chandanyadav", "Prof. Chandan Yadav - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~ibhanchandrath", "Prof. Ibhan Chand Rath - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~kankatghosh", "Prof. Kankat Ghosh - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~karannathwani", "Prof. Karan Nathwani - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~nalinkumarsharma", "Prof. Nalin Kumar Sharma - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~padminisingh", "Prof. Padmini Singh - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~priyankamishra", "Prof. Priyanka Mishra - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~priyatoshjena", "Prof. Priyatosh Jena - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~ravikantsaini", "Prof. Ravikant Saini - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~rohitbuddhiramchaurasiya", "Prof. Rohit Chaurasiya - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~satyadevahlawat", "Prof. Satyadev Ahlawat - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~shikhabaghel", "Prof. Shikha Baghel - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~somasdhavala", "Prof. Soma S Dhavala - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/faculty.html?faculty=~sudhakarmodem", "Prof. Sudhakar Modem - EE", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/research-areas.html", "EE Research Areas", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/placement-industry.html", "EE Placements - Industry", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/ee/placement-academia.html", "EE Placements - Academia", "Electrical Engineering"),

    # ── EE External/Lab Pages ──────────────────────────────────────
    ("https://www.ic-resq.com/", "IC-RESQ Lab (Ambika Prasad Shah)", "Electrical Engineering"),
    ("http://anupshukla.in/", "Anup Shukla Personal Page", "Electrical Engineering"),
    ("https://www.aadhritlab.org/", "AADHRIT Lab (Anup Shukla)", "Electrical Engineering"),
    ("https://sites.google.com/view/badrisubudhi/home", "Badri N Subudhi - Personal Page", "Electrical Engineering"),
    ("https://sites.google.com/view/badrisubudhi/home/publications?authuser=0", "Badri N Subudhi - Publications", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/underwater-artificial-intelligence-lab/index.html", "Underwater AI Lab - Home", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/underwater-artificial-intelligence-lab/facilities.html", "Underwater AI Lab - Facilities", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/underwater-artificial-intelligence-lab/publications.html", "Underwater AI Lab - Publications", "Electrical Engineering"),
    ("https://www.iitjammu.ac.in/underwater-artificial-intelligence-lab/investigators.html", "Underwater AI Lab - Investigators", "Electrical Engineering"),
    ("https://sites.google.com/view/rohitchaurasiyaiitmandi", "Rohit Chaurasiya - Personal Page", "Electrical Engineering"),
    ("https://sites.google.com/view/rohitchaurasiyaiitmandi/research-interests?authuser=0", "Rohit Chaurasiya - Research", "Electrical Engineering"),
    ("https://sites.google.com/view/rohitchaurasiyaiitmandi/research-publications?authuser=0", "Rohit Chaurasiya - Publications", "Electrical Engineering"),
    ("https://sites.google.com/view/rohitchaurasiyaiitmandi/funded-projects?authuser=0", "Rohit Chaurasiya - Projects", "Electrical Engineering"),
    ("https://sites.google.com/view/rohitchaurasiyaiitmandi/asic-chip-design-and-testing?authuser=0", "Rohit Chaurasiya - ASIC Design", "Electrical Engineering"),
    ("https://sites.google.com/view/shikhabaghel/", "Shikha Baghel - Personal Page", "Electrical Engineering"),
    ("https://sites.google.com/view/shikhabaghel/research?authuser=0", "Shikha Baghel - Research", "Electrical Engineering"),
    ("https://sites.google.com/view/shikhabaghel/publications?authuser=0", "Shikha Baghel - Publications", "Electrical Engineering"),

    # ── CSE Department ─────────────────────────────────────────────
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list", "CSE Faculty List", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~aroofaimen", "Prof. Aroofa Imen - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~gauravvarshney", "Prof. Gaurav Varshney - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~harkeeratkaur", "Prof. Harkeerat Kaur - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~manojsinghgaur", "Prof. Manoj Singh Gaur - CSE (Director)", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~mrinmoybhattacharjee", "Prof. Mrinmoy Bhattacharjee - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~samareshbera", "Prof. Samaresh Bera - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~saradaprasadgochhayat", "Prof. Sarada Prasad Gochhayat - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~sayantanmukherjee", "Prof. Sayantan Mukherjee - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~shaifugupta", "Prof. Shaifu Gupta - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~sidharthmaheshwari", "Prof. Sidharth Maheshwari - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~subhasisbhattacharjee", "Prof. Subhasis Bhattacharjee - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~sumanbanerjee", "Prof. Suman Banerjee - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~sumitkpandey", "Prof. Sumit K Pandey - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~vinitjakhetiya", "Prof. Vinit Jakhetiya - CSE", "Computer Science and Engineering"),
    ("https://www.iitjammu.ac.in/computer_science_engineering/faculty-list/~yamunaprasad", "Prof. Yamuna Prasad - CSE", "Computer Science and Engineering"),

    # ── CSE External Pages ─────────────────────────────────────────
    ("https://mrinmoy-iitg.github.io/Webpage/index.html", "Mrinmoy Bhattacharjee - Personal Page", "Computer Science and Engineering"),
    ("https://mrinmoy-iitg.github.io/Webpage/research.html", "Mrinmoy Bhattacharjee - Research", "Computer Science and Engineering"),
    ("https://mrinmoy-iitg.github.io/Webpage/publications.html", "Mrinmoy Bhattacharjee - Publications", "Computer Science and Engineering"),
    ("https://samareshbera.github.io/", "Samaresh Bera - Personal Page", "Computer Science and Engineering"),
    ("https://samareshbera.github.io/projects/", "Samaresh Bera - Projects", "Computer Science and Engineering"),
    ("https://samareshbera.github.io/publication/", "Samaresh Bera - Publications", "Computer Science and Engineering"),
    ("https://samareshbera.github.io/teaching/", "Samaresh Bera - Teaching", "Computer Science and Engineering"),
    ("https://samareshbera.github.io/awards/", "Samaresh Bera - Awards", "Computer Science and Engineering"),
    ("https://sites.google.com/view/vinitjakhetiya/home?authuser=0", "Vinit Jakhetiya - Personal Page", "Computer Science and Engineering"),
    ("https://sites.google.com/view/vinitjakhetiya/publications?authuser=0", "Vinit Jakhetiya - Publications", "Computer Science and Engineering"),
    ("https://www.cse.iitd.ernet.in/~yprasad/", "Yamuna Prasad - IIT Delhi Page", "Computer Science and Engineering"),

    # ── ME Department ──────────────────────────────────────────────
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty-list.html", "ME Faculty List", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~abhaysharma", "Prof. Abhay Sharma - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~akashsubhashawale", "Prof. Akash Subhash Awale - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~angshuman", "Prof. Angshuman - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~arvindkrajput", "Prof. Arvind K Rajput - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~ashutoshbijalwan", "Prof. Ashutosh Bijalwan - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~goutamdutta", "Prof. Goutam Dutta - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~navneetkumar", "Prof. Navneet Kumar - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~pothukuchiharish", "Prof. Pothukuchi Harish - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~vrajkumar", "Prof. V Raj Kumar - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~roshanudarampatil", "Prof. Roshan Udaram Patil - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~sahilkalra", "Prof. Sahil Kalra - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~samratrao", "Prof. Samrat Rao - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~bsatyasekhar", "Prof. B Satya Sekhar - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~saurabhbiswas", "Prof. Saurabh Biswas - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~shanmugadaskp", "Prof. Shanmugadas KP - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~shivas", "Prof. Shiva Sekar - ME", "Mechanical Engineering"),
    ("https://www.iitjammu.ac.in/mechanical_engineering/faculty.html?faculty=~vijaykumarpal", "Prof. Vijay Kumar Pal - ME", "Mechanical Engineering"),
    ("https://www.highpressurefluidlab.com/", "High Pressure Fluid Lab (Goutam Dutta)", "Mechanical Engineering"),

    # ── CE Department ──────────────────────────────────────────────
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list", "CE Faculty List", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~ankitkathuria", "Prof. Ankit Kathuria - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~anuragmisra", "Prof. Anurag Misra - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~chemboluvinay", "Prof. Chembolu Vinay - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~deepakyadav", "Prof. Deepak Yadav - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~dhanendrakumar", "Prof. Dhanendra Kumar - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~divyeshvarade", "Prof. Divyesh Varade - CE", "Civil Engineering"),
    ("http://iitjammu.ac.in/civil_engineering/faculty-list/~mirfaizanulhaq", "Prof. Mir Faizanul Haq - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~nitinjoshi", "Prof. Nitin Joshi - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~pervaizfathimakhatoonm", "Prof. Pervaiz Fathima Khatoon M - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~prasunhalder", "Prof. Prasun Halder - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~pratikkumar", "Prof. Pratik Kumar - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~rajendrakvarma", "Prof. Rajendra K Varma - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~rimenjamatia", "Prof. Rimen Jamatia - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~riyabhowmik", "Prof. Riya Bhowmik - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~sameerksarmap", "Prof. Sameer K Sarma P - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~sarahmariamabraham", "Prof. Sarah Maria M Abraham - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~sivakumarg", "Prof. Sivakumar G - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~srishtisingh", "Prof. Srishti Singh - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~surendrabeniwal", "Prof. Surendra Beniwal - CE", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/faculty-list/~vedprakashranjan", "Prof. Ved Prakash Ranjan - CE", "Civil Engineering"),
    ("https://sites.google.com/view/imass", "IMASS Lab (Ankit Kathuria)", "Civil Engineering"),
    ("https://sites.google.com/view/imass/publications?authuser=0", "IMASS Lab - Publications", "Civil Engineering"),

    # ── CE Programs ────────────────────────────────────────────────
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/environmental-engineering-home", "CE Environmental Engineering", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/environmental-engineering-faculty", "CE Environmental Engineering Faculty", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/geomatics-home", "CE Geomatics", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/geomatics-faculty", "CE Geomatics Faculty", "Civil Engineering"),
    ("https://iitjammu.ac.in/geotechnical-engineering/", "CE Geotechnical Engineering", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/structural-engineering-home", "CE Structural Engineering", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/structural-engineering-faculty", "CE Structural Engineering Faculty", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/transportation-engineering-home", "CE Transportation Engineering", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/transportation-engineering-faculty", "CE Transportation Engineering Faculty", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/water-resource-engineering-home", "CE Water Resource Engineering", "Civil Engineering"),
    ("https://www.iitjammu.ac.in/civil_engineering/program-list/water-resource-engineering-faculty", "CE Water Resource Engineering Faculty", "Civil Engineering"),

    # ── General Pages ──────────────────────────────────────────────
    ("https://www.iitjammu.ac.in/academics/function-list", "Academic Functions List", "Academics"),
    ("https://www.iitjammu.ac.in/academics/people-list", "Academic People List", "Academics"),
    ("https://www.iitjammu.ac.in/holidays-list-2026", "Holidays List 2026", "General"),
    ("https://www.iitjammu.ac.in/faq-main-website", "IIT Jammu FAQ", "General"),
]


def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch URL content and extract text using BeautifulSoup."""
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "noscript", "iframe"]):
            tag.decompose()

        # Try to get main content
        main = soup.find("main") or soup.find("article") or soup.find(class_="content") or soup.find("body")
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up: collapse whitespace, remove very short lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        # Truncate very long pages
        if len(text) > 15000:
            text = text[:15000]

        return text if len(text) >= 50 else None

    except Exception as e:
        logger.warning(f"  ❌ Failed to fetch {url}: {e}")
        return None


def chunk_text_simple(text: str, title: str, max_chunk: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chunk:
        return [text]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + max_chunk
        chunk = text[start:end].strip()
        if chunk and len(chunk) >= 50:
            idx += 1
            chunks.append(chunk)
        start = end - overlap
    return chunks


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("  BATCH CRAWL & INGEST — IIT Jammu Faculty/Dept Pages")
    print(f"  Total URLs to crawl: {len(URLS)}")
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

    # Crawl and process
    print(f"\n🔄 Starting crawl of {len(URLS)} URLs...\n")

    total_docs_added = 0
    total_crawled = 0
    total_failed = 0

    for i, (url, title, department) in enumerate(URLS, 1):
        print(f"  [{i:3d}/{len(URLS)}] {title[:50]:50s} ", end="", flush=True)

        text = fetch_url(url)
        if not text:
            total_failed += 1
            print("❌ SKIP")
            continue

        total_crawled += 1

        # Chunk text
        chunks = chunk_text_simple(text, title)

        # Prepare documents for ChromaDB
        docs = []
        for ci, chunk in enumerate(chunks):
            doc_title = f"{title}" if len(chunks) == 1 else f"{title} (part {ci+1})"
            docs.append({
                "text": f"IIT Jammu Faculty/Department Information\nTitle: {doc_title}\nDepartment: {department}\n\n{chunk}",
                "title": doc_title,
                "topic": "Faculty" if "Prof." in title else department,
                "source_url": url,
                "department": department,
                "doc_type": "faculty_profile" if "Prof." in title or "faculty" in url.lower() else "department_info",
                "year": "2026",
            })

        # Add to ChromaDB
        added = chroma.add_documents(docs, batch_size=50)
        total_docs_added += added

        # Add to Knowledge Graph
        for chunk in chunks:
            kg.extract_and_add_from_text(
                text=chunk,
                title=title,
                source_url=url,
            )

        # Also add faculty as Person entity explicitly
        if "Prof." in title:
            name_match = re.search(r"Prof\.\s+(.+?)\s*-\s*", title)
            if name_match:
                faculty_name = name_match.group(1).strip()
                kg.add_entity(faculty_name, "Person", {
                    "department": department,
                    "source_url": url,
                    "role": "Faculty",
                })
                kg.add_relationship("IIT Jammu", faculty_name, "RELATED_TO")
                kg.add_relationship(department, faculty_name, "LED_BY")

        status = f"✅ {added} docs" if added > 0 else "⏩ (dedup)"
        print(f"{status} ({len(chunks)} chunks, {len(text)} chars)")

        # Small delay to be polite
        time.sleep(0.3)

    # Save Knowledge Graph
    print("\n💾 Saving Knowledge Graph...")
    kg.save()

    # Final stats
    final_count = chroma.count()
    final_nodes = kg.node_count()
    final_edges = kg.edge_count()

    print("\n" + "=" * 60)
    print("  ✅ BATCH CRAWL & INGEST COMPLETE!")
    print("=" * 60)
    print(f"  URLs crawled successfully: {total_crawled}/{len(URLS)}")
    print(f"  URLs failed/skipped:       {total_failed}")
    print(f"  ChromaDB: {initial_count} → {final_count} documents (+{final_count - initial_count})")
    print(f"  Knowledge Graph: {initial_nodes} → {final_nodes} nodes (+{final_nodes - initial_nodes})")
    print(f"  KG Edges: {initial_edges} → {final_edges} edges (+{final_edges - initial_edges})")
    print("=" * 60)


if __name__ == "__main__":
    main()
