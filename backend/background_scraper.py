"""
background_scraper.py — Background Web Scraper for IIT Jammu
=============================================================
Runs as a background task while the FastAPI server is live.
Periodically crawls IIT Jammu website pages and PDFs,
adding new content to ChromaDB and the Knowledge Graph.

Design:
  - Uses requests + BeautifulSoup (lightweight, no Playwright)
  - Incremental: only processes new/changed content
  - Append-only: NEVER deletes existing data in ChromaDB or KG
  - Rate-limited: respects the website with delays between requests
  - Tracks state in a JSON file for resume capability
"""

import os
import re
import json
import hashlib
import logging
import time
from typing import List, Dict, Set, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────
SCRAPER_INTERVAL_HOURS = int(os.getenv("SCRAPER_INTERVAL_HOURS", "6"))
SCRAPER_ENABLED = os.getenv("SCRAPER_ENABLED", "true").lower() == "true"
SCRAPER_MAX_PAGES = int(os.getenv("SCRAPER_MAX_PAGES_PER_RUN", "50"))
SCRAPER_DELAY = float(os.getenv("SCRAPER_DELAY_SECONDS", "2"))
SCRAPER_RUN_ON_STARTUP = os.getenv("SCRAPER_RUN_ON_STARTUP", "true").lower() == "true"
STATE_FILE = os.getenv("SCRAPER_STATE_FILE", "data/processed/scraper_state.json")

BASE_URL = "https://www.iitjammu.ac.in"

# ── Seed URLs ──────────────────────────────────────────────────────
SEED_URLS = [
    f"{BASE_URL}/",
    f"{BASE_URL}/sitemap.xml",
    f"{BASE_URL}/sitemap_index.xml",
    f"{BASE_URL}/about",
    f"{BASE_URL}/about-iit-jammu",
    f"{BASE_URL}/director",
    f"{BASE_URL}/administration",
    f"{BASE_URL}/contact-us",
    f"{BASE_URL}/academics",
    f"{BASE_URL}/btechprogramme",
    f"{BASE_URL}/mtechprogramme",
    f"{BASE_URL}/phd",
    f"{BASE_URL}/fee-structure",
    f"{BASE_URL}/fee",
    f"{BASE_URL}/admissions",
    f"{BASE_URL}/ug-admissions",
    f"{BASE_URL}/pg-admissions",
    f"{BASE_URL}/scholarship",
    f"{BASE_URL}/computer_science_engineering",
    f"{BASE_URL}/electrical_engineering",
    f"{BASE_URL}/mechanical_engineering",
    f"{BASE_URL}/civil_engineering",
    f"{BASE_URL}/chemical-engineering",
    f"{BASE_URL}/mathematics",
    f"{BASE_URL}/physics",
    f"{BASE_URL}/chemistry",
    f"{BASE_URL}/hss",
    f"{BASE_URL}/departments",
    f"{BASE_URL}/research",
    f"{BASE_URL}/hostel",
    f"{BASE_URL}/library",
    f"{BASE_URL}/medical",
    f"{BASE_URL}/sports",
    f"{BASE_URL}/placements",
    f"{BASE_URL}/placement",
    f"{BASE_URL}/training-and-placement",
    f"{BASE_URL}/notices",
    f"{BASE_URL}/news",
    f"{BASE_URL}/events",
    f"{BASE_URL}/alumni",
    f"{BASE_URL}/nirf",
    f"{BASE_URL}/hpc",
    f"{BASE_URL}/academic-calendar",
    f"{BASE_URL}/facilities",
    f"{BASE_URL}/mcm-scholarship",
    f"{BASE_URL}/latest",
    f"{BASE_URL}/announcements",
    f"{BASE_URL}/announcement",
    f"{BASE_URL}/notice",
    f"{BASE_URL}/tenders",
    f"{BASE_URL}/careers",
    f"{BASE_URL}/faculty",
    f"{BASE_URL}/people",
    f"{BASE_URL}/research-projects",
    f"{BASE_URL}/sponsored-research",
    f"{BASE_URL}/contract-project-staff",
    f"{BASE_URL}/project-staff",
]

# Higher score = crawled earlier. These sections change often or are critical
# for the chatbot's official-site answers.
PRIORITY_URL_PATTERNS = [
    (re.compile(r"(notice|notices|announcement|announcements|latest|news|event|events)", re.I), 100),
    (re.compile(r"(admission|admissions|jee|gate|josaa|phd|mtech|btech|msc)", re.I), 95),
    (re.compile(r"(faculty|people|profile|professor|hod|dean)", re.I), 90),
    (re.compile(r"(project|research|sponsored|consultancy|jrf|srf|recruitment|career|job)", re.I), 85),
    (re.compile(r"(tender|circular|office-order|academic-calendar|fee)", re.I), 80),
    (re.compile(r"\.pdf(\?.*)?$", re.I), 75),
    (re.compile(r"(department|computer|electrical|mechanical|civil|chemical|physics|chemistry|hss|bioscience|materials)", re.I), 70),
]

# ── Skip patterns ─────────────────────────────────────────────────
SKIP_DOMAINS = {
    "saral.iitjammu.ac.in", "facebook.com", "twitter.com",
    "youtube.com", "linkedin.com", "instagram.com",
    "google.com", "googleapis.com",
}

SKIP_EXTENSIONS = re.compile(
    r"\.(jpg|jpeg|png|gif|svg|ico|webp|bmp|woff|woff2|ttf|eot|otf"
    r"|mp4|avi|mov|mp3|wav|zip|rar|tar|gz|doc|docx|xls|xlsx|ppt|pptx"
    r"|exe|dmg|apk)(\?.*)?$",
    re.IGNORECASE,
)


def _resolve_state_path() -> str:
    """Resolve state file path."""
    path = STATE_FILE
    if path.startswith("../"):
        path = path[3:]

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    resolved = os.path.join(project_root, path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    return resolved


def _load_state() -> Dict:
    """Load scraper state from disk."""
    path = _resolve_state_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "visited_urls": [],
        "pending_urls": [],
        "content_hashes": [],
        "url_content_hashes": {},
        "last_run": None,
        "total_pages_scraped": 0,
        "total_documents_added": 0,
    }


def _save_state(state: Dict):
    """Save scraper state to disk."""
    path = _resolve_state_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save scraper state: {e}")


# Noise to skip completely (e.g. notices, circulars, advertisements, tenders, job career info, archives)
SKIP_NOISE_PATTERNS = re.compile(
    r"(tender|tenders|circular|circulars|advertisement|advertisements|advt|advert"
    r"|recruitment|vacancy|vacancies|career|careers|job-opportunities"
    r"|office-order|office-orders|office_order|meeting-minutes|minutes-of"
    r"|newsletter|newsletters|press-release|press_release|archive|archives"
    r"|obituary|condolence|/notices/|/news/|/events/|/tenders/|/recruitment/|/opportunities/)",
    re.IGNORECASE,
)

# Pattern to identify faculty and department pages (which we do NOT filter by year)
IS_FACULTY_DEPT = re.compile(
    r"/(faculty|people|professor|staff|departments|computer_science_engineering|electrical_engineering|mechanical_engineering|civil_engineering|chemical-engineering|mathematics|physics|chemistry|hss|materials_engineering|biosciences_bioengineering|idp|cds)\b",
    re.I
)

# Standalone 4-digit years for outdated info (2019-2025)
YEAR_PATTERN = re.compile(r"\b(2019|2020|2021|2022|2023|2024|2025)\b")

def _should_skip_url(url: str) -> bool:
    """Check if URL should be skipped."""
    from urllib.parse import urlparse

    if not url or not url.startswith(("http://", "https://")):
        return True
    
    # Check general skip patterns
    if any(p in url for p in ["javascript:", "mailto:", "tel:", "#"]):
        return True

    if SKIP_EXTENSIONS.search(url):
        return True

    if SKIP_NOISE_PATTERNS.search(url):
        return True

    parsed = urlparse(url)
    if parsed.netloc in SKIP_DOMAINS:
        return True
    
    # Allow iitjammu.ac.in and www.iitjammu.ac.in
    netloc = parsed.netloc.lower()
    if netloc not in ("iitjammu.ac.in", "www.iitjammu.ac.in") and "iitjammu" not in netloc:
        return True

    # Check for outdated year in URL path (unless it is a faculty/department page)
    url_lower = url.lower()
    if not IS_FACULTY_DEPT.search(url_lower):
        if YEAR_PATTERN.search(url_lower):
            # Allow current/upcoming academic years
            if "2025-26" in url_lower or "2025_26" in url_lower or "2025-2026" in url_lower or "2026" in url_lower:
                return False
            return True

    return False


def _is_pdf_url(url: str) -> bool:
    return bool(re.search(r"\.pdf(\?.*)?$", url, re.IGNORECASE))


def _url_priority(url: str) -> int:
    """Score a URL so fresh/high-value IIT Jammu pages are crawled first."""
    score = 10
    for pattern, boost in PRIORITY_URL_PATTERNS:
        if pattern.search(url):
            score = max(score, boost)
    return score


def _is_priority_url(url: str) -> bool:
    return _url_priority(url) >= 70


def _normalize_url(url: str) -> str:
    """Normalize URL enough to avoid duplicate frontier entries."""
    return url.split("#")[0].rstrip("/")


def _enqueue(queue: List[str], queued: Set[str], url: str):
    """Add a URL to the frontier if it is crawlable and not already queued."""
    url = _normalize_url(url)
    if _should_skip_url(url) or url in queued:
        return
    queued.add(url)
    queue.append(url)


def _sort_frontier(queue: List[str]) -> List[str]:
    """Priority-first frontier, stable inside each priority bucket."""
    return sorted(queue, key=lambda u: (-_url_priority(u), u))


def _extract_page_content(html: str, url: str) -> Dict[str, str]:
    """Extract clean text from HTML page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup.find_all(["script", "style", "noscript", "iframe", "nav"]):
        tag.decompose()

    # Remove noisy classes
    noise_re = re.compile(
        r"(navbar|footer|sidebar|breadcrumb|social|cookie|banner|popup|modal)",
        re.I
    )
    for tag in soup.find_all(class_=noise_re):
        tag.decompose()

    # Extract title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
        title = re.sub(r"\s*[\|\\-]\s*IIT Jammu.*$", "", title, flags=re.I).strip()
    if not title:
        title = "IIT Jammu Page"

    # Find main content
    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.find(class_=re.compile(r"(content|main|body|wrapper)", re.I))
        or soup.body
    )

    if main:
        text = main.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    # Clean text
    text = re.sub(r"\s+", " ", text).strip()

    return {"title": title, "text": text, "source_url": url}


def _extract_pdf_content(pdf_bytes: bytes, url: str) -> Optional[Dict[str, str]]:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()

        text = " ".join(pages_text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 50:
            return None

        title = os.path.basename(url).replace(".pdf", "").replace("_", " ").replace("-", " ")

        return {"title": title, "text": text, "source_url": url}

    except ImportError:
        logger.warning("PyMuPDF not installed — skipping PDF extraction")
        return None
    except Exception as e:
        logger.warning(f"PDF extraction failed for {url}: {e}")
        return None


def _chunk_text(
    text: str,
    title: str,
    source_url: str,
    topic: str = "Scraped",
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """Split text into chunks for ChromaDB."""
    if len(text) <= chunk_size:
        return [{
            "text": text,
            "title": title,
            "source_url": source_url,
            "topic": topic,
        }]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk and len(chunk) >= 50:
            idx += 1
            chunks.append({
                "text": chunk,
                "title": f"{title} (part {idx})",
                "source_url": source_url,
                "topic": topic,
            })
        start = end - overlap

    return chunks


def _extract_links(html: str, base_url: str) -> List[str]:
    """Extract all valid internal links from HTML."""
    from urllib.parse import urljoin, urlparse
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue

        abs_url = urljoin(base_url, href)
        # Remove fragments
        abs_url = _normalize_url(abs_url)

        if not _should_skip_url(abs_url):
            links.append(abs_url)

    return list(set(links))


def _extract_sitemap_links(xml_text: str) -> List[str]:
    """Extract URLs from a sitemap XML document."""
    links = re.findall(r"<loc>\s*([^<]+?)\s*</loc>", xml_text, flags=re.I)
    clean = []
    for link in links:
        link = _normalize_url(link.strip())
        if not _should_skip_url(link):
            clean.append(link)
    return list(set(clean))


def run_scrape_cycle(chroma_store=None, knowledge_graph=None):
    """
    Run one full scrape cycle.
    Crawls seed URLs + discovered links, adds content to ChromaDB + KG.

    This is designed to be called by APScheduler or manually.
    """
    import requests

    if not SCRAPER_ENABLED:
        logger.info("Background scraper is disabled")
        return

    logger.info("🕷️  Background scraper starting...")
    state = _load_state()
    visited = set(state.get("visited_urls", []))
    pending = set(state.get("pending_urls", []))
    content_hashes = set(state.get("content_hashes", []))
    url_content_hashes = state.get("url_content_hashes", {})

    # Build a priority frontier. Priority seed pages are revisited every cycle;
    # non-priority pages resume from pending/discovered URLs.
    queue: List[str] = []
    queued: Set[str] = set()
    for url in SEED_URLS:
        if _is_priority_url(url) or url not in visited:
            _enqueue(queue, queued, url)
    for url in pending:
        _enqueue(queue, queued, url)
    queue = _sort_frontier(queue)
    pages_scraped = 0
    docs_added = 0
    attempted = 0
    skipped = 0

    session = requests.Session()
    session.headers.update({
        "User-Agent": "IITJammuChatbot/2.0 (Educational Research Bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    })

    while queue and pages_scraped < SCRAPER_MAX_PAGES:
        url = queue.pop(0)

        if url in visited and not _is_priority_url(url):
            skipped += 1
            continue
        if _should_skip_url(url):
            skipped += 1
            continue

        pending.discard(url)
        attempted += 1
        logger.info(
            f"  [ingested {pages_scraped}/{SCRAPER_MAX_PAGES} | attempted {attempted}] "
            f"Scraping: {url}"
        )

        try:
            resp = session.get(url, timeout=15, allow_redirects=True)

            if resp.status_code != 200:
                skipped += 1
                logger.debug(f"  Skip: HTTP {resp.status_code}")
                continue

            visited.add(url)
            content_type = resp.headers.get("content-type", "")

            # Handle PDFs
            if _is_pdf_url(url) or "application/pdf" in content_type:
                page_data = _extract_pdf_content(resp.content, url)
                if not page_data:
                    skipped += 1
                    continue
            elif "xml" in content_type or url.lower().endswith(".xml"):
                new_links = _extract_sitemap_links(resp.text)
                for link in new_links:
                    if _is_priority_url(link) or link not in visited:
                        _enqueue(queue, queued, link)
                queue = _sort_frontier(queue)
                logger.info(f"  Sitemap discovered {len(new_links)} URLs: {url}")
                skipped += 1
                continue
            elif "text/html" in content_type:
                page_data = _extract_page_content(resp.text, url)

                # Extract new links to crawl
                new_links = _extract_links(resp.text, url)
                for link in new_links:
                    if _is_priority_url(link) or link not in visited:
                        _enqueue(queue, queued, link)
                queue = _sort_frontier(queue)
            else:
                skipped += 1
                continue

            # Check content length
            text = page_data.get("text", "")
            if len(text) < 100:
                skipped += 1
                continue

            # Dedup by content hash
            content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            previous_hash = url_content_hashes.get(url)
            if content_hash in content_hashes and previous_hash == content_hash:
                skipped += 1
                logger.debug(f"  Duplicate content, skipping")
                continue
            content_hashes.add(content_hash)
            url_content_hashes[url] = content_hash

            # Chunk and add to ChromaDB
            if chroma_store:
                try:
                    chunks = _chunk_text(
                        text=text,
                        title=page_data.get("title", ""),
                        source_url=url,
                    )
                    added = chroma_store.add_documents(chunks)
                    docs_added += added
                except Exception as e:
                    logger.warning(f"  ChromaDB insert error: {e}")

            # Extract entities and add to Knowledge Graph
            if knowledge_graph:
                try:
                    knowledge_graph.extract_and_add_from_text(
                        text=text,
                        title=page_data.get("title", ""),
                        source_url=url,
                    )
                except Exception as e:
                    logger.warning(f"  KG extraction error: {e}")

            pages_scraped += 1
            logger.info(f"  ✓ Scraped: {page_data.get('title', '')[:60]}")

        except requests.exceptions.Timeout:
            skipped += 1
            logger.debug(f"  Timeout: {url}")
        except Exception as e:
            skipped += 1
            logger.debug(f"  Error: {type(e).__name__}: {str(e)[:80]}")
        finally:
            # Rate-limit every network attempt, including skipped/duplicate pages.
            time.sleep(SCRAPER_DELAY)

    # Preserve remaining crawl frontier for the next cycle.
    pending.update(queue)

    # Save KG to disk
    if knowledge_graph:
        try:
            knowledge_graph.save()
        except Exception as e:
            logger.warning(f"Failed to save KG after scrape: {e}")

    # Update state
    state["visited_urls"] = sorted(visited)
    state["pending_urls"] = sorted(pending, key=lambda u: (-_url_priority(u), u))
    state["content_hashes"] = sorted(content_hashes)
    state["url_content_hashes"] = url_content_hashes
    state["last_run"] = datetime.utcnow().isoformat()
    state["last_cycle"] = {
        "attempted_urls": attempted,
        "scraped_pages": pages_scraped,
        "skipped_urls": skipped,
        "documents_added": docs_added,
        "stop_reason": "max_pages_reached" if pages_scraped >= SCRAPER_MAX_PAGES else "frontier_exhausted",
        "pending_urls": len(pending),
    }
    state["total_pages_scraped"] = state.get("total_pages_scraped", 0) + pages_scraped
    state["total_documents_added"] = state.get("total_documents_added", 0) + docs_added
    _save_state(state)

    logger.info(
        f"🕷️  Scrape cycle complete: {pages_scraped} pages scraped, "
        f"{attempted} URLs attempted, {skipped} skipped, "
        f"{docs_added} new documents added to ChromaDB, pending={len(pending)}"
    )


def get_scraper_status() -> Dict:
    """Return the current scraper state for the status endpoint."""
    state = _load_state()
    return {
        "enabled": SCRAPER_ENABLED,
        "interval_hours": SCRAPER_INTERVAL_HOURS,
        "run_on_startup": SCRAPER_RUN_ON_STARTUP,
        "last_run": state.get("last_run"),
        "total_pages_scraped": state.get("total_pages_scraped", 0),
        "total_documents_added": state.get("total_documents_added", 0),
        "visited_urls_count": len(state.get("visited_urls", [])),
        "pending_urls_count": len(state.get("pending_urls", [])),
        "priority_pending_count": sum(
            1 for url in state.get("pending_urls", []) if _is_priority_url(url)
        ),
        "last_cycle": state.get("last_cycle", {}),
    }


# ── APScheduler integration ───────────────────────────────────────

_scheduler = None


def start_background_scraper(chroma_store=None, knowledge_graph=None):
    """Start the APScheduler background scraper."""
    global _scheduler

    if not SCRAPER_ENABLED:
        logger.info("Background scraper is disabled (SCRAPER_ENABLED=false)")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        logger.warning("APScheduler not installed. Run: pip install apscheduler")
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        run_scrape_cycle,
        "interval",
        hours=SCRAPER_INTERVAL_HOURS,
        kwargs={"chroma_store": chroma_store, "knowledge_graph": knowledge_graph},
        id="iitj_scraper",
        name="IIT Jammu Background Scraper",
        max_instances=1,
        replace_existing=True,
    )
    _scheduler.start()
    if SCRAPER_RUN_ON_STARTUP:
        import threading
        thread = threading.Thread(
            target=run_scrape_cycle,
            kwargs={"chroma_store": chroma_store, "knowledge_graph": knowledge_graph},
            daemon=True,
        )
        thread.start()
        logger.info("Background scraper initial priority crawl started")
    logger.info(
        f"✅ Background scraper scheduled (every {SCRAPER_INTERVAL_HOURS}h)"
    )


def stop_background_scraper():
    """Stop the background scraper."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Background scraper stopped")


def trigger_manual_scrape(chroma_store=None, knowledge_graph=None):
    """Manually trigger one scrape cycle (for the API endpoint)."""
    import threading
    thread = threading.Thread(
        target=run_scrape_cycle,
        kwargs={"chroma_store": chroma_store, "knowledge_graph": knowledge_graph},
        daemon=True,
    )
    thread.start()
    return {"status": "scrape_triggered", "message": "Background scrape cycle started"}
