"""
ddg_fallback.py — DuckDuckGo Site-Search Fallback for IIT Jammu
================================================================
When ChromaDB + KG don't have enough relevant results,
search DuckDuckGo restricted to iitjammu.ac.in and return results.

Also ingests fetched page content into ChromaDB + KG so the data
is available locally for future queries.
"""

import re
import logging
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DDG_ENABLED = os.getenv("DDG_ENABLED", "true").lower() == "true"
DDG_MAX_RESULTS = int(os.getenv("DDG_MAX_RESULTS", "5"))
DDG_TIMEOUT = int(os.getenv("DDG_TIMEOUT", "10"))


@dataclass
class DDGResult:
    """A single DuckDuckGo search result."""
    title: str
    url: str
    snippet: str


def search_iitj(query: str, max_results: int = DDG_MAX_RESULTS) -> List[DDGResult]:
    """
    Search DuckDuckGo for IIT Jammu-specific results.
    Uses site:iitjammu.ac.in to restrict to official site.

    Returns list of DDGResult with title, url, snippet.
    """
    if not DDG_ENABLED:
        logger.debug("DuckDuckGo fallback is disabled")
        return []

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("DuckDuckGo search package not installed. Run: pip install duckduckgo-search")
        return []

    search_query = f"site:iitjammu.ac.in {query}"
    results = []

    try:
        with DDGS() as ddgs:
            try:
                ddg_results = list(ddgs.text(
                    search_query,
                    max_results=max_results,
                    region="in-en",
                ))
            except TypeError:
                ddg_results = list(ddgs.text(
                    search_query,
                    max_results=max_results,
                ))

        for r in ddg_results:
            url = r.get("href", r.get("link", ""))
            if not url or "iitjammu.ac.in" not in url:
                continue
            results.append(DDGResult(
                title=r.get("title", ""),
                url=url,
                snippet=r.get("body", r.get("snippet", "")),
            ))

        logger.info(f"DuckDuckGo: found {len(results)} results for '{query[:50]}'")

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {type(e).__name__}: {e}")

    return results


def fetch_and_ingest_ddg_results(
    results: List[DDGResult],
    chroma_store=None,
    knowledge_graph=None,
) -> List[Dict[str, str]]:
    """
    Fetch actual page content from DDG result URLs and ingest into
    ChromaDB + Knowledge Graph for future local availability.

    Returns list of fetched page content dicts.
    """
    import requests
    from bs4 import BeautifulSoup

    fetched_pages = []

    for result in results:
        if not result.url:
            continue

        try:
            resp = requests.get(
                result.url,
                timeout=DDG_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36 IITJammuChatbot/2.0"
                    )
                }
            )
            resp.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noise
            for tag in soup.find_all(["script", "style", "noscript", "nav", "footer"]):
                tag.decompose()

            # Extract text
            main_content = (
                soup.find("main")
                or soup.find(attrs={"role": "main"})
                or soup.find("article")
                or soup.body
            )

            if main_content:
                text = main_content.get_text(" ", strip=True)
            else:
                text = soup.get_text(" ", strip=True)

            # Clean text
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) < 100:
                continue

            # Truncate very long pages
            if len(text) > 5000:
                text = text[:5000]

            page_data = {
                "text": text,
                "title": result.title or "IIT Jammu Page",
                "source_url": result.url,
                "topic": "Web Search Result",
            }
            fetched_pages.append(page_data)

            # Ingest into ChromaDB
            if chroma_store:
                try:
                    chunks = _chunk_text(text, result.title, result.url)
                    chroma_store.add_documents(chunks)
                except Exception as e:
                    logger.warning(f"Failed to ingest DDG result into ChromaDB: {e}")

            # Ingest into Knowledge Graph
            if knowledge_graph:
                try:
                    knowledge_graph.extract_and_add_from_text(
                        text, title=result.title, source_url=result.url
                    )
                    knowledge_graph.save()
                except Exception as e:
                    logger.warning(f"Failed to ingest DDG result into KG: {e}")

        except Exception as e:
            logger.debug(f"Failed to fetch DDG result {result.url}: {e}")
            # Still use the snippet as context
            if result.snippet:
                fetched_pages.append({
                    "text": result.snippet,
                    "title": result.title,
                    "source_url": result.url,
                    "topic": "Web Search Snippet",
                })

    return fetched_pages


def _chunk_text(
    text: str,
    title: str = "",
    source_url: str = "",
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """Split text into overlapping chunks for ChromaDB storage."""
    chunks = []
    words = text.split()

    if len(words) <= chunk_size // 4:
        # Short text — single chunk
        return [{
            "text": text,
            "title": title,
            "source_url": source_url,
            "topic": "Web Search Result",
        }]

    # Character-based chunking with overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "title": f"{title} (chunk {len(chunks) + 1})",
                "source_url": source_url,
                "topic": "Web Search Result",
            })

        start = end - overlap

    return chunks


def format_ddg_context(results: List[DDGResult], fetched_pages: List[Dict] = None) -> str:
    """
    Format DuckDuckGo results as a context string for the LLM.
    Prefers fetched full-page content over snippets.
    """
    parts = []

    if fetched_pages:
        for page in fetched_pages:
            parts.append(
                f"### {page.get('title', 'Web Result')}\n"
                f"Source: {page.get('source_url', '')}\n"
                f"{page['text'][:1500]}\n"
            )
    else:
        for r in results:
            parts.append(
                f"### {r.title}\n"
                f"Source: {r.url}\n"
                f"{r.snippet}\n"
            )

    if parts:
        return (
            "\n--- WEB SEARCH RESULTS (from iitjammu.ac.in) ---\n"
            + "\n---\n".join(parts)
        )
    return ""
