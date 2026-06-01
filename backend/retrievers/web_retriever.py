"""
retrievers/web_retriever.py — DuckDuckGo Web Search Fallback
=============================================================
Falls back to DuckDuckGo when local retrieval is insufficient.
Fetches full page content from iitjammu.ac.in and ingests it
into ChromaDB + KG for future local availability.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from config import get_settings
from utils.text import chunk_text, clean_text

logger = logging.getLogger(__name__)


class WebRetriever:
    """DuckDuckGo web search retriever for IIT Jammu content."""

    def __init__(self, chroma_store=None, knowledge_graph=None):
        self.chroma = chroma_store
        self.kg = knowledge_graph
        self.settings = get_settings()

    def retrieve(self, query: str, max_results: Optional[int] = None) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Search DuckDuckGo for IIT Jammu content.

        Returns: (results_list, was_used_flag)
        """
        if not self.settings.ddg_enabled:
            return [], False

        max_results = max_results or self.settings.ddg_max_results
        
        # --- REDIRECTION LOGIC FOR NOTICES, JOBS, AND AWARDS ---
        q_lower = query.lower()
        job_keywords = ["job", "vacancy", "vacancies", "career", "careers", "recruitment", "employment", "hiring", "opportunity", "opportunities", "opening", "openings", "project associate", "project staff", "research associate", "postdoc", "jrf"]
        is_jobs = any(w in q_lower for w in job_keywords)
        is_news = any(w in q_lower for w in ["notification", "notifications", "circular", "circulars", "news", "update", "updates", "notice", "notices", "tender", "tenders", "announcement", "announcements"])
        is_awards = any(w in q_lower for w in ["award", "awards", "achievement", "achievements", "honor", "honors", "honours", "recognition", "recognitions"])
        
        injected = []
        if is_jobs:
            injected.append({
                "title": "Jobs - IIT Jammu",
                "url": "https://www.iitjammu.ac.in/postlist/Jobs",
                "snippet": "Official IIT Jammu Jobs and Career Opportunities page. Contains advertisements, application deadlines, and notifications for faculty, non-faculty, project staff, and temporary/contract positions."
            })
        if is_news:
            injected.append({
                "title": "News and Update - IIT Jammu",
                "url": "https://www.iitjammu.ac.in/postlist/News%20and%20Update",
                "snippet": "Official IIT Jammu News, Updates, Notices, and Announcements page. Contains academic notifications, semester schedule changes, registration updates, fee payment notifications, and other official circulars."
            })
        if is_awards:
            injected.append({
                "title": "Awards and Achievements - IIT Jammu",
                "url": "https://www.iitjammu.ac.in/postlist/Awards%20and%20Achievements",
                "snippet": "Official IIT Jammu Awards and Achievements page. Highlights honors, awards, research recognitions, hackathon wins, and accomplishments of faculty, scholars, and students."
            })
            
        web_results = self._search_ddg(query, max_results)
        
        # Merge injected pages into web_results if not already present
        merged_results = []
        seen_urls = set()
        
        # Add injected first to prioritize them
        for item in injected:
            if item["url"] not in seen_urls:
                merged_results.append(item)
                seen_urls.add(item["url"])
                
        for item in web_results:
            if item["url"] not in seen_urls:
                merged_results.append(item)
                seen_urls.add(item["url"])
                
        if not merged_results:
            return [], False

        fetched = self._fetch_and_ingest(merged_results)
        results = []
        for page in fetched:
            results.append({
                "title": page.get("title", "Web Result"),
                "text": page.get("text", "")[:1500],
                "source_url": page.get("source_url", ""),
                "source_type": f"Web Search ({page.get('source_url', 'iitjammu.ac.in')})",
                "score": 0.4,  # Base score for web results
                "similarity": 0.4,
                "year": None,
                "department": "General",
                "doc_type": "Web",
                "crawl_date": "",
            })

        return results, bool(results)

    def _search_ddg(self, query: str, max_results: int) -> list:
        """Execute DuckDuckGo search."""
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("DuckDuckGo search not installed. Run: pip install duckduckgo-search")
            return []

        # Optimize search query based on category keyword matching
        q_lower = query.lower()
        search_terms = []
        if any(w in q_lower for w in ["job", "vacancy", "vacancies", "career", "careers", "recruitment", "employment", "hiring", "opportunity", "opportunities", "opening", "openings"]):
            search_terms.append('"Jobs" OR "vacancy"')
        if any(w in q_lower for w in ["notification", "notifications", "circular", "circulars", "news", "update", "updates", "notice", "notices", "tender", "tenders", "announcement", "announcements"]):
            search_terms.append('"News and Update" OR "circular" OR "notice"')
        if any(w in q_lower for w in ["award", "awards", "achievement", "achievements", "honor", "honors", "honours", "recognition", "recognitions"]):
            search_terms.append('"Awards and Achievements"')
            
        term_prefix = " OR ".join(search_terms)
        if term_prefix:
            search_query = f"site:iitjammu.ac.in ({term_prefix}) {query}"
        else:
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
                results.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("body", r.get("snippet", "")),
                })

            logger.info(f"DuckDuckGo: found {len(results)} results for '{query[:50]}'")

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {type(e).__name__}: {e}")

        return results

    def _fetch_and_ingest(self, results: list) -> List[Dict[str, str]]:
        """Fetch page content and ingest into ChromaDB + KG."""
        import requests
        from bs4 import BeautifulSoup

        fetched_pages = []

        for result in results:
            url = result.get("url", "")
            if not url:
                continue

            try:
                resp = requests.get(
                    url,
                    timeout=self.settings.ddg_timeout,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36 "
                            "IITJammuChatbot/3.0"
                        )
                    }
                )
                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup.find_all(["script", "style", "noscript", "nav", "footer"]):
                    tag.decompose()

                main = (
                    soup.find("main")
                    or soup.find(attrs={"role": "main"})
                    or soup.find("article")
                    or soup.body
                )

                text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
                
                # --- DEEP FETCH FOR FACULTY PROFILES ---
                # If the page links to a personal Google Site or Scholar, fetch it too!
                if "faculty" in url:
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        if "sites.google.com" in href or "scholar.google" in href:
                            try:
                                ext_resp = requests.get(href, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                                ext_soup = BeautifulSoup(ext_resp.text, "html.parser")
                                ext_text = ext_soup.get_text(" ", strip=True)
                                text += f"\n\n--- External Profile Data ({href}) ---\n" + ext_text[:3000]
                                break  # Just get the first major profile to avoid timeouts
                            except Exception as e:
                                logger.warning(f"Failed to deep-fetch external profile {href}: {e}")

                text = clean_text(text)

                if len(text) < 150:
                    if result.get("snippet"):
                        text = result["snippet"]
                    else:
                        continue
                if len(text) > 5000:
                    text = text[:5000]

                page_data = {
                    "text": text,
                    "title": result.get("title", "IIT Jammu Page"),
                    "source_url": url,
                    "topic": "Web Search Result",
                }
                fetched_pages.append(page_data)

                # NOTE: Web results are intentionally NOT written to ChromaDB or KG.
                # The database contains only curated, verified IIT Jammu data.
                # DDG results are used only for answering the current query in-memory.

            except Exception as e:
                logger.debug(f"Failed to fetch {url}: {e}")
                if result.get("snippet"):
                    fetched_pages.append({
                        "text": result["snippet"],
                        "title": result.get("title", ""),
                        "source_url": url,
                        "topic": "Web Search Snippet",
                    })

        return fetched_pages
