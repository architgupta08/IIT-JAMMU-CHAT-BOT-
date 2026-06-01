"""
rag/engine.py — Hybrid RAG Engine (Production)
================================================
Orchestrates the full retrieval-augmented generation pipeline:

  1. Query processing (NER, abbreviation expansion, intent classification)
  2. Parallel retrieval (ChromaDB + BM25 + Knowledge Graph)
  3. Merge & deduplicate results
  4. Cross-encoder reranking
  5. Recency-weighted scoring
  6. Context compression
  7. Conversation memory injection
  8. DuckDuckGo web fallback (if local results insufficient)
  9. LLM answer generation with anti-hallucination prompt
  10. Source citation formatting

Backward compatible: same RAGResult, FlatNode, get_rag_engine() API.
"""

import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

from config import get_settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  Data classes (backward compatible with old API)
# ══════════════════════════════════════════════════════════════════

@dataclass
class FlatNode:
    node_id: str
    title: str
    path: str
    summary: str
    text: str
    score: float = 0.0


@dataclass
class RAGResult:
    answer: str
    sources: List[FlatNode] = field(default_factory=list)
    confidence: float = 0.0
    detected_language: str = "en"


# ══════════════════════════════════════════════════════════════════
#  Legacy Knowledge Tree (kept for backward compat + stats)
# ══════════════════════════════════════════════════════════════════

class IITJKnowledgeTree:
    """Loads the PageIndex JSON tree. Kept for stats and fallback."""

    def __init__(self, index_path: str):
        self._path = index_path
        self._tree: Dict[str, Any] = {}
        self._flat: List[FlatNode] = []
        self._load()

    def _load(self):
        p = Path(self._resolve(self._path))
        if not p.exists():
            logger.warning(f"Index not found: {p.resolve()} — using empty tree")
            self._tree = {"structure": []}
            return
        self._tree = json.loads(p.read_text(encoding="utf-8"))
        self._flat = []
        self._flatten(self._tree.get("structure", []), parent="")
        logger.info(f"Legacy tree loaded: {len(self._flat)} nodes from {p}")

    def _resolve(self, path: str) -> str:
        if path.startswith("../"):
            path = path[3:]
        if os.path.exists(path):
            return path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(backend_dir)
        from_parent = os.path.join(parent_dir, path)
        if os.path.exists(from_parent):
            return from_parent
        project_root = os.path.dirname(parent_dir)
        from_root = os.path.join(project_root, path)
        if os.path.exists(from_root):
            return from_root
        return path

    def _flatten(self, nodes: list, parent: str):
        for n in nodes:
            title = n.get("title", "Untitled")
            path = f"{parent} > {title}" if parent else title
            self._flat.append(FlatNode(
                node_id=n.get("node_id", ""),
                title=title,
                path=path,
                summary=n.get("summary", ""),
                text=n.get("text", ""),
            ))
            self._flatten(n.get("nodes", []), path)

    def count_nodes(self) -> int:
        return len(self._flat)

    def get_root_nodes(self) -> list:
        return self._tree.get("structure", [])

    def get_top_level_titles(self) -> list:
        return [n.get("title", "") for n in self.get_root_nodes()]

    def get_last_updated(self) -> Optional[str]:
        return self._tree.get("last_updated")


# ══════════════════════════════════════════════════════════════════
#  Hybrid RAG Engine
# ══════════════════════════════════════════════════════════════════

class HybridRAGEngine:
    """
    Production-ready Hybrid RAG Engine.

    Retrieval: ChromaDB + BM25 + KG + DDG fallback
    Reranking: Cross-encoder
    Generation: Ollama LLM with anti-hallucination prompt
    Memory: Conversation context for follow-ups
    """

    def __init__(
        self,
        tree,
        llm_client,
        chroma_store=None,
        knowledge_graph=None,
    ):
        self.tree = tree
        self.llm = llm_client
        self.settings = get_settings()

        # Initialize retrievers
        from retrievers.chroma_retriever import ChromaRetriever
        from retrievers.bm25_retriever import BM25Retriever
        from retrievers.kg_retriever import KGRetriever
        from retrievers.web_retriever import WebRetriever
        from reranker.cross_encoder import get_reranker
        from rag.context_builder import get_context_builder
        from rag.query_processor import QueryProcessor
        from memory.conversation import get_memory
        from retrievers.tabular_retriever import TabularRetriever

        self.chroma_retriever = ChromaRetriever(chroma_store)
        self.bm25_retriever = BM25Retriever(chroma_store)
        self.kg_retriever = KGRetriever(knowledge_graph)
        self.web_retriever = WebRetriever(chroma_store, knowledge_graph)
        self.tabular_retriever = TabularRetriever()
        self.reranker = get_reranker()
        self.context_builder = get_context_builder()
        self.query_processor = QueryProcessor()
        self.memory = get_memory()

        # Build BM25 index in background
        try:
            self.bm25_retriever.build_index()
        except Exception as e:
            logger.warning(f"BM25 index build deferred: {e}")

    # Phrases that indicate the LLM found nothing useful in KB context
    _NO_INFO_PHRASES = [
        "could not find relevant information",
        "i could not find",
        "i don't have information",
        "i do not have information",
        "no specific information",
        "not available in my knowledge",
        "please visit https://www.iitjammu.ac.in",
        "please check the official website",
        "i don't have specific",
        "unable to find",
        "no information found",
        "not found in the knowledge base",
    ]

    def _is_no_info_answer(self, text: str) -> bool:
        """Return True if the LLM answer indicates it could not find a relevant answer."""
        t = text.lower()
        return any(phrase in t for phrase in self._NO_INFO_PHRASES)

    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on text similarity."""
        seen = set()
        deduped = []
        for r in results:
            key = r.get("text", "")[:200].strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    def _results_to_flat_nodes(self, results: List[Dict[str, Any]]) -> List[FlatNode]:
        """Convert result dicts to FlatNode objects for backward-compatible API."""
        nodes = []
        for i, r in enumerate(results[:6]):
            source_url = r.get("source_url", "")
            source_type = r.get("source_type", "")
            doc_type = r.get("doc_type", "")
            
            prefix = "web_search" if doc_type == "Web" else "hybrid"
            
            nodes.append(FlatNode(
                node_id=f"{prefix}_{i:04d}",
                title=r.get("title", ""),
                path=source_url if source_url else source_type,
                summary=r.get("text", "")[:200],
                text=r.get("text", ""),
                score=r.get("score", 0),
            ))
        return nodes

    async def answer(
        self,
        query: str,
        target_language: str = "en",
        session_id: str = "",
    ) -> RAGResult:
        """
        Main entry point. Full hybrid RAG pipeline.

        Steps:
          1. Off-topic guard
          2. Query processing (NER, abbreviation expansion)
          3. Follow-up resolution (conversation memory)
          4. Parallel retrieval (ChromaDB + BM25 + KG)
          5. Merge & deduplicate
          6. Cross-encoder reranking
          7. DuckDuckGo fallback if needed
          8. Context compression
          9. LLM generation with memory context
          10. Store in memory and return
        """
        from rag.guards import is_off_topic, needs_fresh_web_search

        # ── Step 1: Off-topic guard ──────────────────────────────
        if is_off_topic(query):
            lang_map = {
                "hi": "मैं केवल IIT Jammu से संबंधित प्रश्नों का उत्तर दे सकता हूँ। कृपया IIT Jammu के बारे में पूछें।",
            }
            off_msg = lang_map.get(
                target_language,
                "I can only answer questions related to IIT Jammu — admissions, fees, "
                "programs, faculty, research, campus, placements, and other institute-related "
                "topics. Please ask me something about IIT Jammu!"
            )
            return RAGResult(answer=off_msg, sources=[], confidence=0.0, detected_language=target_language)

        # ── Step 2: Query processing ─────────────────────────────
        query_intent = self.query_processor.process(query)
        logger.info(f"Query intent: {query_intent.intent} | entities: {query_intent.entities}")

        # ── Step 3: Follow-up resolution ─────────────────────────
        effective_query = query
        if session_id:
            self.memory.add_message(session_id, "user", query)
            resolved = self.memory.resolve_followup(session_id, query)
            if resolved != query:
                effective_query = resolved
                logger.info(f"Follow-up resolved: '{query}' → '{effective_query}'")
                # Re-process the resolved query to update intent, entities, and corrected/processed queries
                query_intent = self.query_processor.process(effective_query)
                logger.info(f"Resolved query intent: {query_intent.intent} | entities: {query_intent.entities}")

        # Use the typo-corrected query as the effective query for downstream tasks
        effective_query = query_intent.corrected_query

        # ── Step 4: Parallel retrieval ───────────────────────────
        search_query = query_intent.processed_query

        # Early detection of supervisor/scholar list queries to boost retrieval pool
        _supervisor_list_keywords = [
            "list of phd", "phd scholars under", "phd students under",
            "scholars under", "students under", "supervised by",
            "who are the phd", "phd students of", "phd scholars of",
        ]
        _is_supervisor_list_query = any(kw in effective_query.lower() for kw in _supervisor_list_keywords)

        # Use expanded top_k for supervisor list queries so all scholars enter the pool
        retrieval_top_k_chroma = 50 if _is_supervisor_list_query else None  # None uses settings default
        retrieval_top_k_bm25  = 40 if _is_supervisor_list_query else None

        chroma_results = self.chroma_retriever.retrieve(search_query, top_k=retrieval_top_k_chroma)
        logger.info(f"ChromaDB: {len(chroma_results)} results")

        bm25_results = self.bm25_retriever.retrieve(search_query, top_k=retrieval_top_k_bm25)
        logger.info(f"BM25: {len(bm25_results)} results")

        kg_results = self.kg_retriever.retrieve(search_query)
        logger.info(f"KG: {len(kg_results)} results")

        # ── Pre-fetch phd_scholar docs filtered by the faculty name ──
        person_entity = query_intent.entities.get("person", "")
        if _is_supervisor_list_query and person_entity:
            # Search phd_scholar docs using the faculty name as query to pull direct matches
            scholar_prefetch = self.chroma_retriever.retrieve(
                f"phd scholar supervisor {person_entity}",
                top_k=20,
            )
            # Only keep phd_scholar/phd_supervisor_faq docs whose text contains the supervisor name
            name_parts = [p.lower() for p in person_entity.split() if len(p) >= 3]
            filtered_prefetch = [
                r for r in scholar_prefetch
                if r.get("doc_type") in {"phd_scholar", "phd_supervisor_faq"}
                and any(part in r.get("text", "").lower() for part in name_parts)
            ]
            if filtered_prefetch:
                logger.info(f"Pre-fetched {len(filtered_prefetch)} phd_scholar docs for '{person_entity}'")
                chroma_results.extend(filtered_prefetch)

        # ── Smart Pre-fetch for Department Faculty Lists ─────────
        # Detect if query asks for a DEPARTMENT's full faculty list
        _dept_list_keywords = [
            "who are the faculty", "list of faculty", "faculty members in",
            "faculty in", "professors in", "all faculty", "faculty of",
            "members of faculty", "who teaches in", "staff in",
        ]
        _is_dept_list_query = any(kw in effective_query.lower() for kw in _dept_list_keywords)

        if _is_dept_list_query:
            # Force-fetch faculty_list documents matching the query so they don't get buried
            list_results = self.chroma_retriever.retrieve(
                search_query,
                top_k=3,
                where_filter={"doc_type": "faculty_list"}
            )
            if list_results:
                logger.info(f"Pre-fetched {len(list_results)} faculty_list documents")
                chroma_results.extend(list_results)

        # ── Step 5: Merge, filter old documents & deduplicate ──
        all_results = chroma_results + bm25_results + kg_results
        all_results = self._deduplicate(all_results)

        # ── Tabular Data Retrieval ────────────────────────────────
        tabular_context = self.tabular_retriever.retrieve_context(effective_query)
        tabular_chunk = None
        if tabular_context:
            logger.info("Tabular retriever matched query — preparing structured context chunk")
            # Determine title: faculty directory vs structured stats tables
            _is_faculty_tabular = "FACULTY DIRECTORY" in tabular_context.upper()
            tabular_title = (
                "Official IIT Jammu Structured Faculty Directory"
                if _is_faculty_tabular
                else "Official IIT Jammu Structured Data (Placements, Fees, & Cutoffs)"
            )
            tabular_chunk = {
                "title": tabular_title,
                "text": tabular_context,
                "source_url": "https://www.iitjammu.ac.in",
                "source_type": "Structured Schema",
                "doc_type": "faq_factsheet",
                "score": 100.0,
                "similarity": 1.0,
                "year": 2026,
            }
            # Remove general/placeholder curated FAQs if we have official structured tables
            filtered_results = []
            for r in all_results:
                text_lower = r.get("text", "").lower()
                if "safest answer" in text_lower or "instead of relying on old" in text_lower:
                    logger.info(f"Filtering out placeholder/avoidance FAQ: {r.get('title')}")
                    continue
                filtered_results.append(r)
            all_results = filtered_results

        # ── KG Factual HOD Traversal ──────────────────────────────
        kg_hod_chunk = None
        _is_hod_query = any(kw in effective_query.lower() for kw in ["hod", "head of department", "head of the department"])
        dept_entity = query_intent.entities.get("department")
        source_url = "https://www.iitjammu.ac.in"
        
        if _is_hod_query and dept_entity and self.kg_retriever.kg:
            try:
                kg_instance = self.kg_retriever.kg
                dept_node_id = kg_instance._fuzzy_find(dept_entity)
                if dept_node_id:
                    target_hod = None
                    hod_attrs = {}
                    for _, target, edge_data in kg_instance._graph.out_edges(dept_node_id, data=True):
                        if edge_data.get("relationship") == "LED_BY":
                            target_hod = target
                            hod_attrs = dict(kg_instance._graph.nodes.get(target, {}))
                            break
                    
                    if target_hod:
                        logger.info(f"KG Traversal matched HOD: {target_hod} for department {dept_entity}")
                        email_str = hod_attrs.get("email", "")
                        source_url = hod_attrs.get("source_url", source_url)
                        
                        kg_hod_context = (
                            f"Official IIT Jammu Department Head Information:\n"
                            f"Department: {dept_entity}\n"
                            f"Head of Department (HoD): {target_hod}\n"
                        )
                        if email_str:
                            kg_hod_context += f"Contact Email: {email_str}\n"
                        if source_url:
                            kg_hod_context += f"Source: {source_url}\n"
                            
                        kg_hod_chunk = {
                            "title": f"Official IIT Jammu KG Fact: {dept_entity} HOD",
                            "text": kg_hod_context,
                            "source_url": source_url,
                            "source_type": "Knowledge Graph Fact",
                            "doc_type": "kg_fact",
                            "score": 100.0,
                            "similarity": 1.0,
                            "year": 2026,
                        }
            except Exception as e:
                logger.error(f"Error traversing KG for HOD query: {e}")

        # Filter out old files (2021, 2022, 2023) unless explicitly asked for
        filtered_results = []
        old_years = ["2021", "2022", "2023", "2020"]
        query_has_old_year = any(y in effective_query for y in old_years)

        for r in all_results:
            title = r.get("title", "")
            text = r.get("text", "")
            doc_type = r.get("doc_type", "") or r.get("metadata", {}).get("doc_type", "")
            
            # Curated FAQs, official factsheets, and faculty profiles/info should never be filtered out
            if doc_type in {"curated_faq", "faq", "faq_factsheet", "kg_fact"} or doc_type.startswith("faculty") or doc_type == "department_info":
                filtered_results.append(r)
                continue

            # Filter out old documents if their title or text mentions old years, unless they also mention a current year
            is_old_doc = any(y in title or y in text for y in old_years)
            has_current_year = any(y in title or y in text for y in ["2024", "2025", "2026"])
            if is_old_doc and not has_current_year and not query_has_old_year:
                logger.info(f"Filtered out old document: {title}")
                continue
            filtered_results.append(r)

        all_results = filtered_results

        # ── Step 6: Cross-encoder reranking ──────────────────────
        if len(all_results) > 2:
            all_results = self.reranker.rerank(
                effective_query,
                all_results,
                top_k=self.settings.reranker_top_k * 2,  # Keep more for diversity
            )
            logger.info(f"After reranking: {len(all_results)} results")
        else:
            # If not reranked, initialize score to 0.0
            for r in all_results:
                r["score"] = r.get("score", 0.0)

        # ── Step 6.5: Apply Custom Score Boosting after Reranking ─
        # Detect if query asks for a DEPARTMENT's full faculty list
        _dept_list_keywords = [
            "who are the faculty", "list of faculty", "faculty members in",
            "faculty in", "professors in", "all faculty", "faculty of",
            "members of faculty", "who teaches in", "staff in",
        ]
        _is_dept_list_query = any(kw in effective_query.lower() for kw in _dept_list_keywords)

        # Detect if query is about a specific person/faculty
        _faculty_query_keywords = [
            "who is", "professor", "prof.", "dr.", "faculty", "research interest",
            "research area", "publication", "email", "contact", "department of",
            "works on", "working on", "teaches", "specializ",
        ]
        _is_faculty_query = any(kw in effective_query.lower() for kw in _faculty_query_keywords) or bool(query_intent.entities.get("person"))

        # Detect if query is about a HOD
        _is_hod_query = any(kw in effective_query.lower() for kw in ["hod", "head of department", "head of the department"])

        # Detect if query is about office/room location
        _is_office_query = any(kw in effective_query.lower() for kw in ["office", "room", "location", "sit", "sits", "cabin", "floor", "where"])

        # Detect if query is about a supervisor/scholar
        _is_supervisor_query = any(kw in effective_query.lower() for kw in ["supervisor", "supervising", "advisor", "scholar", "phd student", "phd scholar"])

        for r in all_results:
            title = r.get("title", "")
            source_type = r.get("source_type", "")
            doc_type = r.get("doc_type", "") or r.get("metadata", {}).get("doc_type", "")

            # Massive boost for custom FAQs, curated IIT Jammu FAQ seeds, and Structured schemas.
            if "FAQ" in source_type.upper() or doc_type in {"curated_faq", "faq", "faq_factsheet", "kg_fact", "phd_supervisor_faq"}:
                r["score"] = r.get("score", 0.0) + (10.0 if source_type == "Structured Schema" or doc_type == "kg_fact" else 2.0)

            # PhD Scholar & Supervisor Boost
            if _is_supervisor_query and doc_type in {"phd_supervisor_faq", "phd_scholar"}:
                person_name = query_intent.entities.get("person", "")
                if person_name:
                    name_parts = [p.lower() for p in person_name.split() if len(p) >= 2]
                    title_lower = title.lower()
                    text_lower = r.get("text", "").lower()
                    if any(part in title_lower or part in text_lower for part in name_parts):
                        r["score"] = r.get("score", 0.0) + 8.0
                        logger.info(f"PhD Scholar/Supervisor Match boost applied (+8.0): {title[:60]}")
                    else:
                        r["score"] = r.get("score", 0.0) + 3.0
                else:
                    r["score"] = r.get("score", 0.0) + 3.0

            # If query is about a person's office/room/location, boost curated FAQs matching that person's location info
            person_name = query_intent.entities.get("person", "")
            if person_name and _is_office_query and doc_type == "curated_faq":
                name_parts = [p.lower() for p in person_name.split() if len(p) > 2]
                title_lower = title.lower()
                text_lower = r.get("text", "").lower()
                if any(part in title_lower or part in text_lower for part in name_parts):
                    location_keywords = ["office", "room", "location", "floor", "cabin", "11ac"]
                    if any(kw in title_lower or kw in text_lower for kw in location_keywords):
                        r["score"] = r.get("score", 0.0) + 6.0
                        logger.info(f"Curated FAQ Office Location boost applied (+6.0): {title[:60]}")

            # HOD BOOST LOGIC
            if _is_hod_query:
                title_lower = title.lower()
                if "hod" in title_lower or "head of" in title_lower:
                    r["score"] = r.get("score", 0.0) + 4.0
                    logger.info(f"HOD document boost applied post-rerank: {title[:60]}")
                
                # Boost if query department matches doc title/metadata
                is_cse = "cse" in effective_query.lower() or "computer science" in effective_query.lower()
                if is_cse and ("computer science" in title_lower or "cse" in title_lower or r.get("department", "").lower() == "computer science and engineering"):
                    r["score"] = r.get("score", 0.0) + 2.0

            # FACULTY LIST PRIORITY BOOST — department-wide queries
            # Ensures consolidated list doc beats individual profiles
            if _is_dept_list_query and doc_type == "faculty_list":
                r["score"] = r.get("score", 0.0) + 5.0
                logger.info(f"Faculty LIST boost applied post-rerank: {title[:60]}")

            # FACULTY PROFILE PRIORITY BOOST
            # When querying about a person, faculty docs always win over
            # random circulars, NIRF PDFs, or seed data mentioning the name
            elif _is_faculty_query and (doc_type.startswith("faculty") or doc_type == "department_info"):
                if person_name:
                    name_parts = [p.lower() for p in person_name.split() if len(p) > 2]
                    title_lower = title.lower()
                    if any(part in title_lower for part in name_parts):
                        r["score"] = r.get("score", 0.0) + 4.0
                    else:
                        r["score"] = r.get("score", 0.0) + 2.0
                else:
                    r["score"] = r.get("score", 0.0) + 2.0
                logger.info(f"Faculty boost applied post-rerank: {title[:60]} (score: {r['score']})")

        # Re-sort results by score descending after custom boosting
        all_results = sorted(all_results, key=lambda x: x.get("score", 0.0), reverse=True)

        # Prepend high-confidence bypassed chunks (Tabular Data & HOD KG Facts) back into the final results
        if kg_hod_chunk:
            all_results.insert(0, kg_hod_chunk)
        if tabular_chunk:
            all_results.insert(0, tabular_chunk)

        # ── Step 7: DuckDuckGo fallback ──────────────────────────
        used_ddg = False

        # Check if the query asks for specific years (e.g. 2025, 2026, 2027)
        query_years = re.findall(r"\b(202\d)\b", query)

        # A query is a freshness query if it contains freshness keywords or asks for current/future years (>= 2025)
        is_fresh_query = needs_fresh_web_search(query) or any(int(yr) >= 2025 for yr in query_years)

        needs_ddg = False
        if not all_results:
            # Condition 1: No local results at all — always use DDG fallback
            needs_ddg = True
            logger.info("No local results found — triggering DDG fallback")
        elif is_fresh_query and self.settings.force_ddg_for_fresh_queries:
            # Condition 2: It is a freshness query and we have local results
            if query_years:
                # If specific years >= 2025 are requested, check if they are missing from top results
                # (to prevent semantic leakage from low-ranked irrelevant matches)
                top_results = all_results[:3]
                local_context_text = " ".join([r.get("text", "") + " " + r.get("title", "") for r in top_results]).lower()
                missing_any_year = any(year not in local_context_text for year in query_years if int(year) >= 2025)
                if missing_any_year:
                    needs_ddg = True
                    logger.info("Freshness query missing requested year(s) in top local results — triggering DDG search")
            else:
                # If no specific year is requested, but freshness keywords are present (e.g., latest, new, tender, circular),
                # it is a real-time query checking for live site updates, which are not in our static DB.
                # So we always trigger DDG search.
                needs_ddg = True
                logger.info("General freshness query for real-time site updates — triggering DDG search")

        if needs_ddg:
            web_query = self.query_processor.build_web_search_query(query_intent) if is_fresh_query else query
            ddg_results, used_ddg = self.web_retriever.retrieve(web_query)
            if ddg_results:
                if is_fresh_query:
                    all_results = ddg_results + all_results
                else:
                    all_results.extend(ddg_results)
                all_results = self._deduplicate(all_results)

            if not all_results:
                no_info = (
                    "I couldn't find specific information about this in my knowledge base "
                    "or the IIT Jammu website. Please visit https://www.iitjammu.ac.in "
                    "for the most up-to-date information, or try rephrasing your question."
                )
                return RAGResult(answer=no_info, sources=[], confidence=0.1, detected_language=target_language)

        # ── Step 8: Context compression ──────────────────────────
        # Dynamically adjust context size: use 12 documents for list-style queries to prevent truncation of scholar/faculty lists
        top_k_context = 12 if (_is_dept_list_query or _is_supervisor_query or _is_hod_query) else 6
        logger.info(f"Using top_k_context = {top_k_context} for context building")
        
        context = self.context_builder.build(all_results[:top_k_context], query=effective_query)
        source_nodes = self._results_to_flat_nodes(all_results)

        # ── Step 9: LLM generation ───────────────────────────────
        try:
            web_note = ""
            if used_ddg:
                web_note = (
                    "\nNote: Some context below comes from a live web search "
                    "of iitjammu.ac.in. Prefer the live web-search context for current, "
                    "latest, deadline, notice, fee, admission, and placement questions."
                )

            # Get conversation history for follow-up context
            conversation_history = ""
            if session_id:
                conversation_history = self.memory.format_history_for_llm(session_id)

            answer_text = await self.llm.formulate_answer(
                query=effective_query,
                context=context,
                target_language=target_language,
                web_search_context=web_note,
                conversation_history=conversation_history,
            )

            # ── Post-LLM DDG retry: if LLM says "no info", try web ──
            if self._is_no_info_answer(answer_text) and not used_ddg:
                logger.info("LLM returned no-info answer — triggering DDG retry")
                web_query = self.query_processor.build_web_search_query(query_intent)
                ddg_results, used_ddg = self.web_retriever.retrieve(web_query)

                if not ddg_results:
                    # Try with raw query as fallback
                    ddg_results, used_ddg = self.web_retriever.retrieve(query)

                if ddg_results:
                    # Rebuild context with DDG results prepended
                    all_results = ddg_results + all_results
                    all_results = self._deduplicate(all_results)
                    context = self.context_builder.build(all_results[:top_k_context], query=effective_query)
                    source_nodes = self._results_to_flat_nodes(all_results)
                    web_note = (
                        "\nNote: Context below includes LIVE web search results "
                        "from iitjammu.ac.in. Prefer this over any cached KB context."
                    )
                    logger.info(f"DDG retry succeeded — {len(ddg_results)} web results added")
                    answer_text = await self.llm.formulate_answer(
                        query=effective_query,
                        context=context,
                        target_language=target_language,
                        web_search_context=web_note,
                        conversation_history=conversation_history,
                    )
                else:
                    logger.info("DDG retry returned no results")

            confidence = min(0.95, 0.5 + 0.05 * len(all_results))

        except Exception as e:
            logger.error(f"LLM generation error: {type(e).__name__}: {e}")
            raise

        # ── Step 10: Store assistant response in memory ──────────
        if session_id:
            self.memory.add_message(session_id, "assistant", answer_text[:500])

        # ── Clean up sources for the frontend ────────────────────
        cleaned_sources = []
        seen_keys = set()
        for node in source_nodes:
            key = node.path if node.path else node.title
            if not key:
                continue
            # Remove "(part X)" from the title
            clean_title = re.sub(r"\s*\(part \d+\)\s*$", "", node.title, flags=re.IGNORECASE)
            if key not in seen_keys:
                seen_keys.add(key)
                cleaned_sources.append(FlatNode(
                    node_id=node.node_id,
                    title=clean_title,
                    path=node.path,
                    summary=node.summary,
                    text=node.text,
                    score=node.score
                ))

        # Clean up common model prefixes like "**Answer:**" or "Answer:" or "**Response:**"
        answer_text = re.sub(r"^\**Answer:\**\s*", "", answer_text, flags=re.IGNORECASE)
        answer_text = re.sub(r"^\**Response:\**\s*", "", answer_text, flags=re.IGNORECASE)
        answer_text = answer_text.strip()

        return RAGResult(
            answer=answer_text,
            sources=cleaned_sources[:3],
            confidence=round(confidence, 2),
            detected_language=target_language,
        )


# ══════════════════════════════════════════════════════════════════
#  Singletons (backward compatible)
# ══════════════════════════════════════════════════════════════════

_tree: Optional[IITJKnowledgeTree] = None
_engine: Optional[HybridRAGEngine] = None


def get_knowledge_tree() -> IITJKnowledgeTree:
    """Get or create the legacy knowledge tree."""
    global _tree
    if _tree is None:
        settings = get_settings()
        index_path = settings.index_file

        def _resolve(path: str) -> str:
            if path.startswith("../"):
                path = path[3:]
            if os.path.exists(path):
                return path
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(backend_dir)
            from_parent = os.path.join(parent_dir, path)
            if os.path.exists(from_parent):
                return from_parent
            project_root = os.path.dirname(parent_dir)
            from_root = os.path.join(project_root, path)
            if os.path.exists(from_root):
                return from_root
            return path

        _tree = IITJKnowledgeTree(_resolve(index_path))
    return _tree


def get_rag_engine() -> HybridRAGEngine:
    """Get or create the singleton HybridRAGEngine."""
    global _engine
    if _engine is None:
        from llm.client import get_llm_client

        chroma_store = None
        knowledge_graph = None

        try:
            from vectorstore.chroma_store import get_chroma_store
            chroma_store = get_chroma_store()
            logger.info(f"✅ ChromaDB: {chroma_store.count()} documents")
        except Exception as e:
            logger.warning(f"⚠️  ChromaDB not available: {e}")

        try:
            from services.knowledge_graph import get_knowledge_graph
            knowledge_graph = get_knowledge_graph()
            logger.info(f"✅ KG: {knowledge_graph.node_count()} nodes")
        except Exception as e:
            logger.warning(f"⚠️  KG not available: {e}")

        _engine = HybridRAGEngine(
            tree=get_knowledge_tree(),
            llm_client=get_llm_client(),
            chroma_store=chroma_store,
            knowledge_graph=knowledge_graph,
        )
    return _engine
