"""
rag/context_builder.py — Context Compression & Formatting
==========================================================
Builds the final context string sent to the LLM from retrieved
documents. Applies compression to only include relevant paragraphs
and caps total context size.
"""

import re
import logging
from typing import List, Dict, Any

from config import get_settings

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds compressed, relevant context for LLM generation."""

    def __init__(self):
        self.settings = get_settings()

    def build(
        self,
        results: List[Dict[str, Any]],
        query: str = "",
        max_chars: int = None,
    ) -> str:
        """
        Build a context string from retrieved documents.

        Applies:
          1. Relevance filtering — skip very low-score results
          2. Truncation — cap individual documents
          3. Total size limit — prevent context overflow
          4. Source attribution — label each section
        """
        max_chars = max_chars or self.settings.max_context_chars
        max_per_node = self.settings.max_text_per_node
        parts = []
        total_chars = 0

        for src in results:
            title = src.get("title", "")
            text = src.get("text", "").strip()
            source_type = src.get("source_type", "")
            source_url = src.get("source_url", "")

            if not text:
                continue

            # For list-type documents (faculty lists, program lists), NEVER extract paragraphs
            # — they must be sent to the LLM in full so no names are dropped.
            is_list_doc = (
                src.get("doc_type") in ("faculty_list", "list", "faq_factsheet", "kg_fact")
                or "faculty list" in title.lower()
                or "Faculty List" in title
                or "Structured Data" in title
            )

            # Compress: extract only relevant paragraphs if text is long (skip for list docs)
            if query and len(text) > max_per_node and not is_list_doc:
                text = self._extract_relevant_paragraphs(text, query, max_per_node)

            if len(text) > max_per_node and not is_list_doc:
                text = text[:max_per_node] + "…"

            section = f"### {title}\n"
            if source_type:
                section += f"*[{source_type}]*\n"
            if source_url:
                section += f"Source: {source_url}\n"
            section += f"{text}\n"

            if total_chars + len(section) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    section = section[:remaining] + "…"
                else:
                    break

            parts.append(section)
            total_chars += len(section)

        return "\n---\n".join(parts)

    def _extract_relevant_paragraphs(
        self, text: str, query: str, max_chars: int
    ) -> str:
        """
        Extract the most relevant paragraphs from a long text.

        Uses simple query-word overlap scoring to rank paragraphs.
        """
        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n|\n(?=#{1,4}\s)", text)
        if not paragraphs:
            return text[:max_chars]

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not query_words:
            return text[:max_chars]

        # Score paragraphs by query-word overlap
        scored = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 30:
                continue
            para_words = set(re.findall(r"\b\w{3,}\b", para.lower()))
            overlap = len(query_words & para_words)
            scored.append((overlap, para))

        # Sort by relevance, keep top paragraphs within budget
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        total = 0
        for score, para in scored:
            if total + len(para) > max_chars:
                break
            selected.append(para)
            total += len(para)

        return "\n\n".join(selected) if selected else text[:max_chars]


# ── Singleton ─────────────────────────────────────────────────────
_builder = None


def get_context_builder() -> ContextBuilder:
    global _builder
    if _builder is None:
        _builder = ContextBuilder()
    return _builder
