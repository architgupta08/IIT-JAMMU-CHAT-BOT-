"""
utils/text.py — Text Processing Utilities
==========================================
Chunking, cleaning, and metadata extraction helpers used across
retrievers, scrapers, and the RAG engine.
"""

import re
import hashlib
from typing import List, Dict, Optional
from datetime import datetime


def clean_text(text: str) -> str:
    """Remove excess whitespace, control chars, and normalize Unicode."""
    if not text:
        return ""
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def chunk_text(
    text: str,
    title: str = "",
    source_url: str = "",
    topic: str = "General",
    chunk_size: int = 500,
    overlap: int = 100,
    metadata: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks for vector storage.

    Uses sentence-boundary-aware splitting when possible.
    Each chunk carries forward the title, source, and topic metadata.
    """
    text = clean_text(text)
    if not text or len(text) < 50:
        return []

    base_meta = {
        "title": title,
        "source_url": source_url,
        "topic": topic,
    }
    if metadata:
        base_meta.update(metadata)

    # Short text → single chunk
    if len(text) <= chunk_size:
        return [{**base_meta, "text": text}]

    chunks = []
    # Try sentence-boundary splitting first
    sentences = re.split(r"(?<=[.!?])\s+", text)

    current_chunk = ""
    chunk_idx = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk and len(current_chunk) >= 50:
                chunk_idx += 1
                chunk_meta = {**base_meta}
                if chunk_idx > 1 or len(text) > chunk_size:
                    chunk_meta["title"] = f"{title} (part {chunk_idx})"
                chunks.append({**chunk_meta, "text": current_chunk})

            # Start new chunk with overlap from previous
            if current_chunk and overlap > 0:
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = f"{overlap_text} {sentence}".strip()
            else:
                current_chunk = sentence

    # Final chunk
    if current_chunk and len(current_chunk) >= 50:
        chunk_idx += 1
        chunk_meta = {**base_meta}
        if chunk_idx > 1:
            chunk_meta["title"] = f"{title} (part {chunk_idx})"
        chunks.append({**chunk_meta, "text": current_chunk})

    # Fallback: if sentence splitting produced nothing, use character-based
    if not chunks:
        start = 0
        idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= 50:
                idx += 1
                chunk_meta = {**base_meta}
                if idx > 1:
                    chunk_meta["title"] = f"{title} (part {idx})"
                chunks.append({**chunk_meta, "text": chunk})
            start = end - overlap

    return chunks


def content_hash(text: str) -> str:
    """SHA-256 hash for content deduplication."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def extract_year_from_text(text: str) -> Optional[int]:
    """
    Extract the most likely year from text content.
    Prefers years in the 2020-2030 range for recency.
    """
    years = re.findall(r"\b(20[1-3]\d)\b", text)
    if not years:
        return None
    # Return the most recent year found
    return max(int(y) for y in years)


def extract_year_from_url(url: str) -> Optional[int]:
    """Extract year from URL patterns like /2025/ or ?year=2025."""
    match = re.search(r"[/=_-](20[1-3]\d)(?:[/&?_.-]|$)", url)
    return int(match.group(1)) if match else None


def extract_department(text: str, title: str = "") -> str:
    """Infer department from text content."""
    combined = f"{title} {text}".lower()
    dept_map = {
        "Computer Science": ["computer science", "cse", "cs department"],
        "Electrical Engineering": ["electrical engineering", "ee department", "electrical dept"],
        "Mechanical Engineering": ["mechanical engineering", "me department"],
        "Civil Engineering": ["civil engineering", "ce department"],
        "Chemical Engineering": ["chemical engineering", "che department"],
        "Mathematics": ["mathematics", "math department", "mathematics and computing"],
        "Physics": ["physics", "engineering physics"],
        "Chemistry": ["chemistry department", "chemistry"],
        "HSS": ["humanities", "social sciences", "hss"],
        "Materials Engineering": ["materials engineering"],
        "Biosciences": ["biosciences", "bioengineering"],
    }
    for dept, signals in dept_map.items():
        if any(s in combined for s in signals):
            return dept
    return "General"


def infer_doc_type(title: str, text: str, url: str = "") -> str:
    """Classify document type from content."""
    combined = f"{title} {text} {url}".lower()
    type_map = {
        "Notice": ["notice", "circular", "office order", "announcement"],
        "Admission": ["admission", "jee", "gate", "josaa", "eligibility", "apply"],
        "Faculty": ["professor", "faculty", "assistant professor", "hod", "dean"],
        "Fee": ["fee structure", "tuition", "hostel charge", "mess charge"],
        "Placement": ["placement", "ctc", "lpa", "package", "recruitment"],
        "Research": ["research", "project", "publication", "lab", "jrf", "srf"],
        "Scholarship": ["scholarship", "mcm", "pmrf", "fellowship", "freeship"],
        "Academic": ["curriculum", "syllabus", "course", "program", "semester"],
        "Campus": ["hostel", "library", "sports", "medical", "campus", "facility"],
        "Event": ["event", "workshop", "seminar", "conference", "hackathon"],
    }
    for doc_type, signals in type_map.items():
        if any(s in combined for s in signals):
            return doc_type
    return "General"


def build_document_metadata(
    title: str,
    text: str,
    source_url: str = "",
    crawl_date: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build rich metadata dict for a document chunk.
    Extracts year, department, doc type, and crawl date.
    """
    year = extract_year_from_text(text) or extract_year_from_url(source_url) or datetime.now().year
    return {
        "title": title[:200],
        "source_url": source_url[:500],
        "page_title": title[:200],
        "year": str(year),
        "department": extract_department(text, title),
        "doc_type": infer_doc_type(title, text, source_url),
        "crawl_date": crawl_date or datetime.utcnow().isoformat(),
        "content_hash": content_hash(text),
        "char_count": str(len(text)),
    }
