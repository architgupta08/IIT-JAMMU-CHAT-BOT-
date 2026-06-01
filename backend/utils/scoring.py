"""
utils/scoring.py — Scoring Utilities for RAG Retrieval
======================================================
Implements the recency-weighted scoring formula:
  final_score = (semantic_similarity * 0.7) + (recency_score * 0.3)
"""

from datetime import datetime
from typing import Optional

from config import get_settings


def compute_recency_score(
    doc_year: Optional[int],
    current_year: Optional[int] = None,
    decay_per_year: Optional[float] = None,
) -> float:
    """
    Compute recency score for a document.

    Returns 1.0 for current year, decays by `decay_per_year` per year.
    Minimum score is 0.0.

    Examples:
        current year doc → 1.0
        1 year old doc   → 0.85
        2 year old doc   → 0.70
        5+ year old doc  → 0.25
    """
    if doc_year is None:
        return 0.5  # Unknown age → neutral score

    if current_year is None:
        current_year = datetime.now().year

    if decay_per_year is None:
        settings = get_settings()
        decay_per_year = settings.recency_decay_per_year

    age = max(0, current_year - doc_year)
    score = max(0.0, 1.0 - (age * decay_per_year))
    return round(score, 3)


def compute_final_score(
    semantic_score: float,
    recency_score: float,
    semantic_weight: Optional[float] = None,
    recency_weight: Optional[float] = None,
    boost: float = 0.0,
) -> float:
    """
    Compute the final weighted score for a retrieved document.

    Formula: (semantic * weight) + (recency * weight) + boost

    Args:
        semantic_score: Normalized similarity score (0-1, higher = better)
        recency_score: Recency score (0-1, higher = newer)
        semantic_weight: Weight for semantic component (default 0.7)
        recency_weight: Weight for recency component (default 0.3)
        boost: Optional flat boost for priority documents
    """
    if semantic_weight is None or recency_weight is None:
        settings = get_settings()
        semantic_weight = semantic_weight or settings.semantic_weight
        recency_weight = recency_weight or settings.recency_weight

    score = (semantic_score * semantic_weight) + (recency_score * recency_weight) + boost
    return round(min(1.0, max(0.0, score)), 4)


def normalize_chroma_distance(distance: float) -> float:
    """
    Convert ChromaDB distance (lower = better) to similarity score (higher = better).

    ChromaDB uses L2 distance by default. We convert to a 0-1 similarity score.
    Typical L2 distances range from 0 (identical) to ~2.0 (very different).
    """
    # Clamp to reasonable range
    distance = max(0.0, min(distance, 2.0))
    # Convert: similarity = 1 - (distance / 2)
    return round(1.0 - (distance / 2.0), 4)
