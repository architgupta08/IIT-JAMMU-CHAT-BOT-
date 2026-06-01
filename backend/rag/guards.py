"""
rag/guards.py — Off-Topic & Freshness Guards
==============================================
Determines whether a query is relevant to IIT Jammu and whether
it requires fresh web search data.
"""

import re
from typing import Set

# ── IIT Jammu signal words ─────────────────────────────────────────
IITJ_SIGNALS: Set[str] = {
    "iit", "jammu", "iitj", "admission", "btech", "b.tech", "mtech", "m.tech",
    "phd", "ph.d", "msc", "fee", "fees", "hostel", "mess", "placement", "scholarship",
    "gate", "jee", "josaa", "faculty", "professor", "director", "campus", "research",
    "department", "programme", "program", "course", "syllabus", "cutoff", "rank",
    "cse", "ee", "me", "ce", "che", "hss", "library", "medical", "sports", "jagti",
    "stipend", "fellowship", "pmrf", "mcm", "nirf", "tnp", "internship", "lpa", "ctc",
    "nagrota", "paloura", "academic", "semester", "elective", "convocation", "alumni",
    "curriculum", "grading", "cpi", "sgpa", "registration", "exam", "result",
    "notice", "circular", "announcement", "event", "workshop", "seminar", "lab",
    "hackathon", "robotics", "drone", "cybersecurity", "data science", "ai",
}

# ── Off-topic patterns ─────────────────────────────────────────────
OFF_TOPIC_SIGNALS = [
    # Coding / programming tasks
    "write a python", "write a java", "write a c++", "write a code",
    "write code for", "write a program", "write a function", "write a script",
    "code for", "program for", "implement a", "implement the",
    "debug this", "fix this code", "fix my code", "explain this code",
    "python code", "java code", "c++ code", "javascript code", "html code",
    "sql query", "binary search", "linear search", "bubble sort", "merge sort",
    "quick sort", "linked list", "stack implementation", "queue implementation",
    "tree traversal", "tic tac", "snake game", "chess game", "sudoku",
    "calculator app", "fibonacci", "sorting algorithm", "searching algorithm",
    "data structure", "recursion example", "machine learning code",
    "neural network", "deep learning code",
    # General knowledge outside IITJ
    "recipe for", "how to cook", "best movie", "song lyrics",
    "cricket score", "ipl score", "football score", "match score",
    "stock price", "bitcoin", "cryptocurrency", "share price",
    "weather today", "weather in", "news today", "latest news",
    "capital of", "president of", "prime minister of",
    "translate to", "meaning of word", "synonym of", "antonym of",
    # Personal / entertainment
    "love poem", "write a poem", "write an essay", "write a story",
    "tell me a joke", "tell a joke", "funny joke", "make me laugh",
    "horoscope", "astrology", "plan my trip", "hotel in", "restaurant near",
]

# ── Freshness signals (queries needing latest data) ────────────────
FRESHNESS_SIGNALS: Set[str] = {
    "latest", "atest", "current", "recent", "new", "update", "today", "deadline", 
    "schedule", "2025", "2026", "notice", "circular", "tender",
    "job", "jobs", "vacancy", "vacancies", "career", "careers", "recruitment", "recruit",
    "opportunity", "opportunities", "opening", "openings"
}

FRESHNESS_PHRASES = []



def is_off_topic(query: str) -> bool:
    """
    Returns True if query is clearly unrelated to IIT Jammu.

    Step 1: If any IIT Jammu signal word present → always on-topic.
    Step 2: If matches off-topic pattern AND no IITJ signal → off-topic.
    """
    q = query.lower().strip()
    words = set(re.findall(r"\b\w+\b", q))

    # Step 1: Strong IITJ signals → always answer
    if words & IITJ_SIGNALS:
        return False
    if any(sig in q for sig in ["iit jammu", "iit j", "iitjammu", "jagti", "nagrota"]):
        return False

    # Step 2: Check off-topic patterns
    for pattern in OFF_TOPIC_SIGNALS:
        if pattern in q:
            return True

    # Short queries without IITJ context → be generous
    if len(q.split()) <= 4:
        return False

    return False


def needs_fresh_web_search(query: str) -> bool:
    """Return True when stale local data is risky and live search should be used."""
    q = query.lower().strip()
    words = set(re.findall(r"\b[\w.]+\b", q))

    if words & FRESHNESS_SIGNALS:
        return True
    if any(signal in q for signal in FRESHNESS_SIGNALS):
        return True
        
    # Check for multi-word job role freshness signals
    job_roles = ["project associate", "project staff", "research associate", "postdoc", "jrf"]
    if any(role in q for role in job_roles):
        return True
        
    return any(phrase in q for phrase in FRESHNESS_PHRASES)
