#!/usr/bin/env python3
"""
tests/test_faculty_rag.py — Validate Faculty Profile Retrieval and Accuracy
=============================================================================
Tests the RAG engine on 4 questions each for 5 faculty members with typos:
1. Who is the faculty?
2. Their research area / interests?
3. Their publications?
4. Their contact email / designation?
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve paths
project_root = Path(__file__).parent.parent.resolve()
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

# Set environment
os.environ["CHROMA_DB_PATH"] = "data/processed/chroma_db"

from rag.engine import get_rag_engine

# Setup minimal logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Test faculty profiles and their 4 questions
FACULTY_TESTS = [
    {
        "name": "Dr. Badri Narayan Subudhi",
        "questions": [
            "who is badri nayan subudhu?",
            "what is the research interest of badri nayan subudhu?",
            "what are the publications of badri nayan subudhu?",
            "what is the contact email address of badri nayan subudhu?"
        ]
    },
    {
        "name": "Dr. Ankit Dubey",
        "questions": [
            "who is ankiyt dubey?",
            "what is the research area of ankiyt dubey?",
            "what are the publications of ankiyt dubey?",
            "what is the designation of ankiyt dubey?"
        ]
    },
    {
        "name": "Dr. Shaifu Gupta",
        "questions": [
            "who is shaifu gupta?",
            "what is the reseach interest of shaifu gupta?",
            "what are the publications of shaifu gupta?",
            "what is the email address of shaifu gupta?"
        ]
    },
    {
        "name": "Dr. Ajay Singh",
        "questions": [
            "who is ajay singh?",
            "what is the reseach ijtrsdt of ajay singh?",
            "what are the publications of ajay singh?",
            "what is the designation of ajay singh?"
        ]
    },
    {
        "name": "Dr. Karan Nathwani",
        "questions": [
            "who is proff karsan nagthwani?",
            "what is the resdeasrch ares of proff karsan nagthwani?",
            "what are the publications of proff karsan nagthwani?",
            "what is the email of proff karsan nagthwani?"
        ]
    }
]

async def run_tests():
    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}        IIT JAMMU FACULTY PROFILE RAG VALIDATION RUN{RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

    # Initialize engine
    print("Initializing Hybrid RAG Engine...")
    try:
        engine = get_rag_engine()
        print(f"Hybrid RAG Engine initialized. Chroma Documents: {engine.chroma_retriever.store.count()}")
    except Exception as e:
        print(f"{RED}Failed to initialize Hybrid RAG Engine: {e}{RESET}")
        sys.exit(1)

    for faculty in FACULTY_TESTS:
        print(f"\n{BOLD}{YELLOW}============================================================{RESET}")
        print(f"{BOLD}{YELLOW}  FACULTY MEMBER: {faculty['name']}{RESET}")
        print(f"{BOLD}{YELLOW}============================================================{RESET}")
        
        for q_idx, q in enumerate(faculty["questions"], 1):
            print(f"\n{BOLD}[Q{q_idx}] Query: \"{q}\"{RESET}")
            
            # Process query
            qp_res = engine.query_processor.process(q)
            print(f"  - Cleaned Query: \"{qp_res.processed_query}\"")
            print(f"  - Extracted Entities: {qp_res.entities}")
            
            t0 = asyncio.get_event_loop().time()
            res = await engine.answer(q)
            latency = asyncio.get_event_loop().time() - t0
            
            print(f"  - Time taken: {latency:.2f}s | Confidence: {res.confidence}")
            print(f"  - Sources retrieved: {[s.title for s in res.sources]}")
            print(f"\n--- Chatbot Answer ---")
            print(res.answer)
            print(f"----------------------")

    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}  FACULTY PROFILE RAG RUN COMPLETE{RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
