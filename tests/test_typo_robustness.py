#!/usr/bin/env python3
"""
tests/test_typo_robustness.py — Validate typo tolerance and unstructured queries
==================================================================================
Tests spelling correction, fuzzy matching, and end-to-end RAG retrieval under typos.
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

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

TEST_CASES = [
    {
        "name": "CSE HOD Query with Typos",
        "query": "who is dr ankuit dubey?",
        "checks": ["Ankit Dubey", "Associate Professor", "Electrical Engineering"],
        "description": "Fuzzy name matching for Dr. Ankit Dubey."
    },
    {
        "name": "CSE HOD Query with Last Name / Typos",
        "query": "who is ankit duebvy?",
        "checks": ["Ankit Dubey"],
        "description": "Fuzzy name matching with typos in last name."
    },
    {
        "name": "CSE HOD Query with First Name Only",
        "query": "who is badri?",
        "checks": ["Badri Narayan Subudhi"],
        "description": "Fuzzy matching for first name 'badri'."
    },
    {
        "name": "CSE HOD Query with Last Name Only",
        "query": "who is dubey?",
        "checks": ["Ankit Dubey"],
        "description": "Fuzzy matching for last name 'dubey'."
    },
    {
        "name": "Placement Statistics Query with Typos",
        "query": "what are placement stastucs?",
        "checks": ["placements", "salary", "LPA"],
        "description": "General typo matching 'stastucs' to 'statistics' for placements."
    },
    {
        "name": "Cutoff Query with Typos",
        "query": "what is cse cutodd?",
        "checks": ["643", "589", "639", "cutoff"],
        "description": "General typo matching 'cutodd' to 'cutoff' for CSE GATE cutoffs."
    },
    {
        "name": "Placements Records Query with Multi-typos",
        "query": "did u insert cse placemenr recoreds?",
        "checks": ["placement", "placements", "Computer Science"],
        "description": "Verify multi-typo correction for 'placemenr' and 'recoreds'."
    }
]

async def run_tests():
    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}      IIT JAMMU CHATBOT TYPO ROBUSTNESS VALIDATION RUN{RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

    print("Initializing Hybrid RAG Engine...")
    try:
        engine = get_rag_engine()
        print("Hybrid RAG Engine initialized successfully.")
    except Exception as e:
        print(f"{RED}Failed to initialize Hybrid RAG Engine: {e}{RESET}")
        sys.exit(1)

    passed_count = 0
    failed_count = 0

    for idx, tc in enumerate(TEST_CASES, 1):
        print(f"\n{BOLD}[Test {idx}/{len(TEST_CASES)}] {tc['name']}{RESET}")
        print(f"Query: \"{tc['query']}\"")
        
        qp_res = engine.query_processor.process(tc['query'])
        print(f"  - Corrected Query: \"{qp_res.corrected_query}\"")
        print(f"  - Expanded Query:  \"{qp_res.processed_query}\"")
        print(f"  - Extracted Intent: {qp_res.intent}")
        print(f"  - Extracted Entities: {qp_res.entities}")
        
        t0 = asyncio.get_event_loop().time()
        res = await engine.answer(tc['query'])
        latency = asyncio.get_event_loop().time() - t0
        
        print(f"  - Time taken: {latency:.2f}s")
        print(f"  - Confidence: {res.confidence}")
        print(f"  - Sources: {[s.title for s in res.sources]}")
        print(f"\n--- Answer ---")
        print(res.answer)
        print(f"--------------")
        
        # Run verification checks
        missing = []
        for check in tc['checks']:
            if check.lower() not in res.answer.lower():
                missing.append(check)
                
        if not missing:
            print(f"{GREEN}✓ PASS{RESET}")
            passed_count += 1
        else:
            print(f"{RED}✗ FAIL — Missing keywords: {missing}{RESET}")
            failed_count += 1

    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}  TYPO TOLERANCE VERIFICATION COMPLETE{RESET}")
    print(f"  Passed: {passed_count}/{len(TEST_CASES)}")
    print(f"  Failed: {failed_count}/{len(TEST_CASES)}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
