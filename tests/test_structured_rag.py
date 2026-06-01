#!/usr/bin/env python3
"""
tests/test_structured_rag.py — Validate Structured RAG enhancements
======================================================================
Tests typo/compound queries, KG-based HOD extraction, and tabular schema retrieval.
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

# Test queries and verification checks
TEST_CASES = [
    {
        "name": "Fuzzy CSE HOD Query (Compound + Typos)",
        "query": "who is csehod?",
        "checks": ["Yamuna Prasad"],
        "description": "Fuzzy compound query should correctly identify Dr. Yamuna Prasad as CSE HoD using KG boost."
    },
    {
        "name": "Fuzzy CSE HOD Email Query",
        "query": "Who is the Head of Department for CSE and what is their contact email?",
        "checks": ["Yamuna Prasad", "hod.cse@iitjammu.ac.in"],
        "description": "Query for contact details should return CSE HoD's email address and name from KG."
    },
    {
        "name": "Fuzzy Civil HOD Query (Typos)",
        "query": "who is the hied od depaetement od civil engneering?",
        "checks": ["Surendra Beniwal"],
        "description": "Fuzzy query with typos should identify Dr. Surendra Beniwal as Civil Engineering HoD from KG."
    },
    {
        "name": "GATE Cutoff Query",
        "query": "What is the GATE cutoff requirements for MTech?",
        "checks": ["gate", "numerical"],
        "description": "Should route to TabularRetriever and output GATE cutoff guidelines from cutoffs.json."
    },
    {
        "name": "GATE Cutoff Query 2025 (CSE)",
        "query": "What was the GATE cutoff score for CSE in 2025?",
        "checks": ["643", "589", "639", "Computer Science"],
        "description": "Should retrieve the exact 2025 GATE cutoff score table including CSE (GEN: 643)."
    },
    {
        "name": "GATE Cutoff Query 2024 (Structural)",
        "query": "What was the structural engineering GATE cutoff for all categories in 2024?",
        "checks": ["418", "393", "320", "Structural"],
        "description": "Should retrieve the 2024 GATE cutoff scores for Structural Engineering."
    },
    {
        "name": "GATE Cutoff Query All Branches Previous Years",
        "query": "gate cutoff scores of previous yr for all branches",
        "checks": ["2025", "2024", "2023", "VLSI Design", "Tunnel Engineering", "Thermal & Energy"],
        "description": "Should output all tables of GATE cutoffs for 2025, 2024, and 2023."
    },
    {
        "name": "Placements Query",
        "query": "What is the placement statistics at IIT Jammu?",
        "checks": ["Mechanical Engineering", "Electrical Engineering", "LPA"],
        "description": "Should route to TabularRetriever and output placement tables from placements.json."
    },
    {
        "name": "Fee Waiver Query",
        "query": "What are the B.Tech fee waiver rules for general and SC ST students?",
        "checks": ["waiver", "SC/ST", "fee", "1,00,000"],
        "description": "Should retrieve fee structure tables from fees.json."
    },
    {
        "name": "JEE Advanced CSE Cutoff 2025 Query",
        "query": "What was the B.Tech JEE cutoff for Computer Science in 2025?",
        "checks": ["6,651", "Computer Science", "closing rank"],
        "description": "Should retrieve 2025 JEE Advanced B.Tech closing ranks and display CSE as 6,651."
    },
    {
        "name": "JEE Advanced B.Tech Cutoffs Previous Years",
        "query": "What are the JEE Advanced cutoff ranks for previous years?",
        "checks": ["2025", "2024", "2023", "2022", "Engineering Physics", "Mathematics and Computing"],
        "description": "Should display all tables of B.Tech JEE Advanced closing ranks for 2025, 2024, 2023, and 2022."
    }
]

async def run_tests():
    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}         IIT JAMMU STRUCTURED RAG VALIDATION RUN{RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

    # Initialize engine
    print("Initializing Hybrid RAG Engine...")
    try:
        engine = get_rag_engine()
        print(f"Hybrid RAG Engine initialized. Legacy nodes: {engine.tree.count_nodes()}")
    except Exception as e:
        print(f"{RED}Failed to initialize Hybrid RAG Engine: {e}{RESET}")
        sys.exit(1)

    passed_count = 0
    failed_count = 0

    for idx, tc in enumerate(TEST_CASES, 1):
        print(f"\n{BOLD}[Test {idx}/{len(TEST_CASES)}] {tc['name']}{RESET}")
        print(f"Query: \"{tc['query']}\"")
        print(f"Description: {tc['description']}")
        
        # Process and output internal query processor details
        qp_res = engine.query_processor.process(tc['query'])
        print(f"  - Cleaned Query: \"{qp_res.processed_query}\"")
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
    print(f"{BOLD}  VERIFICATION COMPLETE{RESET}")
    print(f"  Passed: {passed_count}/{len(TEST_CASES)}")
    print(f"  Failed: {failed_count}/{len(TEST_CASES)}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
