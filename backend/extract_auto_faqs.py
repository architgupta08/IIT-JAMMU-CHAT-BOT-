"""
backend/extract_auto_faqs.py — Extract FAQs from raw markdown documents using LLM.
================================================================================
Reads all files from data/raw/, generates structured FAQ Q&As using the LLM client,
and saves them to data/processed/auto_generated_faqs.json with progressive state saving
to allow resuming if interrupted.
"""

import os
import sys
import json
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from llm.client import get_llm_client
from ingest_raw_md import filename_to_url, filename_to_title, infer_department

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_auto_faqs")

# Silence noisy libraries
for noisy in ['httpx', 'httpcore', 'huggingface_hub', 'sentence_transformers', 'urllib3']:
    logging.getLogger(noisy).setLevel(logging.ERROR)


def clean_json_text(text: str) -> str:
    """Extract and clean the JSON array string from LLM output."""
    text = text.strip()
    # Remove markdown code blocks if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    
    # Try to find array brackets [ ... ]
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


async def extract_faqs_for_file(client, text: str, title: str, source_url: str, department: str) -> List[Dict[str, str]]:
    """Send document content to LLM to extract up to 10 Q&A pairs."""
    system_instruction = (
        "You are an expert factual FAQ extraction system. Your task is to extract up to 10 highly accurate "
        "and clean Q&A pairs from the provided text about IIT Jammu.\n\n"
        "STRICT RULES:\n"
        "1. Questions must be natural, clear, and represent what a real student, parent, or researcher would ask.\n"
        "2. Answers must be direct, completely factual, and derived ONLY from the provided text. Never hallucinate details.\n"
        "3. Preserve all contact details, names, emails, and exact numbers.\n"
        "4. Format the output strictly as a JSON array of objects with keys 'q' and 'a'. E.g.:\n"
        '[\n  {"q": "What is the contact email?", "a": "You can email at..."}\n]\n'
        "Do not include any explanation or markdown formatting outside of the JSON array."
    )

    prompt = (
        f"DOCUMENT TITLE: {title}\n"
        f"DEPARTMENT: {department}\n"
        f"SOURCE URL: {source_url}\n"
        f"DOCUMENT CONTENT:\n"
        f"----------------------------------------\n"
        f"{text[:6000]}\n"
        f"----------------------------------------\n\n"
        f"Extract Q&A pairs now. Return ONLY the JSON array."
    )

    try:
        response = await client.generate(prompt, system_instruction=system_instruction)
        cleaned = clean_json_text(response)
        qa_list = json.loads(cleaned)
        if isinstance(qa_list, list):
            # Validate structure
            validated = []
            for item in qa_list:
                if isinstance(item, dict) and "q" in item and "a" in item:
                    q = str(item["q"]).strip()
                    a = str(item["a"]).strip()
                    if q and a:
                        validated.append({"q": q, "a": a})
            return validated
    except Exception as e:
        logger.warning(f"Error extracting FAQs: {e}. Raw response: {response[:200] if 'response' in locals() else 'None'}")
    return []


async def main():
    project_root = Path(backend_dir).parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "auto_generated_faqs.json"
    
    # Load existing state if it exists
    state: Dict[str, List[Dict[str, str]]] = {}
    if output_file.exists():
        try:
            state = json.loads(output_file.read_text(encoding="utf-8"))
            logger.info(f"Loaded existing FAQs for {len(state)} files from {output_file}")
        except Exception as e:
            logger.warning(f"Failed to load existing FAQs: {e}. Starting fresh.")

    # Find all raw md files
    md_files = sorted(raw_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in {raw_dir}")
    
    client = get_llm_client()
    
    # Filter files that need to be processed
    to_process = []
    for idx, md_file in enumerate(md_files, 1):
        filename = md_file.name
        if filename not in state:
            to_process.append((idx, md_file))
        
    logger.info(f"Files already processed: {len(state)}. Files remaining to process: {len(to_process)}")
    
    if not to_process:
        logger.info("All files are already processed!")
        return

    # Semaphore to limit concurrency
    sem = asyncio.Semaphore(8)
    # Lock to safely write to output_file and update state
    write_lock = asyncio.Lock()
    
    async def process_file(idx, md_file):
        filename = md_file.name
        title = filename_to_title(filename)
        source_url = filename_to_url(filename)
        
        try:
            text = md_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.error(f"[{idx}/{len(md_files)}] Failed to read {filename}: {e}")
            return
            
        if len(text) < 100:
            logger.info(f"[{idx}/{len(md_files)}] Skipping {filename} (too short: {len(text)} chars)")
            async with write_lock:
                state[filename] = []
                try:
                    output_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception as e:
                    logger.error(f"Failed to save state to {output_file}: {e}")
            return
            
        department = infer_department(text, title, filename)
        
        async with sem:
            logger.info(f"[{idx}/{len(md_files)}] Extracting FAQs from {filename} ({title})")
            faqs = await extract_faqs_for_file(client, text, title, source_url, department)
            
        # Add source metadata to each FAQ item
        for faq in faqs:
            faq["source_url"] = source_url
            faq["department"] = department
            faq["filename"] = filename
            
        async with write_lock:
            state[filename] = faqs
            # Write state progress after every file
            try:
                output_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to save state to {output_file}: {e}")
                
        logger.info(f"[{idx}/{len(md_files)}] Extracted {len(faqs)} Q&A pairs for {filename}")

    # Run remaining tasks concurrently
    tasks = [process_file(idx, md_file) for idx, md_file in to_process]
    await asyncio.gather(*tasks)

    # Flatten and compile final statistics
    all_faqs = []
    for filename, faqs in state.items():
        all_faqs.extend(faqs)
        
    logger.info("=" * 60)
    logger.info("  FAQ EXTRACTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Processed files: {len(state)}")
    logger.info(f"  Total Q&A pairs generated: {len(all_faqs)}")
    logger.info(f"  Output saved to: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
