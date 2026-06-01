"""
llm/client.py — LLM Client (Ollama / Local Models)
====================================================
Handles all LLM interactions: answer generation, tree navigation,
and coding request detection. Communicates with Ollama API.

Keeps the GeminiClient name for backward compatibility but
uses Ollama under the hood.
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict

import httpx

from config import get_settings

logger = logging.getLogger(__name__)

# ── Coding detection signals ────────────────────────────────────
CODE_SIGNALS = [
    "write a program", "write a code", "write code", "write a function",
    "write a script", "implement", "bubble sort", "binary search", "merge sort",
    "fibonacci", "linked list", "stack", "queue", "algorithm", "data structure",
]

# ── Language map ────────────────────────────────────────────────
LANG_MAP = {
    "hi": "Hindi", "de": "German", "fr": "French", "it": "Italian",
    "pt": "Portuguese", "es": "Spanish", "th": "Thai",
    "ur": "Urdu", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
}


class LLMClient:
    """
    LLM client for the IIT Jammu chatbot.
    Supports 'ollama' and 'bedrock' (Amazon Benchmark API) providers.
    Named to maintain backward compatibility with existing imports.
    """

    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.llm_provider.lower()
        
        if self.provider == "bedrock":
            self.model = self.settings.bedrock_model_id
            self.region = self.settings.aws_region
            self.endpoint = self.settings.bedrock_endpoint_url or f"https://bedrock-runtime.{self.region}.amazonaws.com"
            self.token = self.settings.aws_bearer_token_bedrock
            logger.info(f"LLMClient ready — provider: Bedrock | model: {self.model} @ {self.endpoint}")
        else:
            self.base_url = self.settings.ollama_base_url.rstrip("/")
            self.model = self.settings.llm_model
            self._verify_connection_ollama()
            logger.info(f"LLMClient ready — provider: Ollama | model: {self.model} @ {self.base_url}")

    def _verify_connection_ollama(self):
        """Check if Ollama is running and model is available."""
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as r:
                data = json.loads(r.read())
                available = [m["name"] for m in data.get("models", [])]
                if not any(self.model in m for m in available):
                    logger.warning(
                        f"Model '{self.model}' not found. Available: {available}. "
                        f"Run: ollama pull {self.model}"
                    )
                else:
                    logger.info(f"Model '{self.model}' confirmed available in Ollama")
        except Exception as e:
            logger.warning(f"Cannot reach Ollama at {self.base_url}: {e}")

    async def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate a response from the configured LLM provider."""
        if self.provider == "bedrock":
            return await self._generate_bedrock(prompt, system_instruction)
        else:
            return await self._generate_ollama(prompt, system_instruction)

    async def _generate_ollama(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_max_tokens,
                "num_ctx": self.settings.llm_context_window,
            },
        }

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                text = data.get("message", {}).get("content", "").strip()
                if not text:
                    raise RuntimeError(f"Ollama returned empty response: {data}")
                return text
            except httpx.ConnectError:
                raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}.")
            except Exception as e:
                logger.error(f"Ollama error: {type(e).__name__}: {e}")
                raise RuntimeError(f"Ollama call failed: {type(e).__name__}: {e}") from e

    async def _generate_bedrock(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        url = f"{self.endpoint}/model/{self.model}/converse"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": self.settings.bedrock_max_tokens,
                "temperature": self.settings.bedrock_temperature,
                "topP": self.settings.bedrock_top_p
            }
        }
        if system_instruction:
            payload["system"] = [{"text": system_instruction}]

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"Bedrock API Error ({response.status_code}): {response.text}")
                    response.raise_for_status()
                
                data = response.json()
                output_message = data.get("output", {}).get("message", {})
                content_blocks = output_message.get("content", [])
                
                text_response = ""
                for block in content_blocks:
                    if "text" in block:
                        text_response += block["text"]
                
                if not text_response:
                    raise RuntimeError(f"Bedrock returned empty response: {data}")
                
                return text_response.strip()
            except Exception as e:
                logger.error(f"Bedrock error: {type(e).__name__}: {e}")
                raise RuntimeError(f"Bedrock call failed: {type(e).__name__}: {e}") from e

    async def formulate_answer(
        self,
        query: str,
        context: str,
        target_language: str = "en",
        web_search_context: str = "",
        conversation_history: str = "",
    ) -> str:
        """
        Generate a contextual answer using the LLM.

        Args:
            query: User's question
            context: Retrieved context from RAG pipeline
            target_language: Target response language
            web_search_context: Note about web search sources
            conversation_history: Previous conversation for follow-ups
        """
        # ── GUARD: Detect coding requests ────────────────────────
        is_coding_request = any(sig in query.lower() for sig in CODE_SIGNALS)
        if is_coding_request:
            logger.warning(f"Coding request detected: {query[:100]}")
            return (
                "I'm the IIT Jammu Assistant and can only answer questions about "
                "IIT Jammu. I cannot help with coding or programming tasks. "
                "Please ask me about admissions, fees, programs, placements, "
                "or campus life!"
            )

        # ── Language instruction ─────────────────────────────────
        lang_name = LANG_MAP.get(target_language, "")
        lang_instr = (
            f"IMPORTANT: Your entire response MUST be written in {lang_name}. "
            f"Do not use English except for proper nouns like 'IIT Jammu', 'B.Tech', 'GATE'.\n"
            if lang_name else ""
        )

        # ── Topic hint for non-English ───────────────────────────
        topic_hint = ""
        if lang_name:
            q_lower = query.lower()
            if any(w in q_lower for w in ["fee", "fees", "kitni", "charge", "tuition", "shulk"]):
                topic_hint = "FOCUS ONLY on fee and charges information. Do NOT discuss admissions. "
            elif any(w in q_lower for w in ["admission", "apply", "jee", "gate", "josaa", "kaise"]):
                topic_hint = "FOCUS ONLY on admission process information. "

        # ── System prompt ────────────────────────────────────────
        system = (
            "You are the official AI Assistant for IIT Jammu "
            "(Indian Institute of Technology Jammu, India). "
            "You are helpful, accurate, and friendly.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY from the provided context — never use outside knowledge.\n"
            "2. If the answer is not in the context, say: 'I could not find relevant "
            "information from IIT Jammu data. Please visit https://www.iitjammu.ac.in'\n"
            "3. Never invent facts, numbers, names, or URLs. However, you MUST provide and output the EXACT URLs if they are present in the context.\n"
            "4. Include specific numbers and contacts from the context. For DATES: only mention"
            " them if they are in the future or are part of a general process description."
            " NEVER show past application deadlines or past start dates as if they are current.\n"
            "5. Use bullet points for lists. For tabular/numerical data (like cutoffs, fees, placement percentages, packages), ALWAYS use Markdown tables to present the data clearly.\n"
            "6. RESPONSE DEPTH RULE — Match your detail level to the question:\n"
            "   - If the user asks WHO (e.g., 'who are the faculty', 'list professors'): output ONLY names and designation. Nothing else.\n"
            "   - If the user asks ABOUT a specific person or WHAT ARE RESEARCH INTERESTS: give full profile details.\n"
            "   - If the user asks for programs/courses list: output only names/titles of programs.\n"
            "   - NEVER give unsolicited details when only names are requested.\n"
            "7. Always mention the source/reference when citing specific information.\n"
            "8. MTECH ADMISSIONS & GATE CUTOFFS: GATE cutoff scores are NOT ranks (like top 4000-5000). JEE Advanced ranks (e.g. top 4000-5000) are ONLY for B.Tech. Do NOT apply B.Tech JEE Advanced ranks to M.Tech admissions or GATE cutoffs. When explaining GATE requirements or cutoffs, always mention that the numerical cutoffs are not fixed and vary each year depending on factors such as applicant pool size, vacancies, and department criteria.\n"
            "9. Answer queries directly, correctly, and concisely. Keep answers to the point without extra fluff.\n"
            "10. DESIGNATION ACCURACY (CRITICAL — ANTI-HALLUCINATION RULE):\n"
            "    - ONLY state a designation (Assistant Professor / Associate Professor / Professor) "
            "if it is EXPLICITLY written in the context FOR THAT SPECIFIC PERSON.\n"
            "    - NEVER borrow, infer, or guess a designation from another person's entry in the context.\n"
            "    - If no designation is visible for a person in the retrieved context, write ONLY their name — do NOT add any title.\n"
            "    - This is a ZERO-TOLERANCE rule. Mixing up Assistant Professor and Associate Professor is a serious factual error.\n"
            + topic_hint
            + lang_instr
        )

        # ── Build prompt ─────────────────────────────────────────
        history_block = ""
        if conversation_history:
            history_block = f"\nCONVERSATION HISTORY:\n{conversation_history}\n"

        web_note = ""
        if web_search_context:
            web_note = f"\n{web_search_context}\n"

        prompt = (
            f"CONTEXT FROM IIT JAMMU KNOWLEDGE BASE:\n{context}"
            f"{web_note}"
            f"{history_block}"
            f"\nUSER QUESTION: {query}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Use ONLY information from the CONTEXT above — never use outside knowledge\n"
            f"2. Copy numbers EXACTLY as they appear in context — NEVER calculate or estimate\n"
            f"3. If the context does not contain the exact answer, say that clearly\n"
            f"4. If the user asks for a future year (e.g., 2026) and the context contains data for the current academic session (e.g., 2025-26), assume it applies and use that data.\n"
            f"5. Distinguish Dean from Head of Department/HoD\n"
            f"6. Prefer live web-search context over cached Vector DB context when both are present\n"
            f"7. IMPORTANT DATE RULE: If the context mentions an application start/end date that has already passed (e.g., 'Start Date: 25 March 2025'), DO NOT include it in your answer. Instead write: 'Please check https://www.iitjammu.ac.in/admissions for current dates.'\n"
            f"8. RESPONSE DEPTH: If the question only asks WHO or LIST, return names + designations ONLY. Give research interests/email ONLY if explicitly asked.\n"
            f"9. Be concise but NEVER truncate a list — show ALL items.\n"
            f"10. GATE CUTOFFS vs JEE ADMISSIONS: Never output JEE Advanced ranks (e.g. top 4000-5000) for M.Tech admissions or GATE cutoff queries. Always explicitly state that the numerical cutoffs are not fixed and vary depending on vacancies and applicant pool. Separate undergraduate JEE ranks from postgraduate GATE requirements.\n"
            f"11. Be direct and to the point. Answer the question directly with correct details.\n"
            f"12. End with relevant source URL if available.\n"
            f"13. AVOID REPETITION: Never repeat the same name, item, or paragraph multiple times. Once an item is listed, do not list it again.\n"
            f"14. NO LOOPING: If you find yourself writing the same text over and over, stop immediately.\n"
            f"15. PREVIOUS YEARS / PAST YEARS DATA: When a user asks for 'previous years' or 'past years' or 'previous cutoff' scores/statistics, this includes the most recent year's data available in the context (e.g., Year 2025 or 2024-25), as it represents the most recently completed cycle relative to the upcoming/current admission cycle. Always present all available years (e.g., 2025, 2024, 2023) when asked for previous/past years.\n"
            f"16. TABLE FORMAT FOR DATA: When presenting numerical tables of data (such as GATE cutoffs, fees, or placement statistics), ALWAYS format them using clear, readable Markdown tables (with columns like category, branch/specialization, placement %, package, cutoff score, etc.) instead of long lists of bullet points. This ensures a premium, readable, and highly structured presentation for the user.\n"
            f"17. AMBIGUOUS CUTOFFS: If the user asks for 'cutoff' or 'cse cutoff' without specifying undergraduate (B.Tech) or postgraduate (M.Tech), you must provide BOTH: 1) the B.Tech JEE Advanced rank details (e.g., top 4000-5000) and 2) the M.Tech GATE cutoff scores (e.g., GEN: 643, etc. for 2025) formatted as Markdown tables. Do not omit either unless the query specifically requests B.Tech or M.Tech.\n"
            f"18. PRESERVE AND DO NOT CONVERT TABLES TO LISTS: If the retrieved context contains Markdown tables (such as for fees, cutoffs, or placements), you MUST output them as Markdown tables in your response. Never convert Markdown tables from the context into bullet points or list items."
        )

        return await self.generate(prompt, system_instruction=system)

    async def navigate_tree(self, query: str, node_context: str, children_list: str) -> dict:
        """Navigate a knowledge tree node (backward compatibility)."""
        prompt = (
            f"You are helping navigate an IIT Jammu knowledge base.\n\n"
            f"USER QUERY: {query}\n\n"
            f"CURRENT SECTION:\n{node_context}\n\n"
            f"AVAILABLE SUBSECTIONS:\n{children_list}\n\n"
            f"Reply ONLY with valid JSON (no markdown):\n"
            f'{{"action":"answer or drill","target":"answer or subsection title",'
            f'"confidence":0.0_to_1.0,"reason":"one line"}}'
        )
        try:
            raw = await self.generate(prompt)
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"navigate_tree parse failed ({e}). Raw: {raw[:150]}")
            return {"action": "answer", "target": raw, "confidence": 0.5, "reason": "parse-fallback"}


# ── Singleton ─────────────────────────────────────────────────────
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


# ── Backward compatibility aliases ────────────────────────────────
GeminiClient = LLMClient
get_gemini_client = get_llm_client
