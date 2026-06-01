"""
config/settings.py — Centralized Settings for IIT Jammu AI Assistant
=====================================================================
All environment variables in one place. Uses pydantic-settings for
validation and type coercion. Import anywhere via:

    from config import get_settings
    settings = get_settings()
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


def _project_root() -> Path:
    """Resolve the project root (parent of backend/)."""
    return Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Typed, validated settings loaded from .env file."""

    # ── Server ────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:5173"
    log_level: str = "INFO"

    # ── CORS ──────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── Data paths ────────────────────────────────────────────────
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    index_file: str = "data/processed/iitj_index.json"

    def resolve_path(self, rel: str) -> Path:
        """Resolve a relative path against the project root."""
        p = Path(rel)
        if p.is_absolute():
            return p
        # Try from project root first
        from_root = _project_root() / rel.lstrip("../")
        if from_root.exists():
            return from_root
        # Try from backend dir
        backend_dir = Path(__file__).resolve().parent.parent
        from_backend = backend_dir / rel
        if from_backend.exists():
            return from_backend
        # Default to project root
        return from_root

    # ── LLM Configuration ─────────────────────────────────────────
    llm_provider: str = "ollama"  # "ollama" or "bedrock"

    # ── LLM (Ollama) ─────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemma3:4b"
    llm_temperature: float = 0.2
    llm_timeout: int = 60
    llm_max_tokens: int = 1024
    llm_context_window: int = 4096

    # ── LLM (AWS Bedrock / Benchmark API) ─────────────────────────
    aws_region: str = "us-east-1"
    aws_bearer_token_bedrock: str = ""
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_max_tokens: int = 512
    bedrock_temperature: float = 0.1
    bedrock_top_p: float = 0.95
    bedrock_endpoint_url: str = "" # Optional override

    # ── ChromaDB ──────────────────────────────────────────────────
    chroma_db_path: str = "data/processed/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_chroma: int = 25

    # ── Knowledge Graph ───────────────────────────────────────────
    kg_file: str = "data/processed/knowledge_graph.graphml"
    top_k_kg: int = 5

    # ── BM25 ──────────────────────────────────────────────────────
    top_k_bm25: int = 20
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Reranker ──────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10
    reranker_enabled: bool = True

    # ── Retrieval scoring ─────────────────────────────────────────
    semantic_weight: float = 0.7
    recency_weight: float = 0.3
    recency_decay_per_year: float = 0.15
    min_similarity_threshold: float = 0.3
    max_text_per_node: int = 10000
    max_context_chars: int = 20000

    # ── DuckDuckGo fallback ───────────────────────────────────────
    ddg_enabled: bool = True
    ddg_max_results: int = 5
    ddg_timeout: int = 10
    min_results_before_ddg: int = 3
    force_ddg_for_fresh_queries: bool = True

    # ── Scraper ───────────────────────────────────────────────────
    scraper_enabled: bool = True
    scraper_run_on_startup: bool = True
    scraper_interval_hours: int = 6
    scraper_max_pages_per_run: int = 50
    scraper_delay_seconds: float = 2.0
    scraper_state_file: str = "data/processed/scraper_state.json"
    target_url: str = "https://www.iitjammu.ac.in"

    # ── Memory ────────────────────────────────────────────────────
    memory_max_messages: int = 10
    memory_ttl_minutes: int = 30
    redis_url: Optional[str] = None

    # ── Rate limiting ─────────────────────────────────────────────
    rate_limit_per_minute: int = 30

    # ── Multilingual ──────────────────────────────────────────────
    default_language: str = "en"
    supported_languages: str = "en,hi,ur,ks,pa,ta,te,bn,mr,gu"

    class Config:
        env_file = str(_project_root() / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance (cached)."""
    return Settings()
