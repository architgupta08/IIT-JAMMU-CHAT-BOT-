"""
api/routes_scraper.py — Scraper Management Endpoints
=====================================================
"""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Scraper"])

# Module-level references set during startup
_chroma_store = None
_knowledge_graph = None


def set_stores(chroma_store=None, knowledge_graph=None):
    """Set store references (called from main.py startup)."""
    global _chroma_store, _knowledge_graph
    _chroma_store = chroma_store
    _knowledge_graph = knowledge_graph


@router.get("/scraper/status")
async def scraper_status():
    """Get background scraper status."""
    try:
        from background_scraper import get_scraper_status
        return get_scraper_status()
    except Exception as e:
        return {"error": str(e), "enabled": False}


@router.post("/scraper/trigger")
async def scraper_trigger():
    """Manually trigger a background scrape cycle."""
    try:
        from background_scraper import trigger_manual_scrape
        return trigger_manual_scrape(
            chroma_store=_chroma_store,
            knowledge_graph=_knowledge_graph,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraper trigger failed: {e}")
