"""
api/routes_system.py — System, Debug & Knowledge Base Endpoints
================================================================
"""

import os
import logging
import traceback
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.models import HealthResponse, IndexStatsResponse
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check with component status."""
    settings = get_settings()
    try:
        from rag.engine import get_knowledge_tree
        tree = get_knowledge_tree()

        chroma_docs = 0
        kg_nodes = 0
        memory_sessions = 0
        bm25_ready = False

        try:
            from vectorstore.chroma_store import get_chroma_store
            chroma_docs = get_chroma_store().count()
        except Exception:
            pass
        try:
            from services.knowledge_graph import get_knowledge_graph
            kg_nodes = get_knowledge_graph().node_count()
        except Exception:
            pass
        try:
            from memory.conversation import get_memory
            memory_sessions = len(get_memory()._sessions)
        except Exception:
            pass

        return HealthResponse(
            status="ok",
            index_loaded=tree.count_nodes() > 0 or chroma_docs > 0,
            total_nodes=tree.count_nodes(),
            gemini_model=settings.llm_model,
            chroma_docs=chroma_docs,
            kg_nodes=kg_nodes,
            memory_sessions=memory_sessions,
            bm25_indexed=bm25_ready,
        )
    except Exception as e:
        return HealthResponse(
            status=f"degraded: {e}",
            index_loaded=False,
            total_nodes=0,
            gemini_model=settings.llm_model,
        )


@router.get("/index/stats", response_model=IndexStatsResponse, tags=["Knowledge Base"])
async def index_stats():
    """Get knowledge base index statistics."""
    from rag.engine import get_knowledge_tree
    tree = get_knowledge_tree()
    return IndexStatsResponse(
        total_sections=len(tree.get_root_nodes()),
        total_nodes=tree.count_nodes(),
        top_level_sections=tree.get_top_level_titles(),
        last_updated=tree.get_last_updated(),
    )


@router.get("/knowledge-graph/stats", tags=["Knowledge Base"])
async def kg_stats():
    """Get Knowledge Graph statistics."""
    try:
        from services.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        return kg.get_stats()
    except Exception as e:
        return {"error": str(e), "total_nodes": 0, "total_edges": 0}


@router.get("/chroma/stats", tags=["Knowledge Base"])
async def chroma_stats():
    """Get ChromaDB statistics."""
    try:
        from vectorstore.chroma_store import get_chroma_store
        store = get_chroma_store()
        return store.get_stats()
    except Exception as e:
        return {"error": str(e), "total_documents": 0}


@router.get("/memory/stats", tags=["System"])
async def memory_stats():
    """Get conversation memory statistics."""
    try:
        from memory.conversation import get_memory
        return get_memory().get_stats()
    except Exception as e:
        return {"error": str(e)}


# ── Debug endpoints ───────────────────────────────────────────────

@router.post("/debug/chat", tags=["Debug"])
async def debug_chat(request: Request):
    """Debug endpoint with full error tracebacks."""
    from api.models import ChatRequest
    body = ChatRequest(**(await request.json()))
    session_id = "debug_" + str(uuid.uuid4())[:8]

    from language_handler import LanguageContext
    lang_ctx = LanguageContext(body.message)

    debug_info = {
        "message": body.message,
        "detected_language": lang_ctx.detected_lang,
        "session_id": session_id,
        "llm_model": get_settings().llm_model,
        "chroma_docs": 0,
        "kg_nodes": 0,
        "tree_nodes": 0,
        "error": None,
        "traceback": None,
        "answer": None,
    }

    try:
        from rag.engine import get_knowledge_tree, get_rag_engine
        tree = get_knowledge_tree()
        debug_info["tree_nodes"] = tree.count_nodes()

        try:
            from vectorstore.chroma_store import get_chroma_store
            debug_info["chroma_docs"] = get_chroma_store().count()
        except Exception:
            pass
        try:
            from services.knowledge_graph import get_knowledge_graph
            debug_info["kg_nodes"] = get_knowledge_graph().node_count()
        except Exception:
            pass

        engine = get_rag_engine()
        result = await engine.answer(body.message, target_language=lang_ctx.detected_lang)
        debug_info["answer"] = result.answer
        debug_info["confidence"] = result.confidence
        debug_info["sources"] = [{"title": s.title, "path": s.path} for s in result.sources]

    except Exception as e:
        debug_info["error"] = f"{type(e).__name__}: {e}"
        debug_info["traceback"] = traceback.format_exc()

    return JSONResponse(content=debug_info)


@router.get("/debug/llm", tags=["Debug"])
async def debug_llm():
    """Test LLM connection directly."""
    settings = get_settings()
    info = {
        "llm_model": settings.llm_model,
        "ollama_url": settings.ollama_base_url,
        "status": None,
        "error": None,
        "response": None,
    }
    try:
        from llm.client import get_llm_client
        client = get_llm_client()
        response = await client.generate("Reply with exactly one word: WORKING")
        info["status"] = "ok"
        info["response"] = response
    except Exception as e:
        info["status"] = "error"
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()

    return JSONResponse(content=info, status_code=200 if info["status"] == "ok" else 500)
