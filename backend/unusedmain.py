"""
main.py — FastAPI Backend for IIT Jammu AI Assistant (Hybrid RAG)
===================================================================
Now with ChromaDB + Knowledge Graph + Background Scraper + DuckDuckGo fallback.
"""
import os
import logging
import time
import uuid
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import (
    ChatRequest, ChatResponse, HealthResponse,
    IndexStatsResponse, SuggestedQuestionsResponse, SourceNode
)
from rag_engine import get_rag_engine, get_knowledge_tree
from language_handler import LanguageContext

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Fix INDEX_FILE path — resolve ../data/... from project root ───
def _resolve_index_path():
    raw = os.getenv("INDEX_FILE", "data/processed/iitj_index.json")
    if raw.startswith("../"):
        resolved = raw[3:]  # strip leading ../
        os.environ["INDEX_FILE"] = resolved
        return resolved
    return raw

INDEX_FILE_RESOLVED = _resolve_index_path()

# ── Rate limiter ──────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Store references for scraper ──────────────────────────────────
_chroma_store = None
_knowledge_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chroma_store, _knowledge_graph

    logger.info("🚀 Starting IIT Jammu AI Assistant (Hybrid RAG) …")

    # ── Initialize legacy tree (for stats) ────────────────────────
    try:
        tree = get_knowledge_tree()
        logger.info(f"✅ Legacy knowledge tree: {tree.count_nodes()} nodes")
        if tree.count_nodes() == 0:
            logger.warning("⚠  Knowledge tree is EMPTY — run scraper/indexer.py first")
    except Exception as e:
        logger.error(f"⚠  Startup warning (tree): {e}")

    # ── Initialize ChromaDB ───────────────────────────────────────
    try:
        from chroma_store import get_chroma_store
        _chroma_store = get_chroma_store()
        logger.info(f"✅ ChromaDB: {_chroma_store.count()} documents")

        # Auto-migrate if ChromaDB is empty but tree has data
        if _chroma_store.count() == 0:
            try:
                tree = get_knowledge_tree()
                if tree.count_nodes() > 0:
                    logger.info("📥 ChromaDB is empty — auto-migrating seed data...")
                    from migrate_seed_data import migrate
                    migrate()
                    logger.info(f"✅ Migration complete: {_chroma_store.count()} documents in ChromaDB")
            except Exception as e:
                logger.warning(f"⚠  Auto-migration failed: {e}")
    except Exception as e:
        logger.warning(f"⚠  ChromaDB not available: {e}")

    # ── Initialize Knowledge Graph ────────────────────────────────
    try:
        from knowledge_graph import get_knowledge_graph
        _knowledge_graph = get_knowledge_graph()
        logger.info(
            f"✅ Knowledge Graph: {_knowledge_graph.node_count()} nodes, "
            f"{_knowledge_graph.edge_count()} edges"
        )
    except Exception as e:
        logger.warning(f"⚠  Knowledge Graph not available: {e}")

    # ── Initialize RAG engine ─────────────────────────────────────
    try:
        get_rag_engine()
        logger.info("✅ Hybrid RAG engine ready")
    except Exception as e:
        logger.error(f"⚠  RAG engine init error: {e}")

    # ── Start background scraper ──────────────────────────────────
    try:
        from background_scraper import start_background_scraper
        start_background_scraper(
            chroma_store=_chroma_store,
            knowledge_graph=_knowledge_graph,
        )
    except Exception as e:
        logger.warning(f"⚠  Background scraper not started: {e}")

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("🛑 Shutting down")
    try:
        from background_scraper import stop_background_scraper
        stop_background_scraper()
    except Exception:
        pass


app = FastAPI(
    title="IIT Jammu AI Assistant",
    description="Hybrid RAG chatbot: ChromaDB + Knowledge Graph + DuckDuckGo + Ollama",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # widen for dev; restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({time.time()-start:.2f}s)")
    return response


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    try:
        tree = get_knowledge_tree()

        # Include ChromaDB + KG stats
        chroma_docs = 0
        kg_nodes = 0
        try:
            from chroma_store import get_chroma_store
            chroma_docs = get_chroma_store().count()
        except Exception:
            pass
        try:
            from knowledge_graph import get_knowledge_graph
            kg_nodes = get_knowledge_graph().node_count()
        except Exception:
            pass

        return HealthResponse(
            status="ok",
            index_loaded=tree.count_nodes() > 0 or chroma_docs > 0,
            total_nodes=tree.count_nodes(),
            gemini_model=os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "llama3.2:3b"))
        )
    except Exception as e:
        return HealthResponse(
            status=f"degraded: {e}", index_loaded=False,
            total_nodes=0, gemini_model=os.getenv("LLM_MODEL", "unknown")
        )


@app.get("/index/stats", response_model=IndexStatsResponse, tags=["Knowledge Base"])
async def index_stats():
    tree = get_knowledge_tree()
    return IndexStatsResponse(
        total_sections=len(tree.get_root_nodes()),
        total_nodes=tree.count_nodes(),
        top_level_sections=tree.get_top_level_titles(),
        last_updated=tree.get_last_updated()
    )


@app.get("/suggestions", response_model=SuggestedQuestionsResponse, tags=["Chat"])
async def suggestions():
    return SuggestedQuestionsResponse(questions=[
        "What are the B.Tech programs offered at IIT Jammu?",
        "What is the fee structure for B.Tech 2024-25?",
        "How do I apply for M.Tech admission?",
        "What is the GATE cutoff for CSE at IIT Jammu?",
        "Tell me about hostel facilities and charges",
        "Who are the faculty members in Computer Science?",
        "What are the placement statistics for 2024?",
        "What scholarships are available for students?",
        "What is the eligibility for PhD programs?",
        "How to reach the IIT Jammu campus?",
    ])


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '30')}/minute")
async def chat(request: Request, body: ChatRequest):
    session_id = body.session_id or str(uuid.uuid4())
    lang_ctx = LanguageContext(body.message, forced_lang=body.language)
    logger.info(f"[{session_id}] '{body.message[:60]}' | lang={lang_ctx.detected_lang}")

    try:
        engine = get_rag_engine()
        result = await engine.answer(body.message, target_language=lang_ctx.detected_lang)
    except Exception as e:
        logger.error(f"[{session_id}] RAG engine exception:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI engine error: {type(e).__name__}: {e}")

    return ChatResponse(
        answer=result.answer,
        detected_language=lang_ctx.detected_lang,
        sources=[SourceNode(title=s.title, path=s.path, node_id=s.node_id) for s in result.sources],
        confidence=round(result.confidence, 2),
        session_id=session_id
    )


# ══════════════════════════════════════════════════════════════════
#  New endpoints: Scraper, KG, ChromaDB status
# ══════════════════════════════════════════════════════════════════

@app.get("/scraper/status", tags=["Scraper"])
async def scraper_status():
    """Get background scraper status."""
    try:
        from background_scraper import get_scraper_status
        return get_scraper_status()
    except Exception as e:
        return {"error": str(e), "enabled": False}


@app.post("/scraper/trigger", tags=["Scraper"])
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


@app.get("/knowledge-graph/stats", tags=["Knowledge Base"])
async def kg_stats():
    """Get Knowledge Graph statistics."""
    try:
        from knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        return kg.get_stats()
    except Exception as e:
        return {"error": str(e), "total_nodes": 0, "total_edges": 0}


@app.get("/chroma/stats", tags=["Knowledge Base"])
async def chroma_stats():
    """Get ChromaDB statistics."""
    try:
        from chroma_store import get_chroma_store
        store = get_chroma_store()
        return store.get_stats()
    except Exception as e:
        return {"error": str(e), "total_documents": 0}


# ══════════════════════════════════════════════════════════════════
#  Debug endpoints (unchanged from original)
# ══════════════════════════════════════════════════════════════════

@app.post("/debug/chat", tags=["Debug"])
async def debug_chat(request: Request, body: ChatRequest):
    """
    Debug endpoint — returns full error tracebacks instead of generic messages.
    Use this when /chat returns 'I encountered an error…' to see the real cause.
    Remove or protect this endpoint before going to production.
    """
    session_id = "debug_" + str(uuid.uuid4())[:8]
    lang_ctx = LanguageContext(body.message)
    debug_info = {
        "message": body.message,
        "detected_language": lang_ctx.detected_lang,
        "session_id": session_id,
        "index_file": INDEX_FILE_RESOLVED,
        "llm_model": os.getenv("LLM_MODEL"),
        "chroma_docs": 0,
        "kg_nodes": 0,
        "tree_nodes": 0,
        "error": None,
        "traceback": None,
        "answer": None,
    }

    try:
        tree = get_knowledge_tree()
        debug_info["tree_nodes"] = tree.count_nodes()

        try:
            from chroma_store import get_chroma_store
            debug_info["chroma_docs"] = get_chroma_store().count()
        except Exception:
            pass

        try:
            from knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            debug_info["kg_nodes"] = kg.node_count()
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
        logger.error(f"Debug chat error:\n{traceback.format_exc()}")

    return JSONResponse(content=debug_info)


@app.get("/debug/llm", tags=["Debug"])
async def debug_llm():
    """Test LLM (Ollama) connection directly and return real error if any."""
    info = {
        "llm_model": os.getenv("LLM_MODEL", "llama3.2:3b"),
        "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "status": None,
        "error": None,
        "response": None,
    }
    try:
        from gemini_client import get_gemini_client
        client = get_gemini_client()
        response = await client.generate("Reply with exactly one word: WORKING")
        info["status"] = "ok"
        info["response"] = response
    except Exception as e:
        info["status"] = "error"
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()

    return JSONResponse(content=info, status_code=200 if info["status"] == "ok" else 500)


# ── Error handlers ────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Not found", "path": str(request.url.path)})

@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", "8000")),
        reload=True
    )
