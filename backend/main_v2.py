"""
main_v2.py — IIT Jammu AI Assistant (Production Server)
========================================================
Slim FastAPI entry point. All logic delegated to modular packages.

Run:
  uvicorn main_v2:app --reload --port 8000

Or:
  python main_v2.py
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from config import get_settings
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('iitj-chatbot')

_NOISY_LOGGERS = [
    'httpx',
    'httpcore',
    'httpcore.http11',
    'httpcore.connection',
    'huggingface_hub',
    'huggingface_hub.utils._http',
    'huggingface_hub.file_download',
    'sentence_transformers',
    'sentence_transformers.SentenceTransformer',
    'transformers',
    'transformers.modeling_utils',
    'torch',
    'urllib3',
    'urllib3.connectionpool',
    'filelock',
    'asyncio'
]
for _noisy in _NOISY_LOGGERS:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import os as _os
_os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
_os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
_os.environ.setdefault('HF_HUB_VERBOSITY', 'error')

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('============================================================')
    logger.info('  🚀 IIT Jammu AI Assistant — Starting Up')
    logger.info('============================================================')
    
    chroma_store = None
    try:
        from vectorstore.chroma_store import get_chroma_store
        chroma_store = get_chroma_store()
        logger.info(f'  ✅ ChromaDB: {chroma_store.count()} documents')
    except Exception as e:
        logger.warning(f'  ⚠️  ChromaDB: {e}')
        
    knowledge_graph = None
    try:
        from services.knowledge_graph import get_knowledge_graph
        knowledge_graph = get_knowledge_graph()
        logger.info(f'  ✅ Knowledge Graph: {knowledge_graph.node_count()} nodes')
    except Exception as e:
        logger.warning(f'  ⚠️  Knowledge Graph: {e}')
        
    try:
        from rag.engine import get_rag_engine
        engine = get_rag_engine()
        logger.info('  ✅ RAG Engine ready')
    except Exception as e:
        logger.error(f'  ❌ RAG Engine: {e}')
        
    try:
        from autocomplete.service import get_autocomplete_service
        autocomplete = get_autocomplete_service()
        autocomplete.build_from_database(chroma_store)
        logger.info(f'  ✅ Autocomplete: {autocomplete._trie.count()} entries')
    except Exception as e:
        logger.warning(f'  ⚠️  Autocomplete: {e}')
        
    try:
        from memory.conversation import get_memory
        memory = get_memory()
        logger.info('  ✅ Conversation memory ready')
    except Exception as e:
        logger.warning(f'  ⚠️  Memory: {e}')
        
    if settings.scraper_enabled:
        try:
            from background_scraper import start_background_scraper
            from api.routes_scraper import set_stores
            set_stores(chroma_store, knowledge_graph)
            start_background_scraper(chroma_store=chroma_store, knowledge_graph=knowledge_graph)
            logger.info(f'  ✅ Scraper: every {settings.scraper_interval_hours}h')
        except Exception as e:
            logger.warning(f'  ⚠️  Scraper: {e}')
            
    logger.info('============================================================')
    logger.info('  ✅ All systems initialized — ready to serve')
    logger.info(f'  📍 http://{settings.backend_host}:{settings.backend_port}')
    logger.info('============================================================')
    
    yield
    
    logger.info('🛑 Shutting down IIT Jammu AI Assistant...')
    try:
        from background_scraper import stop_background_scraper
        stop_background_scraper()
    except Exception:
        pass

app = FastAPI(
    title='IIT Jammu AI Assistant',
    version='3.0.0',
    description='Production-ready Hybrid RAG Chatbot for IIT Jammu',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    allow_headers=['*']
)

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except ImportError:
    logger.info('slowapi not installed — rate limiting disabled')

from api.routes_chat import router as chat_router
from api.routes_system import router as system_router
from api.routes_scraper import router as scraper_router

app.include_router(chat_router)
app.include_router(system_router)
app.include_router(scraper_router)

@app.get('/', tags=['System'])
async def root():
    return {
        'name': 'IIT Jammu AI Assistant',
        'version': '3.0.0',
        'description': 'Hybrid RAG Chatbot for IIT Jammu',
        'docs': '/docs'
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'main_v2:app',
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level=settings.log_level.lower()
    )