"""
api/routes_chat.py — Chat & Suggestion Endpoints
==================================================
"""

import uuid
import logging
import traceback
import re

from fastapi import APIRouter, HTTPException, Request, Query
from config import get_settings

from api.models import (
    ChatRequest, ChatResponse, SourceNode,
    SuggestedQuestionsResponse, AutocompleteResponse,
    FeedbackRequest, FeedbackResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


@router.get("/suggestions", response_model=SuggestedQuestionsResponse)
async def suggestions():
    """Get suggested starter questions."""
    return SuggestedQuestionsResponse(questions=[
        "What are the B.Tech programs offered at IIT Jammu?",
        "What is the fee structure for B.Tech?",
        "How do I apply for M.Tech admission?",
        "What is the GATE cutoff for CSE at IIT Jammu?",
        "Tell me about hostel facilities and charges",
        "Who are the faculty members in Computer Science?",
        "What are the placement statistics?",
        "What scholarships are available for students?",
        "What is the eligibility for PhD programs?",
        "Which professors work on machine learning?",
    ])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Main chat endpoint — processes a user query through the RAG pipeline."""
    session_id = body.session_id or str(uuid.uuid4())

    # Detect language
    from language_handler import LanguageContext
    lang_ctx = LanguageContext(body.message, forced_lang=body.language)
    logger.info(f"[{session_id[:12]}] '{body.message[:60]}' | lang={lang_ctx.detected_lang}")

    # Sanitize input
    sanitized = _sanitize_input(body.message)

    try:
        from rag.engine import get_rag_engine
        engine = get_rag_engine()
        result = await engine.answer(
            sanitized,
            target_language=lang_ctx.detected_lang,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"[{session_id[:12]}] RAG error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI engine error: {type(e).__name__}: {e}")

    # Determine if web search was used
    from_web = any("web_search" in (s.node_id or "") for s in result.sources)

    return ChatResponse(
        answer=result.answer,
        detected_language=lang_ctx.detected_lang,
        sources=[
            SourceNode(
                title=re.sub(r"^(Auto FAQ|Curated FAQ|FAQ)\s*\d*\s*:\s*", "", s.title, flags=re.I).strip(),
                path=s.path,
                node_id=s.node_id,
                url=s.path if s.path.startswith("http") else None,
            )
            for s in result.sources
        ],
        confidence=round(result.confidence, 2),
        session_id=session_id,
        from_web_search=from_web,
    )


@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(q: str = Query(..., min_length=1, max_length=100)):
    """Get autocomplete suggestions for a query prefix."""
    try:
        from autocomplete.service import get_autocomplete_service
        service = get_autocomplete_service()
        results = service.search(q, limit=8)
        return AutocompleteResponse(suggestions=results)
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return AutocompleteResponse(suggestions=[])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest):
    """Handles feedback (thumbs up/down) for answers, especially from web searches."""
    logger.info(f"Feedback received: session={body.session_id}, msg={body.message_id}, positive={body.is_positive}")
    
    if body.is_positive and body.text_content:
        # User liked a web search answer -> inject into ChromaDB for future use
        try:
            from vectorstore.chroma_store import get_chroma_store
            from config import get_settings
            
            settings = get_settings()
            if settings.ddg_enabled:
                store = get_chroma_store()
                
                # We only want to inject if we know it came from the web
                # In production, we'd check if the sources were from DDG
                is_web_source = False
                source_urls = []
                if body.sources:
                    for src in body.sources:
                        if "web_search" in (src.node_id or ""):
                            is_web_source = True
                            if src.url:
                                source_urls.append(src.url)
                                
                if is_web_source:
                    logger.info("Injecting validated web search answer into ChromaDB.")
                    # Create a simple document block
                    doc_content = f"Q: {body.text_content[:100]}...\nA: {body.text_content}\nSources: {', '.join(source_urls)}"
                    
                    store.add_documents([{
                        "title": "Validated Web Answer",
                        "text": doc_content,
                        "source_url": source_urls[0] if source_urls else "User Feedback",
                        "doc_type": "validated_web_answer",
                        "department": "General",
                        "year": "2026", # Current year
                        "topic": "General"
                    }])
                    return FeedbackResponse(status="success", message="Feedback recorded and knowledge base updated.")
        except Exception as e:
            logger.error(f"Failed to process positive feedback injection: {e}")
            
    return FeedbackResponse(status="success", message="Feedback recorded.")


def _sanitize_input(text: str) -> str:
    """Sanitize user input — strip HTML tags and excessive whitespace."""
    # Remove HTML tags
    text = re.sub(r"<[^>]{1,200}>", "", text)
    # Remove script-like content
    text = re.sub(r"javascript:", "", text, flags=re.I)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
