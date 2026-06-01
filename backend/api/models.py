"""
api/models.py — Pydantic Schemas for IIT Jammu Chatbot API
============================================================
All request/response models in one place.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Request Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    language: Optional[str] = Field(None, description="Override language (ISO 639-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the B.Tech fee structure at IIT Jammu?",
                "session_id": "user_abc123",
                "language": None
            }
        }


# ── Response Models ───────────────────────────────────────────────

class SourceNode(BaseModel):
    title: str
    path: str
    node_id: str
    url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    detected_language: str
    sources: List[SourceNode] = []
    confidence: float = Field(..., ge=0.0, le=1.0)
    session_id: Optional[str] = None
    from_web_search: bool = False


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_nodes: int
    gemini_model: str
    chroma_docs: int = 0
    kg_nodes: int = 0
    memory_sessions: int = 0
    bm25_indexed: bool = False


class IndexStatsResponse(BaseModel):
    total_sections: int
    total_nodes: int
    top_level_sections: List[str]
    last_updated: Optional[str]


class SuggestedQuestionsResponse(BaseModel):
    questions: List[str]


class AutocompleteResponse(BaseModel):
    suggestions: List[dict]


class MemoryStatsResponse(BaseModel):
    active_sessions: int
    total_messages: int
    max_messages_per_session: int
    ttl_minutes: int


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    is_positive: bool
    text_content: Optional[str] = None
    sources: Optional[List[SourceNode]] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
