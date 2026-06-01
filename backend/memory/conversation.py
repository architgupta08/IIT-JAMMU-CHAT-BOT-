"""
memory/conversation.py — Conversation Memory for Follow-Up Support
===================================================================
Stores recent messages per session so the chatbot can understand
follow-up questions like:
  User: "Who is Professor Subudhi?"
  User: "What are his research interests?"  ← needs context

Uses in-memory dict with TTL-based expiration.
Optional Redis backend if REDIS_URL is configured.
"""

import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str          # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """A conversation session with message history."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class ConversationMemory:
    """
    In-memory conversation store with TTL expiration.

    Features:
      - Stores last N messages per session
      - Auto-expires inactive sessions
      - Formats conversation history for LLM injection
      - Extracts context for follow-up resolution
    """

    def __init__(self):
        self.settings = get_settings()
        self._sessions: Dict[str, Session] = {}
        self._max_messages = self.settings.memory_max_messages
        self._ttl_seconds = self.settings.memory_ttl_minutes * 60
        logger.info(
            f"ConversationMemory initialized: "
            f"max_messages={self._max_messages}, ttl={self.settings.memory_ttl_minutes}min"
        )

    def _cleanup_expired(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_activity > self._ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired sessions")

    def _get_or_create(self, session_id: str) -> Session:
        """Get existing session or create a new one."""
        self._cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        session = self._sessions[session_id]
        session.last_activity = time.time()
        return session

    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to the session history.

        Args:
            session_id: Unique session identifier
            role: "user" or "assistant"
            content: Message text
        """
        session = self._get_or_create(session_id)
        session.messages.append(Message(role=role, content=content))

        # Trim to max messages
        if len(session.messages) > self._max_messages:
            session.messages = session.messages[-self._max_messages:]

    def get_history(self, session_id: str) -> List[Message]:
        """Get conversation history for a session."""
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id].messages

    def format_history_for_llm(self, session_id: str, max_turns: int = 4) -> str:
        """
        Format recent conversation history as a string for LLM context.

        Returns a formatted string like:
          User: Who is Professor Subudhi?
          Assistant: Dr. Badri N Subudhi is...
          User: What are his research interests?
        """
        messages = self.get_history(session_id)
        if not messages or len(messages) <= 1:
            return ""

        # Take the last N turns (excluding the current query which is the last message)
        recent = messages[-(max_turns * 2 + 1):-1] if len(messages) > 1 else []
        if not recent:
            return ""

        lines = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            # Truncate long assistant responses for context
            content = msg.content[:300] if msg.role == "assistant" else msg.content
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    def resolve_followup(self, session_id: str, query: str) -> str:
        """
        Resolve follow-up queries by injecting context from previous messages.

        If the query uses pronouns like "he", "she", "their", "that",
        prepend relevant context from the previous exchange.

        Example:
            Previous: "Who is Prof Subudhi?"
            Current:  "What are his research interests?"
            Returns:  "What are Prof Subudhi's research interests?"
        """
        import re

        # Check if this looks like a follow-up
        followup_signals = [
            r"\b(he|him|his|she|her|they|their|them|it|its)\b",
            r"\b(that|this|these|those)\b",
            r"\b(same|above|mentioned|previous)\b",
            r"\b(also|too|more|else|another)\b",
        ]

        is_followup = any(re.search(pattern, query, re.I) for pattern in followup_signals)
        if not is_followup:
            return query

        messages = self.get_history(session_id)
        if len(messages) < 2:
            return query

        # Find the last user query and assistant response
        last_user_query = ""
        last_topic = ""
        for msg in reversed(messages[:-1]):
            if msg.role == "user":
                last_user_query = msg.content
                # Extract likely topic/name from previous query
                name_match = re.search(
                    r"(?:who is|about|tell me about)\s+(.+?)[\?!.]?$",
                    msg.content, re.I
                )
                if not name_match:
                    name_match = re.search(
                        r"(?:what\s+(?:is|are)\s+the\s+)?(?:publications?|research\s+interests?|research\s+areas?|"
                        r"email|contact|designation|qualification|profile|department)\s+"
                        r"(?:of|for|by)\s+(?:professor|prof\.?|dr\.?)?[\s]*(.+?)[\?!.]?$",
                        msg.content, re.I
                    )
                
                if name_match:
                    last_topic = name_match.group(1).strip()
                break

        if last_topic:
            # Replace pronouns with the topic
            enhanced = re.sub(
                r"\b(his|her|their)\b",
                f"{last_topic}'s",
                query, flags=re.I
            )
            enhanced = re.sub(
                r"\b(he|she|they)\b",
                last_topic,
                enhanced, flags=re.I
            )
            if enhanced != query:
                logger.info(f"Follow-up resolved: '{query}' → '{enhanced}'")
                return enhanced

        # Fallback: prepend context
        if last_user_query:
            return f"(Context: previous question was '{last_user_query}') {query}"

        return query

    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_stats(self) -> Dict:
        """Return memory stats."""
        self._cleanup_expired()
        return {
            "active_sessions": len(self._sessions),
            "total_messages": sum(len(s.messages) for s in self._sessions.values()),
            "max_messages_per_session": self._max_messages,
            "ttl_minutes": self.settings.memory_ttl_minutes,
        }


# ── Singleton ─────────────────────────────────────────────────────
_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Get or create the singleton ConversationMemory."""
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
