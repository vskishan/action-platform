"""
Conversation Memory

Session-based conversation memory that allows the ReAct agent to
maintain context across multiple user turns.

Features
--------
- In-memory store keyed by ``session_id`` (UUID string).
- Stores a rolling window of messages (user / assistant / tool).
- Provides context injection into the ReAct prompt.
- Auto-trims old messages to stay within token budgets.
- Thread-safe via a simple lock (one agent instance per app).
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Message types

MessageRole = Literal["user", "assistant", "tool_call", "tool_result", "system"]


@dataclass
class Message:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A full conversation session."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Memory store

class ConversationMemory:
    """In-memory conversation store with session management.

    Parameters
    ----------
    max_messages_per_session : int
        Maximum number of messages to retain per session.  Older
        messages are summarised or dropped when the limit is reached.
    max_sessions : int
        Maximum number of active sessions.  Oldest sessions are
        evicted when the limit is exceeded (LRU-style).

    Usage::

        memory = ConversationMemory()
        sid = memory.create_session()
        memory.add_message(sid, Message(role="user", content="Hi"))
        history = memory.get_context(sid)
    """

    # Singleton
    _instance: ConversationMemory | None = None

    def __init__(
        self,
        max_messages_per_session: int = 50,
        max_sessions: int = 200,
    ) -> None:
        self._sessions: dict[str, Conversation] = {}
        self._max_messages = max_messages_per_session
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls, **kwargs) -> "ConversationMemory":
        """Return (and optionally create) the shared singleton."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    # Session lifecycle

    def create_session(self) -> str:
        """Create a new conversation session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._evict_if_needed()
            self._sessions[session_id] = Conversation(session_id=session_id)
        logger.info("Created conversation session: %s", session_id)
        return session_id

    def get_or_create_session(self, session_id: str | None) -> str:
        """Return an existing session or create a new one.

        If *session_id* is ``None`` or not found, a new session is
        created and its ID is returned.
        """
        if session_id and session_id in self._sessions:
            return session_id
        return self.create_session()

    def session_exists(self, session_id: str) -> bool:
        """Return ``True`` if the session exists."""
        return session_id in self._sessions

    def delete_session(self, session_id: str) -> None:
        """Remove a session and all its messages."""
        with self._lock:
            self._sessions.pop(session_id, None)

    # Message management

    def add_message(self, session_id: str, message: Message) -> None:
        """Append a message to the session, trimming if needed."""
        with self._lock:
            conv = self._sessions.get(session_id)
            if conv is None:
                logger.warning(
                    "Session %s not found — creating on the fly.", session_id
                )
                conv = Conversation(session_id=session_id)
                self._sessions[session_id] = conv

            conv.messages.append(message)
            conv.updated_at = datetime.now(timezone.utc).isoformat()

            # Trim oldest messages if over limit (keep system messages).
            if len(conv.messages) > self._max_messages:
                self._trim(conv)

    def add_user_message(self, session_id: str, content: str) -> None:
        """Convenience: add a user message."""
        self.add_message(session_id, Message(role="user", content=content))

    def add_assistant_message(self, session_id: str, content: str) -> None:
        """Convenience: add an assistant message."""
        self.add_message(session_id, Message(role="assistant", content=content))

    def add_tool_call(
        self,
        session_id: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> None:
        """Record that the agent called a tool."""
        self.add_message(
            session_id,
            Message(
                role="tool_call",
                content=f"Called tool: {tool_name}",
                metadata={"tool": tool_name, "parameters": parameters},
            ),
        )

    def add_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        """Record the result of a tool invocation."""
        # Truncate very long results to keep context reasonable.
        result_str = json.dumps(result, indent=2, default=str)
        if len(result_str) > 3000:
            result_str = result_str[:3000] + "\n... (truncated)"

        self.add_message(
            session_id,
            Message(
                role="tool_result",
                content=result_str,
                metadata={"tool": tool_name},
            ),
        )

    # Context retrieval

    def get_messages(self, session_id: str) -> list[Message]:
        """Return the raw message list for a session."""
        conv = self._sessions.get(session_id)
        return list(conv.messages) if conv else []

    def get_context_for_prompt(
        self,
        session_id: str,
        max_messages: int = 20,
    ) -> str:
        """Render recent conversation history as a text block for
        injection into the ReAct system prompt.

        Only user and assistant messages are included (tool calls are
        summarised).  This keeps the context concise.
        """
        conv = self._sessions.get(session_id)
        if not conv or not conv.messages:
            return ""

        # Take the most recent messages.
        recent = conv.messages[-max_messages:]

        lines: list[str] = ["## Conversation History"]
        for msg in recent:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"Assistant: {msg.content}")
            elif msg.role == "tool_call":
                tool = msg.metadata.get("tool", "unknown")
                lines.append(f"[Agent called tool: {tool}]")
            elif msg.role == "tool_result":
                tool = msg.metadata.get("tool", "unknown")
                # Show only a brief summary
                content_preview = msg.content[:200]
                if len(msg.content) > 200:
                    content_preview += "..."
                lines.append(f"[Tool {tool} returned: {content_preview}]")

        return "\n".join(lines)

    def get_chat_messages_for_llm(
        self,
        session_id: str,
        max_turns: int = 10,
    ) -> list[dict[str, str]]:
        """Return conversation history formatted for the Ollama
        ``messages`` list (user/assistant pairs only).

        This is used to provide multi-turn context to ``chat_messages()``.
        """
        conv = self._sessions.get(session_id)
        if not conv:
            return []

        result: list[dict[str, str]] = []
        # Filter to user/assistant messages only, take last N turns
        relevant = [
            m for m in conv.messages
            if m.role in ("user", "assistant")
        ]
        for msg in relevant[-(max_turns * 2):]:
            role = "user" if msg.role == "user" else "assistant"
            result.append({"role": role, "content": msg.content})

        return result

    # Internal helpers

    def _trim(self, conv: Conversation) -> None:
        """Remove oldest non-system messages to stay within limits."""
        excess = len(conv.messages) - self._max_messages
        if excess <= 0:
            return

        # Keep system messages, drop oldest user/assistant/tool messages
        kept: list[Message] = []
        dropped = 0
        for msg in conv.messages:
            if dropped < excess and msg.role != "system":
                dropped += 1
                continue
            kept.append(msg)

        conv.messages = kept
        logger.debug(
            "Trimmed %d messages from session %s",
            dropped,
            conv.session_id,
        )

    def _evict_if_needed(self) -> None:
        """Evict the oldest session if at capacity."""
        if len(self._sessions) >= self._max_sessions:
            # Find the oldest by updated_at
            oldest_id = min(
                self._sessions,
                key=lambda sid: self._sessions[sid].updated_at,
            )
            del self._sessions[oldest_id]
            logger.info("Evicted oldest session: %s", oldest_id)
