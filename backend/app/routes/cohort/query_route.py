"""
Routes corresponding to AI Control Arm

Handles requests related to queries related to control arm cohort formation.

Uses the **ReAct Agent** — an agentic AI orchestrator that autonomously:
1. Reasons about what data is needed.
2. Selects and calls tools (survival engine, analytics engine, etc.).
3. Iterates until it has enough information.
4. Maintains conversation memory across turns via session_id.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.schema.prediction_schema import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cohort", tags=["query"])

# The ReAct agent is initialised lazily on first request.
_react_agent = None


def _get_react_agent():
    """Lazy-initialise the ReactAgent singleton."""
    global _react_agent
    if _react_agent is None:
        from backend.app.llm.react_agent import ReactAgent
        _react_agent = ReactAgent()
    return _react_agent


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Handle a natural-language query using the ReAct agent.

    The agentic pipeline:
    1. Agent reasons about the query (Thought).
    2. Agent autonomously selects and calls tools (Action).
    3. Agent observes tool results and decides next step (Observation).
    4. Repeats until it has enough data, then produces a final Answer.
    5. Conversation memory allows multi-turn follow-ups via session_id.
    """
    try:
        agent = _get_react_agent()
        result = agent.handle(
            user_query=request.query,
            session_id=request.session_id,
        )

        return QueryResponse(
            query=result["query"],
            response=result["response"],
            session_id=result.get("session_id"),
            tools_used=result.get("tools_used", []),
            steps=result.get("steps", 1),
        )

    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )
