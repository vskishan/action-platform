"""
Routes corresponding to AI Control Arm

Handles requests related to queries related to control arm cohort formation
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.schema.prediction_schema import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cohort", tags=["query"])

# The agent is initialised lazily on first request (see _get_agent).
_agent = None


def _get_agent():
    """Lazy-initialise the Agent singleton."""
    global _agent
    if _agent is None:
        from backend.app.llm.agent import Agent
        _agent = Agent()
    return _agent


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Handle a natural-language query from the frontend.

    The pipeline is:
    1. LLM classifies the query → engine + intent + parameters
    2. Route to the appropriate engine
    3. LLM formats the raw result into a human-readable answer
    """
    try:
        agent = _get_agent()
        result = agent.handle(request.query)

        return QueryResponse(
            query=result["query"],
            response=result["response"],
        )

    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )
