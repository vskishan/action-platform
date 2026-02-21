"""
Prediction Schema

Pydantic models for the survival/progression - prediction request / response cycle.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A natural-language query from the frontend."""

    query: str = Field(
        ...,
        description="The user's natural-language question.",
        min_length=1,
        examples=["What is the average survival rate of patients with the age of 60 and above?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Conversation session ID for multi-turn context.  "
            "Omit or pass null for the first message — the response "
            "will include a session_id for follow-ups."
        ),
    )


class ClassificationResult(BaseModel):
    """The LLM's intent-classification output."""

    engine: str = Field(..., description="Target engine: 'survival', 'analytics', or 'none'.")
    intent: str = Field(..., description="Fine-grained intent label.")
    parameters: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Full response returned to the frontend."""

    query: str = Field(..., description="Original user query.")
    response: str = Field(
        ..., description="Formatted natural-language answer."
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for follow-up queries in the same conversation.",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Tools the agent called during this query.",
    )
    steps: int = Field(
        default=1,
        description="Number of reasoning steps the agent took.",
    )
