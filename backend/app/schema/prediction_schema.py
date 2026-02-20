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
