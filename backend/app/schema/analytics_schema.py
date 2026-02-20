"""
Analytics Schema

Pydantic models for analytics-specific payloads
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyticsQueryRequest(BaseModel):
    """Direct analytics query (bypasses LLM classification)."""

    intent: str = Field(
        ...,
        description="Analytics intent label.",
        examples=["cohort_summary"],
    )
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnalyticsQueryResponse(BaseModel):
    """Raw analytics engine output."""

    intent: str
    data: dict[str, Any]
