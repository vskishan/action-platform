"""
Monitoring Schema

Pydantic models for the federated patient-monitoring query system.

Defines the monitoring query types, per-site aggregate results, and
the combined API response — mirroring the screening schema pattern
but for treatment-arm progress monitoring.

Also includes clinical-document models used by the MedGemma
extraction layer to parse unstructured clinician notes into
structured monitoring data.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Monitoring query categories
class MonitoringQueryType(str, Enum):
    """Supported monitoring query types."""

    ADVERSE_EVENTS   = "adverse_events"      # Safety / side-effect queries
    VISIT_PROGRESS   = "visit_progress"      # Visit adherence and completion
    RESPONSE_SUMMARY = "response_summary"    # Treatment response (RECIST-like)
    DROPOUT_SUMMARY  = "dropout_summary"     # Dropouts, withdrawals, retention
    LAB_TRENDS       = "lab_trends"          # Lab-value trajectories over time
    OVERALL_PROGRESS = "overall_progress"    # High-level trial progress dashboard


# Clinical-document models (for MedGemma extraction)

class ClinicalDocument(BaseModel):
    """A single clinical visit note / report."""

    visit_number: int = Field(..., description="Protocol visit number.")
    date: str = Field(..., description="Date of the clinical encounter (YYYY-MM-DD).")
    type: str = Field(
        "progress_note",
        description="Document type: progress_note, missed_visit_note, progress_note_with_imaging.",
    )
    text: str = Field(..., description="Free-text clinical narrative.")


class PatientClinicalNotes(BaseModel):
    """All clinical documents for a single patient at a site."""

    patient_id: str = Field(..., description="Patient identifier.")
    documents: list[ClinicalDocument] = Field(
        default_factory=list,
        description="Ordered list of clinical visit notes.",
    )


# Query sent from the server to each monitoring client
class MonitoringQuery(BaseModel):
    """The monitoring query that the central server encodes and distributes
    to each site via Flower's evaluate round."""

    trial_name: str = Field(
        ..., description="Name or ID of the clinical trial."
    )
    query_type: MonitoringQueryType = Field(
        ..., description="The type of monitoring query to execute."
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional parameters for the query, e.g. "
            "{'grade_threshold': 3} for AE queries, or "
            "{'lab_name': 'PSA'} for lab-trend queries."
        ),
    )
    natural_language_query: str = Field(
        "",
        description="Original user question (for context in LLM formatting).",
    )
    use_extraction: bool = Field(
        False,
        description=(
            "When True, each site uses MedGemma to extract structured "
            "monitoring data from raw clinical notes before aggregation. "
            "When False, sites use pre-processed structured JSON files."
        ),
    )


# Aggregate result returned by a single site
class SiteMonitoringResult(BaseModel):
    """Aggregate monitoring metrics that a single site returns.

    No patient-level data is included — only counts, rates, and
    statistical summaries.
    """

    site_id: str = Field(
        ..., description="Unique identifier for this clinical site."
    )
    query_type: MonitoringQueryType = Field(
        ..., description="The query type this result corresponds to."
    )
    total_patients_monitored: int = Field(
        ..., description="Number of patients being monitored at this site."
    )
    result_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Query-type-specific aggregate data (counts, rates, summaries).",
    )
    data_as_of: str = Field(
        "",
        description="ISO timestamp indicating when the site's data was last updated.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during computation.",
    )


class AggregateMonitoringResult(BaseModel):
    """Global aggregate monitoring metrics merged across all sites."""

    query_type: MonitoringQueryType = Field(
        ..., description="The query type this aggregate corresponds to."
    )
    total_sites: int = Field(
        0, description="Number of sites contributing to the aggregate."
    )
    total_patients_monitored: int = Field(
        0, description="Total monitored patients across contributing sites."
    )
    result_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged cross-site aggregate metrics for the query type.",
    )


# API request from the frontend
class MonitoringQueryRequest(BaseModel):
    """API request payload for monitoring queries."""

    trial_name: str = Field(
        ...,
        description="Name or ID of the clinical trial to query.",
        examples=["PROSTATE-CANCER"],
    )
    query: str = Field(
        ...,
        description="Natural-language monitoring question.",
        min_length=1,
        examples=[
            "What is the Grade 3+ adverse event rate across all sites?",
            "Show me patient visit adherence",
            "What are the most common side effects?",
        ],
    )
    use_extraction: bool = Field(
        False,
        description=(
            "When True, each federated site uses MedGemma to extract "
            "structured data from raw clinical notes before aggregation. "
            "When False (default), sites use pre-processed JSON files."
        ),
    )


# Full API response
class MonitoringQueryResponse(BaseModel):
    """Combined response returned after all sites report monitoring data."""

    trial_name: str
    query: str = Field(..., description="Original user query.")
    query_type: MonitoringQueryType
    site_results: list[SiteMonitoringResult] = Field(default_factory=list)
    global_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Cross-site aggregate metrics.",
    )
    response: str = Field(
        "",
        description="LLM-formatted natural-language answer.",
    )
    status: str = Field(
        "completed",
        description="Overall status: 'completed' | 'partial' | 'error'.",
    )
    message: str = ""
