"""
Workflow Schema

Pydantic models for the clinical-trial workflow orchestration system.

A workflow captures the end-to-end lifecycle of a clinical trial on
the ACTION platform, moving through three ordered stages:

    1. Patient Screening   - federated eligibility screening
    2. Cohort Formation    - analytics / survival queries to define treatment arms
    3. Cohort Monitoring   - federated treatment-arm monitoring

Each stage records its own status, timestamps, input/output data, and
errors so the workflow can be paused, inspected, and resumed at any
point.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Enums
class WorkflowStage(str, Enum):
    """Ordered stages of a clinical-trial workflow."""

    PATIENT_SCREENING = "patient_screening"
    COHORT_FORMATION  = "cohort_formation"
    COHORT_MONITORING  = "cohort_monitoring"


# Canonical ordering used by the engine to determine "next stage".
STAGE_ORDER: list[WorkflowStage] = [
    WorkflowStage.PATIENT_SCREENING,
    WorkflowStage.COHORT_FORMATION,
    WorkflowStage.COHORT_MONITORING,
]


class StageStatus(str, Enum):
    """Execution status of an individual stage."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    SKIPPED     = "skipped"


class WorkflowStatus(str, Enum):
    """Overall status of the workflow."""

    CREATED   = "created"
    RUNNING   = "running"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"


# Stage-level models
class StageResult(BaseModel):
    """Tracks execution state and data for a single workflow stage."""

    stage: WorkflowStage
    status: StageStatus = StageStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Optional[dict[str, Any]] = Field(
        None,
        description="Payload sent to the stage (e.g. screening criteria).",
    )
    output_data: Optional[dict[str, Any]] = Field(
        None,
        description="Result returned by the stage.",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the stage failed.",
    )


# Conversation models
class ConversationMessage(BaseModel):
    """A single message in a stage conversation."""

    role: str = Field(..., description="'user' or 'ai'.")
    text: str = Field(..., description="Message content.")
    timestamp: Optional[str] = Field(None, description="ISO timestamp.")


class ConversationUpdateRequest(BaseModel):
    """Payload to save/append conversation messages for a workflow stage."""

    messages: list[ConversationMessage] = Field(
        ..., description="Full conversation history to persist.",
    )


# Workflow entity
class Workflow(BaseModel):
    """Full representation of a clinical-trial workflow."""

    id: str = Field(..., description="Unique workflow identifier (UUID).")
    name: str = Field(..., description="Human-readable workflow name.")
    description: Optional[str] = Field(
        None, description="Free-text description of the trial / workflow."
    )
    trial_name: str = Field(
        ..., description="Trial name used across screening & monitoring."
    )
    status: WorkflowStatus = WorkflowStatus.CREATED
    current_stage: Optional[WorkflowStage] = Field(
        None, description="The stage the workflow is currently at."
    )
    stages: dict[WorkflowStage, StageResult] = Field(
        default_factory=dict,
        description="Per-stage execution state.",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary user-defined metadata for the trial.",
    )
    conversations: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Per-stage conversation history. Keys are stage names.",
    )


# API request / response models
class WorkflowCreateRequest(BaseModel):
    """Payload to create a new workflow."""

    name: str = Field(
        ...,
        description="Human-readable workflow name.",
        min_length=1,
        examples=["Phase-III Prostate Cancer Trial"],
    )
    description: Optional[str] = Field(
        None,
        description="Free-text description of the trial.",
        examples=["Multi-site RCT for novel androgen-deprivation therapy."],
    )
    trial_name: str = Field(
        ...,
        description="Trial identifier used in screening & monitoring.",
        min_length=1,
        examples=["PROSTATE-CANCER"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (sponsor, phase, etc.).",
    )


class StageUpdateRequest(BaseModel):
    """Payload to update a stage's status and results.

    Called by the frontend after executing the stage via the dedicated
    API endpoints (screening, cohort, monitoring).
    """

    status: StageStatus = Field(
        ...,
        description=(
            "New status for the stage. Typically 'in_progress' when starting, "
            "'completed' when done, or 'failed' on error."
        ),
    )
    output_data: Optional[dict[str, Any]] = Field(
        None,
        description="Result data returned by the dedicated API endpoint.",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the stage failed.",
    )


class WorkflowSummary(BaseModel):
    """Lightweight view of a workflow for list endpoints."""

    id: str
    name: str
    description: Optional[str] = None
    trial_name: str
    status: WorkflowStatus
    current_stage: Optional[WorkflowStage] = None
    stages_summary: dict[str, StageStatus] = Field(
        default_factory=dict,
        description="Stage name → status mapping.",
    )
    created_at: datetime
    updated_at: datetime


class WorkflowListResponse(BaseModel):
    """Response for listing all workflows."""

    workflows: list[WorkflowSummary] = Field(default_factory=list)
    total: int = 0


class WorkflowDetailResponse(BaseModel):
    """Detailed response for a single workflow."""

    workflow: Workflow


class WorkflowStageResponse(BaseModel):
    """Response when querying a single stage's result."""

    workflow_id: str
    stage: StageResult


class WorkflowActionResponse(BaseModel):
    """Generic response after a workflow action (start, advance, resume)."""

    workflow_id: str
    status: WorkflowStatus
    current_stage: Optional[WorkflowStage] = None
    message: str = ""
