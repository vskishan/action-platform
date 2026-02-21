"""
Job API Routes

Endpoints for submitting and polling background jobs.

Submitting a job immediately returns a ``job_id``.  The frontend
polls ``GET /api/jobs/{job_id}`` to check progress.  When the
job completes, the result is returned in the response body.

Endpoints
---------
POST  /api/jobs/submit          - submit a new job
GET   /api/jobs/{job_id}        - get job status / result
GET   /api/jobs/stage/{wf}/{s}  - get active job for a workflow+stage
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.engine.job_store import JobStore, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_store = JobStore()


# ── Request / response schemas ────────────────────────────

class JobSubmitRequest(BaseModel):
    workflow_id: str
    stage: str  # patient_screening | cohort_formation | cohort_monitoring
    payload: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class JobResponse(BaseModel):
    job_id: str
    status: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    description: Optional[str] = None


def _job_to_response(job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        result=job.result,
        error=job.error,
        description=job.description,
    )


# ── Worker functions for each stage ──────────────────────

def _screening_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run federated screening synchronously."""
    from backend.app.schema.screening_schema import ScreeningCriteria
    from backend.app.federated.central_server import CentralServer

    criteria = ScreeningCriteria(**payload)
    server = CentralServer()
    result = server.run_screening(criteria)
    return result.model_dump()


def _cohort_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run cohort query using the ReAct agent (agentic AI)."""
    from backend.app.llm.react_agent import ReactAgent

    agent = ReactAgent()
    result = agent.handle(
        user_query=payload["query"],
        session_id=payload.get("session_id"),
    )
    return result


def _monitoring_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run federated monitoring query synchronously."""
    from backend.app.federated.monitoring_server import MonitoringOrchestrator

    orchestrator = MonitoringOrchestrator()
    result = orchestrator.query(
        payload["trial_name"],
        payload["query"],
        use_extraction=payload.get("use_extraction", False),
    )
    return result.model_dump()


_STAGE_WORKERS = {
    "patient_screening": _screening_worker,
    "cohort_formation": _cohort_worker,
    "cohort_monitoring": _monitoring_worker,
}


# ── Endpoints ─────────────────────────────────────────────

@router.post("/submit", response_model=JobResponse, status_code=202)
def submit_job(request: JobSubmitRequest) -> JobResponse:
    """Submit a stage job for background execution.

    Returns immediately with the job id and ``pending`` status.
    """
    worker = _STAGE_WORKERS.get(request.stage)
    if worker is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown stage '{request.stage}'. "
                   f"Valid stages: {list(_STAGE_WORKERS.keys())}",
        )

    # Reject if a job is already running for this workflow+stage
    active = _store.get_active_for_stage(request.workflow_id, request.stage)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A job is already {active.status.value} for stage "
                   f"'{request.stage}'. Please wait for it to finish.",
        )

    job = _store.submit(
        workflow_id=request.workflow_id,
        stage=request.stage,
        payload=request.payload,
        worker_fn=worker,
        description=request.description,
    )
    return _job_to_response(job)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Poll a job's status and result."""
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _job_to_response(job)


@router.get("/stage/{workflow_id}/{stage}", response_model=Optional[JobResponse])
def get_active_stage_job(workflow_id: str, stage: str):
    """Get the latest job for a workflow+stage.

    Returns the job if one exists (any status), or 204 if none.
    """
    job = _store.get_latest_for_stage(workflow_id, stage)
    if job is None:
        return None
    return _job_to_response(job)
