"""
Workflow API Routes

REST endpoints for creating, inspecting, and controlling clinical-trial
workflows.  Each workflow progresses through three stages:

    patient_screening → cohort_formation → cohort_monitoring

These endpoints handle **CRUD and state management only**.  The actual
stage work (screening, cohort queries, monitoring) is performed by
the dedicated API endpoints.  The frontend calls back here to record
stage results via ``PUT /api/workflow/{id}/stage/{stage}``.

Endpoints
---------
POST   /api/workflow                       - create a workflow
GET    /api/workflow                       - list all workflows
GET    /api/workflow/{id}                  - get full workflow detail
DELETE /api/workflow/{id}                  - delete a workflow
POST   /api/workflow/{id}/start            - start (paused at stage 1)
POST   /api/workflow/{id}/resume           - resume a paused/failed workflow
POST   /api/workflow/{id}/advance          - move to next stage (paused)
POST   /api/workflow/{id}/pause            - pause a running workflow
PUT    /api/workflow/{id}/stage/{stage}    - update stage status & results
GET    /api/workflow/{id}/stage/{stage}    - get a stage's result
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.engine.workflow_engine import WorkflowEngine
from backend.app.engine.workflow_store import WorkflowStore
from backend.app.schema.workflow_schema import (
    ConversationUpdateRequest,
    StageUpdateRequest,
    WorkflowActionResponse,
    WorkflowCreateRequest,
    WorkflowDetailResponse,
    WorkflowListResponse,
    WorkflowRecommendationResponse,
    WorkflowStage,
    WorkflowStageResponse,
    WorkflowSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflow", tags=["workflow"])

# Shared engine & store instances (singletons under the hood)
_engine = WorkflowEngine()
_store = WorkflowStore()


# ── CRUD ──────────────────────────────────────────────────────────

@router.post("", response_model=WorkflowDetailResponse, status_code=201)
def create_workflow(request: WorkflowCreateRequest) -> WorkflowDetailResponse:
    """Create a new clinical-trial workflow.

    Returns 409 if another workflow is already in progress.
    """
    try:
        workflow = _engine.create_workflow(request)
        return WorkflowDetailResponse(workflow=workflow)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to create workflow: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("", response_model=WorkflowListResponse)
def list_workflows(trial_name: str | None = None) -> WorkflowListResponse:
    """List all workflows, optionally filtered by trial name."""
    if trial_name:
        items = _store.get_by_trial(trial_name)
    else:
        items = _store.list_all()
    return WorkflowListResponse(workflows=items, total=len(items))


@router.get("/{workflow_id}", response_model=WorkflowDetailResponse)
def get_workflow(workflow_id: str) -> WorkflowDetailResponse:
    """Retrieve full detail for a single workflow."""
    workflow = _store.get(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    return WorkflowDetailResponse(workflow=workflow)


@router.delete("/{workflow_id}", status_code=204)
def delete_workflow(workflow_id: str) -> None:
    """Delete a workflow."""
    if not _store.delete(workflow_id):
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")


# ── Lifecycle controls (state transitions only) ───────────────────

@router.post("/{workflow_id}/start", response_model=WorkflowActionResponse)
def start_workflow(workflow_id: str) -> WorkflowActionResponse:
    """Start the workflow — pauses at stage 1 (patient_screening).

    Navigate to the Screening page to execute the stage.
    """
    try:
        workflow = _engine.start_workflow(workflow_id)
        return WorkflowActionResponse(
            workflow_id=workflow.id,
            status=workflow.status,
            current_stage=workflow.current_stage,
            message=f"Workflow started — ready at '{workflow.current_stage}'. Navigate to the stage page to execute.",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to start workflow %s: %s", workflow_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{workflow_id}/resume", response_model=WorkflowActionResponse)
def resume_workflow(workflow_id: str) -> WorkflowActionResponse:
    """Resume a paused or failed workflow.

    If the current stage previously failed, its status is reset to
    ``not_started`` so the user can re-try from the dedicated page.
    """
    try:
        workflow = _engine.resume_workflow(workflow_id)
        return WorkflowActionResponse(
            workflow_id=workflow.id,
            status=workflow.status,
            current_stage=workflow.current_stage,
            message=f"Workflow resumed at stage '{workflow.current_stage}'. Navigate to the stage page to execute.",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to resume workflow %s: %s", workflow_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{workflow_id}/advance", response_model=WorkflowActionResponse)
def advance_workflow(workflow_id: str) -> WorkflowActionResponse:
    """Advance to the next stage (current stage must be COMPLETED).

    The workflow moves to the next stage in **paused** state so the
    user can navigate to the stage page to execute it.
    """
    try:
        workflow = _engine.advance_workflow(workflow_id)
        return WorkflowActionResponse(
            workflow_id=workflow.id,
            status=workflow.status,
            current_stage=workflow.current_stage,
            message=(
                f"Advanced. Current stage: {workflow.current_stage or 'all completed'}."
            ),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to advance workflow %s: %s", workflow_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{workflow_id}/pause", response_model=WorkflowActionResponse)
def pause_workflow(workflow_id: str) -> WorkflowActionResponse:
    """Pause a running workflow."""
    try:
        workflow = _engine.pause_workflow(workflow_id)
        return WorkflowActionResponse(
            workflow_id=workflow.id,
            status=workflow.status,
            current_stage=workflow.current_stage,
            message=f"Workflow paused at stage: {workflow.current_stage}.",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to pause workflow %s: %s", workflow_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Per-stage status updates & inspection ─────────────────────────

@router.put(
    "/{workflow_id}/stage/{stage}",
    response_model=WorkflowActionResponse,
)
def update_stage(
    workflow_id: str,
    stage: WorkflowStage,
    body: StageUpdateRequest,
) -> WorkflowActionResponse:
    """Update a stage's status and results.

    Called by the frontend after running the stage via the dedicated
    API endpoint (screening, cohort, or monitoring).  This records the
    outcome in the workflow without duplicating execution.
    """
    try:
        workflow = _engine.update_stage(
            workflow_id,
            stage,
            status=body.status,
            output_data=body.output_data,
            error=body.error,
        )
        stage_result = workflow.stages[stage]
        return WorkflowActionResponse(
            workflow_id=workflow.id,
            status=workflow.status,
            current_stage=workflow.current_stage,
            message=f"Stage '{stage.value}' updated to '{stage_result.status.value}'.",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "Failed to update stage %s for workflow %s: %s",
            stage, workflow_id, exc,
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/{workflow_id}/stage/{stage}",
    response_model=WorkflowStageResponse,
)
def get_stage_result(
    workflow_id: str,
    stage: WorkflowStage,
) -> WorkflowStageResponse:
    """Get the result for a specific stage of a workflow."""
    workflow = _store.get(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    stage_result = workflow.stages.get(stage)
    if stage_result is None:
        raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found.")

    return WorkflowStageResponse(workflow_id=workflow.id, stage=stage_result)


# ── Per-stage conversation persistence ────────────────────────────

@router.get("/{workflow_id}/stage/{stage}/conversation")
def get_stage_conversation(
    workflow_id: str,
    stage: WorkflowStage,
):
    """Retrieve the persisted conversation for a workflow stage."""
    workflow = _store.get(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    messages = workflow.conversations.get(stage.value, [])
    return {"workflow_id": workflow_id, "stage": stage.value, "messages": messages}


@router.put("/{workflow_id}/stage/{stage}/conversation")
def save_stage_conversation(
    workflow_id: str,
    stage: WorkflowStage,
    body: ConversationUpdateRequest,
):
    """Persist the full conversation history for a workflow stage."""
    workflow = _store.get(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    workflow.conversations[stage.value] = [m.model_dump() for m in body.messages]
    _store.save(workflow)
    return {
        "workflow_id": workflow_id,
        "stage": stage.value,
        "message": f"Conversation saved ({len(body.messages)} messages).",
    }


# ── Autonomous Workflow Orchestration ─────────────────────────────

@router.post(
    "/{workflow_id}/analyze/{stage}",
    response_model=WorkflowRecommendationResponse,
)
def analyze_stage(
    workflow_id: str,
    stage: WorkflowStage,
    auto_advance: bool = False,
) -> WorkflowRecommendationResponse:
    """Analyse a completed stage and get an AI recommendation.

    The workflow orchestration agent examines the stage results and
    returns a recommendation (PROCEED / ADJUST / REVIEW / ALERT)
    with quality scoring, anomaly detection, and focus areas for
    the next stage.

    If ``auto_advance=True`` and the recommendation is PROCEED,
    the workflow is automatically advanced to the next stage.
    """
    from backend.app.llm.workflow_orchestrator import WorkflowOrchestrationAgent

    try:
        orchestrator = WorkflowOrchestrationAgent()
        recommendation, was_advanced = orchestrator.analyse_and_recommend(
            workflow_id=workflow_id,
            completed_stage=stage,
            auto_advance=auto_advance,
        )
        return WorkflowRecommendationResponse(
            workflow_id=workflow_id,
            recommendation=recommendation,
            auto_advanced=was_advanced,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "Failed to analyse stage %s for workflow %s: %s",
            stage, workflow_id, exc,
        )
        raise HTTPException(status_code=500, detail=str(exc))
