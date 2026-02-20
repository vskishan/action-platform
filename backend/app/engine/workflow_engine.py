"""
Workflow Engine

Manages the lifecycle and state transitions for clinical-trial
workflows.  This engine is **CRUD + state-management only** — it does
NOT execute the actual screening, cohort, or monitoring work.

The heavy lifting is performed by the dedicated API endpoints
(``/api/screening``, ``/api/cohort``, ``/api/monitoring``).  The
frontend navigates to those pages when a stage is active and calls
back to update stage results via the ``update_stage`` method.

Stages
------
1. **Patient Screening** — run via ``/api/screening``
2. **Cohort Formation**  — run via ``/api/cohort``
3. **Cohort Monitoring** — run via ``/api/monitoring``
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from backend.app.schema.workflow_schema import (
    STAGE_ORDER,
    StageResult,
    StageStatus,
    Workflow,
    WorkflowCreateRequest,
    WorkflowStage,
    WorkflowStatus,
)
from backend.app.engine.workflow_store import WorkflowStore

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Stateless engine that mutates workflows stored in :class:`WorkflowStore`.

    All methods are pure state transitions — no subsystem execution.
    """

    def __init__(self) -> None:
        self._store = WorkflowStore()

    # ------------------------------------------------------------------
    # Workflow lifecycle
    # ------------------------------------------------------------------

    def create_workflow(self, request: WorkflowCreateRequest) -> Workflow:
        """Create and persist a new workflow, auto-started at stage 1.

        The workflow is immediately set to PAUSED at the first stage
        so the user can navigate directly to the stage page without
        an extra "Start" step.
        """
        first_stage = STAGE_ORDER[0]

        workflow = Workflow(
            id=str(uuid.uuid4()),
            name=request.name,
            description=request.description,
            trial_name=request.trial_name,
            status=WorkflowStatus.RUNNING,
            current_stage=first_stage,
            stages=self._store.new_stages(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata=request.metadata,
        )
        self._store.save(workflow)
        logger.info(
            "Created workflow %s (%s) — auto-started at stage '%s'.",
            workflow.id, workflow.name, first_stage,
        )
        return workflow

    def start_workflow(self, workflow_id: str) -> Workflow:
        """Start a workflow — sets current stage to the first stage (paused).

        Allowed from ``CREATED`` or ``PAUSED`` state (idempotent for
        auto-started workflows).
        """
        workflow = self._get_or_raise(workflow_id)

        if workflow.status not in (WorkflowStatus.CREATED, WorkflowStatus.PAUSED):
            raise ValueError(
                f"Cannot start workflow in '{workflow.status}' state. "
                "Only workflows in 'created' or 'paused' state can be started."
            )

        first_stage = STAGE_ORDER[0]
        workflow.status = WorkflowStatus.PAUSED
        workflow.current_stage = first_stage
        workflow.updated_at = datetime.now(timezone.utc)
        self._store.save(workflow)

        logger.info(
            "Workflow %s started — paused at stage '%s'. "
            "Navigate to the stage page to execute.",
            workflow_id,
            first_stage,
        )
        return workflow

    def resume_workflow(self, workflow_id: str) -> Workflow:
        """Resume a paused or failed workflow at its current stage.

        Resets a failed stage back to ``NOT_STARTED`` so the user can
        re-try from the dedicated stage page.
        """
        workflow = self._get_or_raise(workflow_id)

        if workflow.status not in (WorkflowStatus.PAUSED, WorkflowStatus.FAILED):
            raise ValueError(
                f"Cannot resume workflow in '{workflow.status}' state. "
                "Only workflows in 'paused' or 'failed' state can be resumed."
            )

        if workflow.current_stage is None:
            raise ValueError("Workflow has no current stage to resume from.")

        # If the current stage failed, reset it so the user can retry
        stage_result = workflow.stages[workflow.current_stage]
        if stage_result.status == StageStatus.FAILED:
            stage_result.status = StageStatus.NOT_STARTED
            stage_result.error = None

        workflow.status = WorkflowStatus.RUNNING
        workflow.updated_at = datetime.now(timezone.utc)
        self._store.save(workflow)

        logger.info(
            "Workflow %s resumed at stage '%s'. "
            "Navigate to the stage page to execute.",
            workflow_id,
            workflow.current_stage,
        )
        return workflow

    def advance_workflow(self, workflow_id: str) -> Workflow:
        """Move the workflow to the next stage without executing it.

        The current stage must already be COMPLETED.  Sets the next
        stage as ``current_stage`` and pauses the workflow so the
        caller can navigate to the stage page.
        """
        workflow = self._get_or_raise(workflow_id)

        if workflow.current_stage is None:
            raise ValueError("Workflow has not been started yet.")

        current_result = workflow.stages[workflow.current_stage]
        if current_result.status != StageStatus.COMPLETED:
            raise ValueError(
                f"Current stage '{workflow.current_stage}' is not completed "
                f"(status={current_result.status}). Complete it before advancing."
            )

        next_stage = self._next_stage(workflow.current_stage)
        if next_stage is None:
            # All stages done
            workflow.status = WorkflowStatus.COMPLETED
            workflow.current_stage = None
            workflow.updated_at = datetime.now(timezone.utc)
            self._store.save(workflow)
            return workflow

        workflow.current_stage = next_stage
        workflow.status = WorkflowStatus.RUNNING
        workflow.updated_at = datetime.now(timezone.utc)
        self._store.save(workflow)
        logger.info(
            "Workflow %s advanced to stage '%s' (running).",
            workflow_id,
            next_stage,
        )
        return workflow

    def pause_workflow(self, workflow_id: str) -> Workflow:
        """Pause a running workflow so it can be resumed later."""
        workflow = self._get_or_raise(workflow_id)

        if workflow.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PAUSED):
            raise ValueError(
                f"Cannot pause workflow in '{workflow.status}' state."
            )

        workflow.status = WorkflowStatus.PAUSED
        workflow.updated_at = datetime.now(timezone.utc)
        self._store.save(workflow)
        logger.info("Workflow %s paused at stage '%s'.", workflow_id, workflow.current_stage)
        return workflow

    # ------------------------------------------------------------------
    # Stage status updates  (called by frontend after using dedicated APIs)
    # ------------------------------------------------------------------

    def update_stage(
        self,
        workflow_id: str,
        stage: WorkflowStage,
        status: StageStatus,
        output_data: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Workflow:
        """Update a stage's status and results.

        This is the bridge between the dedicated API endpoints and the
        workflow tracker.  After the frontend runs screening / cohort /
        monitoring via their own endpoints, it calls this to record the
        outcome in the workflow.
        """
        workflow = self._get_or_raise(workflow_id)
        stage_result = workflow.stages[stage]

        stage_result.status = status

        if status == StageStatus.IN_PROGRESS:
            stage_result.started_at = datetime.now(timezone.utc)
            stage_result.error = None
            workflow.status = WorkflowStatus.RUNNING
        elif status == StageStatus.COMPLETED:
            stage_result.completed_at = datetime.now(timezone.utc)
            stage_result.error = None
        elif status == StageStatus.FAILED:
            stage_result.completed_at = datetime.now(timezone.utc)
            stage_result.error = error
            workflow.status = WorkflowStatus.FAILED
        elif status == StageStatus.NOT_STARTED:
            # Reset — move current_stage back to this stage so the
            # user can re-run it, and keep the workflow RUNNING.
            stage_result.started_at = None
            stage_result.completed_at = None
            stage_result.output_data = None
            stage_result.error = None
            workflow.current_stage = stage
            workflow.status = WorkflowStatus.RUNNING

            # Also reset every stage that comes *after* this one,
            # so we don't leave stale completed data downstream.
            stage_idx = STAGE_ORDER.index(stage)
            for later_stage in STAGE_ORDER[stage_idx + 1:]:
                later = workflow.stages[later_stage]
                later.status = StageStatus.NOT_STARTED
                later.started_at = None
                later.completed_at = None
                later.output_data = None
                later.error = None

        if output_data is not None:
            stage_result.output_data = output_data

        workflow.updated_at = datetime.now(timezone.utc)
        self._store.save(workflow)

        logger.info(
            "Workflow %s — stage '%s' updated to %s.",
            workflow_id, stage, status,
        )
        return workflow

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, workflow_id: str) -> Workflow:
        wf = self._store.get(workflow_id)
        if wf is None:
            raise KeyError(f"Workflow '{workflow_id}' not found.")
        return wf

    @staticmethod
    def _next_stage(current: WorkflowStage) -> Optional[WorkflowStage]:
        """Return the stage after *current*, or ``None`` if at the end."""
        idx = STAGE_ORDER.index(current)
        if idx + 1 < len(STAGE_ORDER):
            return STAGE_ORDER[idx + 1]
        return None
