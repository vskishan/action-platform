"""
Workflow Store — In-Memory Persistence Layer

Thread-safe, singleton in-memory store for clinical-trial workflows.
All data lives in a Python ``dict`` keyed by workflow ID and is lost
when the process exits.

In a production system this would be backed by a real database (e.g.PostgreSQL, MongoDB).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.app.schema.workflow_schema import (
    STAGE_ORDER,
    StageResult,
    StageStatus,
    Workflow,
    WorkflowStage,
    WorkflowStatus,
    WorkflowSummary,
)

logger = logging.getLogger(__name__)


class WorkflowStore:
    """Thread-safe singleton in-memory store for workflows."""

    _instance: Optional["WorkflowStore"] = None
    _lock = threading.Lock()

    # Statuses that denote an "active" (in-flight) workflow.
    _ACTIVE_STATUSES = frozenset({
        WorkflowStatus.RUNNING,
        WorkflowStatus.PAUSED,
        WorkflowStatus.CREATED,
    })

    def __new__(cls) -> "WorkflowStore":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._workflows: dict[str, Workflow] = {}
                inst._rw_lock = threading.RLock()
                cls._instance = inst
        return cls._instance

    # CRUD

    def save(self, workflow: Workflow) -> Workflow:
        """Insert or update a workflow."""
        with self._rw_lock:
            self._workflows[workflow.id] = workflow
            logger.info("Workflow %s saved (status=%s).", workflow.id, workflow.status)
        return workflow

    def get(self, workflow_id: str) -> Optional[Workflow]:
        """Retrieve a workflow by ID, or ``None`` if not found."""
        with self._rw_lock:
            return self._workflows.get(workflow_id)

    def list_all(self) -> list[WorkflowSummary]:
        """Return lightweight summaries for every workflow."""
        with self._rw_lock:
            return [self._to_summary(wf) for wf in self._workflows.values()]

    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow.  Returns ``True`` if it existed."""
        with self._rw_lock:
            return self._workflows.pop(workflow_id, None) is not None

    def count(self) -> int:
        """Return the total number of stored workflows."""
        with self._rw_lock:
            return len(self._workflows)

    # Active-workflow queries

    def has_active_workflow(self) -> bool:
        """Return ``True`` if any workflow is currently in progress.

        A workflow is considered *active* if its status is one of
        ``RUNNING``, ``PAUSED``, or ``CREATED``.
        """
        with self._rw_lock:
            return any(
                wf.status in self._ACTIVE_STATUSES
                for wf in self._workflows.values()
            )

    def get_active_workflow(self) -> Optional[Workflow]:
        """Return the currently active workflow, or ``None``."""
        with self._rw_lock:
            for wf in self._workflows.values():
                if wf.status in self._ACTIVE_STATUSES:
                    return wf
        return None

    # Filtered queries

    def get_by_trial(self, trial_name: str) -> list[WorkflowSummary]:
        """Return summaries for workflows matching a trial name."""
        with self._rw_lock:
            return [
                self._to_summary(wf)
                for wf in self._workflows.values()
                if wf.trial_name == trial_name
            ]

    # Factory & lifecycle utilities

    @staticmethod
    def new_stages() -> dict[WorkflowStage, StageResult]:
        """Create a fresh stages dict with every stage ``NOT_STARTED``."""
        return {
            stage: StageResult(stage=stage, status=StageStatus.NOT_STARTED)
            for stage in STAGE_ORDER
        }

    def reset(self) -> None:
        """Clear all workflows.  Intended for test teardown."""
        with self._rw_lock:
            self._workflows.clear()
            logger.info("Workflow store reset.")

    # Internal helpers

    @staticmethod
    def _to_summary(wf: Workflow) -> WorkflowSummary:
        """Convert a full :class:`Workflow` into a lightweight summary."""
        return WorkflowSummary(
            id=wf.id,
            name=wf.name,
            description=wf.description,
            trial_name=wf.trial_name,
            status=wf.status,
            current_stage=wf.current_stage,
            stages_summary={
                stage.value: result.status
                for stage, result in wf.stages.items()
            },
            created_at=wf.created_at,
            updated_at=wf.updated_at,
        )
