"""
Job Store

Lightweight in-memory background-job system for long-running stage
operations (screening, cohort queries, monitoring).

When a stage page submits a request it receives a ``job_id``
immediately.  The actual work runs in a background thread and the
frontend polls ``GET /api/jobs/{job_id}`` until it completes.

This lets users navigate away and come back without losing work.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    workflow_id: str
    stage: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    request_payload: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    description: Optional[str] = None  # e.g. "Natural language screening"


class JobStore:
    """Thread-safe singleton job store."""

    _instance: Optional["JobStore"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "JobStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._jobs: dict[str, Job] = {}
                cls._instance._stage_jobs: dict[str, str] = {}  # "wf_id:stage" → job_id
            return cls._instance

    # ── Create & run ──────────────────────────────────────

    def submit(
        self,
        workflow_id: str,
        stage: str,
        payload: dict[str, Any],
        worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
        description: str = "",
    ) -> Job:
        """Submit a job for background execution.

        Parameters
        ----------
        workflow_id : str
            Workflow this job belongs to.
        stage : str
            Stage key (``patient_screening``, ``cohort_formation``,
            ``cohort_monitoring``).
        payload : dict
            The request body forwarded to *worker_fn*.
        worker_fn : callable
            Synchronous function ``(payload) → result_dict``.
            Will be called in a daemon thread.
        description : str
            Human-readable label for the job.

        Returns
        -------
        Job
            The newly created job (status = ``pending``).
        """
        job = Job(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            stage=stage,
            request_payload=payload,
            description=description,
        )

        with self._lock:
            self._jobs[job.id] = job
            self._stage_jobs[f"{workflow_id}:{stage}"] = job.id

        # Launch background thread
        thread = threading.Thread(
            target=self._run,
            args=(job.id, worker_fn, payload),
            daemon=True,
        )
        thread.start()

        logger.info(
            "Job %s submitted for workflow %s stage %s (%s).",
            job.id, workflow_id, stage, description,
        )
        return job

    def _run(
        self,
        job_id: str,
        worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
        payload: dict[str, Any],
    ) -> None:
        """Execute the worker in a background thread."""
        job = self._jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)

        try:
            result = worker_fn(payload)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            logger.info("Job %s completed successfully.", job_id)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            logger.exception("Job %s failed: %s", job_id, exc)

    # ── Queries ───────────────────────────────────────────

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def get_active_for_stage(self, workflow_id: str, stage: str) -> Optional[Job]:
        """Return the most recent job for a workflow+stage if it's still
        pending or running.  Returns ``None`` if there is no active job."""
        with self._lock:
            job_id = self._stage_jobs.get(f"{workflow_id}:{stage}")
        if not job_id:
            return None
        job = self._jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            return job
        return None

    def get_latest_for_stage(self, workflow_id: str, stage: str) -> Optional[Job]:
        """Return the most recent job for a workflow+stage regardless of
        status."""
        with self._lock:
            job_id = self._stage_jobs.get(f"{workflow_id}:{stage}")
        if not job_id:
            return None
        return self._jobs.get(job_id)
