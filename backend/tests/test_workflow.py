"""
Tests for the Workflow Engine & Store

Verifies the in-memory workflow lifecycle: create → start → advance →
resume → complete.  The engine is **CRUD + state-management only** —
no subsystem execution occurs.  Stage results are recorded via
``update_stage`` (simulating the frontend calling back after using the
dedicated API endpoints).
"""

from __future__ import annotations

import pytest

from backend.app.schema.workflow_schema import (
    STAGE_ORDER,
    StageStatus,
    WorkflowCreateRequest,
    WorkflowStage,
    WorkflowStatus,
)
from backend.app.engine.workflow_store import WorkflowStore
from backend.app.engine.workflow_engine import WorkflowEngine


# Fixtures

@pytest.fixture(autouse=True)
def _clean_store():
    """Reset the singleton store before each test."""
    store = WorkflowStore()
    store.reset()
    yield
    store.reset()


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def store() -> WorkflowStore:
    return WorkflowStore()


def _make_request(**overrides) -> WorkflowCreateRequest:
    defaults = dict(
        name="Test Trial",
        description="A test workflow",
        trial_name="TEST-001",
        metadata={"phase": "III"},
    )
    defaults.update(overrides)
    return WorkflowCreateRequest(**defaults)


def _complete_workflow(engine: WorkflowEngine, wf_id: str) -> None:
    """Helper to run a workflow through all stages to COMPLETED."""
    for stage in STAGE_ORDER:
        engine.update_stage(wf_id, stage, StageStatus.COMPLETED, output_data={})
        engine.advance_workflow(wf_id)


# Store tests

class TestWorkflowStore:
    def test_save_and_get(self, engine: WorkflowEngine, store: WorkflowStore):
        wf = engine.create_workflow(_make_request())
        fetched = store.get(wf.id)
        assert fetched is not None
        assert fetched.id == wf.id
        assert fetched.name == "Test Trial"

    def test_list_all(self, engine: WorkflowEngine, store: WorkflowStore):
        wf_a = engine.create_workflow(_make_request(name="A"))
        # Complete first workflow before creating the second
        _complete_workflow(engine, wf_a.id)
        engine.create_workflow(_make_request(name="B"))
        items = store.list_all()
        assert len(items) == 2

    def test_delete(self, engine: WorkflowEngine, store: WorkflowStore):
        wf = engine.create_workflow(_make_request())
        assert store.delete(wf.id) is True
        assert store.get(wf.id) is None
        assert store.delete(wf.id) is False

    def test_get_by_trial(self, engine: WorkflowEngine, store: WorkflowStore):
        wf1 = engine.create_workflow(_make_request(trial_name="ALPHA"))
        _complete_workflow(engine, wf1.id)
        wf2 = engine.create_workflow(_make_request(trial_name="BETA"))
        _complete_workflow(engine, wf2.id)
        engine.create_workflow(_make_request(trial_name="ALPHA"))
        assert len(store.get_by_trial("ALPHA")) == 2
        assert len(store.get_by_trial("BETA")) == 1

    def test_has_active_workflow(self, engine: WorkflowEngine, store: WorkflowStore):
        assert store.has_active_workflow() is False
        wf = engine.create_workflow(_make_request())
        assert store.has_active_workflow() is True
        _complete_workflow(engine, wf.id)
        assert store.has_active_workflow() is False

    def test_get_active_workflow(self, engine: WorkflowEngine, store: WorkflowStore):
        assert store.get_active_workflow() is None
        wf = engine.create_workflow(_make_request())
        active = store.get_active_workflow()
        assert active is not None
        assert active.id == wf.id

    def test_new_stages(self, store: WorkflowStore):
        stages = store.new_stages()
        assert len(stages) == len(STAGE_ORDER)
        for s in STAGE_ORDER:
            assert stages[s].status == StageStatus.NOT_STARTED


# Engine lifecycle tests (no execution — state management only)

class TestWorkflowEngine:
    def test_create_workflow(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())
        assert wf.status == WorkflowStatus.RUNNING
        assert wf.current_stage == WorkflowStage.PATIENT_SCREENING
        assert len(wf.stages) == 3
        assert wf.trial_name == "TEST-001"
        assert wf.metadata == {"phase": "III"}

    def test_create_blocked_while_active(self, engine: WorkflowEngine):
        """Cannot create a second workflow while one is still active."""
        engine.create_workflow(_make_request(name="First"))
        with pytest.raises(ValueError, match="already in progress"):
            engine.create_workflow(_make_request(name="Second"))

    def test_create_allowed_after_completion(self, engine: WorkflowEngine):
        """Can create a new workflow after the previous one is completed."""
        wf = engine.create_workflow(_make_request(name="First"))
        _complete_workflow(engine, wf.id)
        wf2 = engine.create_workflow(_make_request(name="Second"))
        assert wf2.name == "Second"

    def test_create_allowed_after_deletion(self, engine: WorkflowEngine, store: WorkflowStore):
        """Can create a new workflow after deleting the active one."""
        wf = engine.create_workflow(_make_request(name="First"))
        store.delete(wf.id)
        wf2 = engine.create_workflow(_make_request(name="Second"))
        assert wf2.name == "Second"

    def test_start_workflow_from_paused(self, engine: WorkflowEngine):
        """Start works from PAUSED state (e.g. after pausing)."""
        wf = engine.create_workflow(_make_request())
        wf = engine.pause_workflow(wf.id)
        assert wf.status == WorkflowStatus.PAUSED

        wf = engine.start_workflow(wf.id)
        assert wf.status == WorkflowStatus.PAUSED
        assert wf.current_stage == WorkflowStage.PATIENT_SCREENING
        assert wf.stages[WorkflowStage.PATIENT_SCREENING].status == StageStatus.NOT_STARTED

    def test_start_already_running_raises(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())
        wf.status = WorkflowStatus.RUNNING
        WorkflowStore().save(wf)
        with pytest.raises(ValueError, match="Cannot start"):
            engine.start_workflow(wf.id)

    def test_advance_workflow(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        # Simulate stage 1 completion via update_stage
        engine.update_stage(
            wf.id,
            WorkflowStage.PATIENT_SCREENING,
            StageStatus.COMPLETED,
            output_data={"eligible": 10},
        )

        wf = engine.advance_workflow(wf.id)
        assert wf.current_stage == WorkflowStage.COHORT_FORMATION
        assert wf.status == WorkflowStatus.RUNNING

    def test_advance_not_completed_raises(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        # Stage 1 still NOT_STARTED — cannot advance
        with pytest.raises(ValueError, match="not completed"):
            engine.advance_workflow(wf.id)

    def test_resume_workflow_from_paused(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())
        wf = engine.pause_workflow(wf.id)  # pause first

        wf = engine.resume_workflow(wf.id)
        assert wf.status == WorkflowStatus.RUNNING
        assert wf.current_stage == WorkflowStage.PATIENT_SCREENING

    def test_resume_resets_failed_stage(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        # Simulate stage failure
        engine.update_stage(
            wf.id,
            WorkflowStage.PATIENT_SCREENING,
            StageStatus.FAILED,
            error="connection timeout",
        )

        wf = engine.resume_workflow(wf.id)
        assert wf.status == WorkflowStatus.RUNNING
        stage = wf.stages[WorkflowStage.PATIENT_SCREENING]
        assert stage.status == StageStatus.NOT_STARTED
        assert stage.error is None

    def test_update_stage_completed(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        wf = engine.update_stage(
            wf.id,
            WorkflowStage.PATIENT_SCREENING,
            StageStatus.COMPLETED,
            output_data={"eligible": 5},
        )

        stage = wf.stages[WorkflowStage.PATIENT_SCREENING]
        assert stage.status == StageStatus.COMPLETED
        assert stage.output_data == {"eligible": 5}
        assert stage.completed_at is not None

    def test_update_stage_in_progress(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        wf = engine.update_stage(
            wf.id,
            WorkflowStage.PATIENT_SCREENING,
            StageStatus.IN_PROGRESS,
        )

        assert wf.status == WorkflowStatus.RUNNING
        stage = wf.stages[WorkflowStage.PATIENT_SCREENING]
        assert stage.status == StageStatus.IN_PROGRESS
        assert stage.started_at is not None

    def test_update_stage_failed(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        wf = engine.update_stage(
            wf.id,
            WorkflowStage.PATIENT_SCREENING,
            StageStatus.FAILED,
            error="something broke",
        )

        assert wf.status == WorkflowStatus.FAILED
        stage = wf.stages[WorkflowStage.PATIENT_SCREENING]
        assert stage.status == StageStatus.FAILED
        assert "something broke" in stage.error

    def test_update_stage_reset(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        # Complete then reset
        engine.update_stage(wf.id, WorkflowStage.PATIENT_SCREENING, StageStatus.COMPLETED, output_data={"x": 1})
        wf = engine.update_stage(wf.id, WorkflowStage.PATIENT_SCREENING, StageStatus.NOT_STARTED)

        stage = wf.stages[WorkflowStage.PATIENT_SCREENING]
        assert stage.status == StageStatus.NOT_STARTED
        assert stage.output_data is None
        assert stage.started_at is None

    def test_full_lifecycle(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        # Auto-started at stage 1 in RUNNING state
        assert wf.status == WorkflowStatus.RUNNING
        assert wf.current_stage == WorkflowStage.PATIENT_SCREENING

        # Stage 1: mark in-progress then completed
        engine.update_stage(wf.id, WorkflowStage.PATIENT_SCREENING, StageStatus.IN_PROGRESS)
        engine.update_stage(wf.id, WorkflowStage.PATIENT_SCREENING, StageStatus.COMPLETED, output_data={"eligible": 10})

        # Advance to stage 2
        wf = engine.advance_workflow(wf.id)
        assert wf.current_stage == WorkflowStage.COHORT_FORMATION

        # Stage 2: complete
        engine.update_stage(wf.id, WorkflowStage.COHORT_FORMATION, StageStatus.COMPLETED, output_data={"query": "q", "response": "r"})

        # Advance to stage 3
        wf = engine.advance_workflow(wf.id)
        assert wf.current_stage == WorkflowStage.COHORT_MONITORING

        # Stage 3: complete
        engine.update_stage(wf.id, WorkflowStage.COHORT_MONITORING, StageStatus.COMPLETED, output_data={"status": "ok"})

        # Advance past last stage → completed
        wf = engine.advance_workflow(wf.id)
        assert wf.status == WorkflowStatus.COMPLETED
        assert wf.current_stage is None

    def test_pause_workflow(self, engine: WorkflowEngine):
        wf = engine.create_workflow(_make_request())

        # Already RUNNING from auto-start, can pause directly
        wf = engine.pause_workflow(wf.id)
        assert wf.status == WorkflowStatus.PAUSED


# Route tests (using FastAPI TestClient)

class TestWorkflowRoutes:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from backend.app.routes.workflow.workflow_route import router
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router)
        self.client = TestClient(test_app)

    def test_create_and_get(self):
        resp = self.client.post("/api/workflow", json={
            "name": "My Trial",
            "trial_name": "T-001",
            "description": "desc",
            "metadata": {"sponsor": "ACME"},
        })
        assert resp.status_code == 201
        data = resp.json()
        wf_id = data["workflow"]["id"]

        resp2 = self.client.get(f"/api/workflow/{wf_id}")
        assert resp2.status_code == 200
        assert resp2.json()["workflow"]["name"] == "My Trial"

    def test_create_conflict_returns_409(self):
        resp1 = self.client.post("/api/workflow", json={"name": "A", "trial_name": "T1"})
        assert resp1.status_code == 201
        resp2 = self.client.post("/api/workflow", json={"name": "B", "trial_name": "T2"})
        assert resp2.status_code == 409
        assert "already in progress" in resp2.json()["detail"]

    def test_list_workflows(self):
        self.client.post("/api/workflow", json={"name": "A", "trial_name": "T1"})
        resp = self.client.get("/api/workflow")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_list_filtered_by_trial(self):
        self.client.post("/api/workflow", json={"name": "A", "trial_name": "T1"})
        resp = self.client.get("/api/workflow?trial_name=T1")
        assert resp.json()["total"] == 1

    def test_delete_workflow(self):
        resp = self.client.post("/api/workflow", json={"name": "X", "trial_name": "T"})
        wf_id = resp.json()["workflow"]["id"]
        assert self.client.delete(f"/api/workflow/{wf_id}").status_code == 204
        assert self.client.get(f"/api/workflow/{wf_id}").status_code == 404

    def test_404_for_missing_workflow(self):
        assert self.client.get("/api/workflow/nonexistent").status_code == 404

    def test_start_workflow_route(self):
        """Start from paused state via route."""
        resp = self.client.post("/api/workflow", json={"name": "W", "trial_name": "T"})
        wf_id = resp.json()["workflow"]["id"]

        # Pause first, then start
        self.client.post(f"/api/workflow/{wf_id}/pause")
        start = self.client.post(f"/api/workflow/{wf_id}/start")
        assert start.status_code == 200
        data = start.json()
        assert data["status"] == "paused"
        assert data["current_stage"] == "patient_screening"

    def test_update_stage_and_get(self):
        resp = self.client.post("/api/workflow", json={"name": "W", "trial_name": "T"})
        wf_id = resp.json()["workflow"]["id"]

        # Update stage via PUT
        update = self.client.put(
            f"/api/workflow/{wf_id}/stage/patient_screening",
            json={"status": "completed", "output_data": {"eligible": 3}},
        )
        assert update.status_code == 200

        # Verify via GET
        stage = self.client.get(f"/api/workflow/{wf_id}/stage/patient_screening")
        assert stage.status_code == 200
        assert stage.json()["stage"]["status"] == "completed"
        assert stage.json()["stage"]["output_data"]["eligible"] == 3

    def test_advance_after_stage_update(self):
        resp = self.client.post("/api/workflow", json={"name": "W", "trial_name": "T"})
        wf_id = resp.json()["workflow"]["id"]

        # Complete stage 1
        self.client.put(
            f"/api/workflow/{wf_id}/stage/patient_screening",
            json={"status": "completed", "output_data": {}},
        )

        # Advance
        adv = self.client.post(f"/api/workflow/{wf_id}/advance")
        assert adv.status_code == 200
        assert adv.json()["current_stage"] == "cohort_formation"
