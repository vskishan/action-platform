"""
Integration test for the Flower-based federated monitoring pipeline.

Runs the full end-to-end flow with MedGemma:
    Flower gRPC server  -->  2x Flower monitoring clients  -->  aggregate results  -->  LLM formatting

Requires:
- Ollama running with the MedGemma model
- Monitoring data in data/monitoring/site_a and data/monitoring/site_b
"""

from __future__ import annotations

import logging

import pytest

from backend.app.schema.monitoring_schema import (
    MonitoringQueryType,
)

# Keep Flower / gRPC logs quiet during tests
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.WARNING)


def _medgemma_available() -> bool:
    """Check if MedGemma is reachable via Ollama."""
    try:
        from backend.app.llm.medgemma_client import MedGemmaClient
        client = MedGemmaClient.get_instance()
        return client.is_available()
    except Exception:
        return False


requires_medgemma = pytest.mark.skipif(
    not _medgemma_available(),
    reason="MedGemma / Ollama not available",
)


@requires_medgemma
class TestMonitoringIntegration:
    """End-to-end integration tests for the federated monitoring pipeline."""

    def test_adverse_event_query(self):
        """Run an AE query across federated sites."""
        from backend.app.federated.monitoring_server import MonitoringOrchestrator

        orchestrator = MonitoringOrchestrator()
        result = orchestrator.query(
            trial_name="ONCO-2026",
            user_query="What is the Grade 3+ adverse event rate across all sites?",
        )

        assert result.trial_name == "ONCO-2026"
        assert result.status in ("completed", "completed_with_warnings")
        assert len(result.site_results) == 2
        assert result.query_type == MonitoringQueryType.ADVERSE_EVENTS

        # Global result should have merged AE data
        assert "total_ae_count" in result.global_result or "ae_rate_pct" in result.global_result

        # Each site should have reported
        site_ids = {r.site_id for r in result.site_results}
        assert "site_a" in site_ids
        assert "site_b" in site_ids

        # LLM should have formatted a response
        assert len(result.response) > 10

    def test_visit_progress_query(self):
        """Run a visit adherence query."""
        from backend.app.federated.monitoring_server import MonitoringOrchestrator

        orchestrator = MonitoringOrchestrator()
        result = orchestrator.query(
            trial_name="ONCO-2026",
            user_query="Show me patient visit adherence across sites.",
        )

        assert result.status in ("completed", "completed_with_warnings")
        assert len(result.site_results) == 2
        assert "completed_visits" in result.global_result or "adherence_rate_pct" in result.global_result

    def test_overall_progress_query(self):
        """Run an overall progress query."""
        from backend.app.federated.monitoring_server import MonitoringOrchestrator

        orchestrator = MonitoringOrchestrator()
        result = orchestrator.query(
            trial_name="ONCO-2026",
            user_query="Give me an overall progress summary of the trial.",
        )

        assert result.status in ("completed", "completed_with_warnings")
        assert len(result.site_results) == 2
        total_patients = sum(r.total_patients_monitored for r in result.site_results)
        assert total_patients > 0

    def test_response_summary_query(self):
        """Run a treatment response query."""
        from backend.app.federated.monitoring_server import MonitoringOrchestrator

        orchestrator = MonitoringOrchestrator()
        result = orchestrator.query(
            trial_name="ONCO-2026",
            user_query="What are the treatment response rates?",
        )

        assert result.status in ("completed", "completed_with_warnings")
        assert len(result.site_results) == 2

    def test_dropout_query(self):
        """Run a dropout/retention query."""
        from backend.app.federated.monitoring_server import MonitoringOrchestrator

        orchestrator = MonitoringOrchestrator()
        result = orchestrator.query(
            trial_name="ONCO-2026",
            user_query="How many patients have dropped out and why?",
        )

        assert result.status in ("completed", "completed_with_warnings")
        assert "dropout_count" in result.global_result or "by_reason" in result.global_result

    def test_lab_trends_query(self):
        """Run a lab trends query."""
        from backend.app.federated.monitoring_server import MonitoringOrchestrator

        orchestrator = MonitoringOrchestrator()
        result = orchestrator.query(
            trial_name="ONCO-2026",
            user_query="Show me PSA lab trends over time.",
        )

        assert result.status in ("completed", "completed_with_warnings")
        assert len(result.site_results) == 2
