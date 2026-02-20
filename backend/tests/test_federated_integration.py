"""
Integration test for the Flower-based federated screening pipeline.

Runs the full end-to-end flow with MedGemma:
    Flower gRPC server  -->  2x Flower clients  -->  MedGemma screening  -->  aggregate results

Requires:
- Ollama running with the MedGemma model
- FHIR patient bundles in data/ehr/site_a and data/ehr/site_b
"""

from __future__ import annotations

import logging

import pytest

from backend.app.schema.screening_schema import (
    Criterion,
    CriterionCategory,
    Operator,
    ScreeningCriteria,
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
class TestFederatedIntegration:
    """End-to-end integration tests for the Flower + MedGemma pipeline."""

    @pytest.fixture()
    def sample_criteria(self) -> ScreeningCriteria:
        return ScreeningCriteria(
            trial_name="BREAST-ONCO-2026",
            inclusion=[
                Criterion(
                    category=CriterionCategory.DEMOGRAPHIC,
                    field="age",
                    operator=Operator.GTE,
                    value=18,
                    description="Patient must be 18 years or older",
                ),
                Criterion(
                    category=CriterionCategory.CONDITION,
                    field="condition_name",
                    operator=Operator.IN,
                    value=["Breast Cancer"],
                    description="Patient must have a diagnosis of breast cancer",
                ),
            ],
            exclusion=[
                Criterion(
                    category=CriterionCategory.CONDITION,
                    field="condition_name",
                    operator=Operator.IN,
                    value=["Chronic Kidney Disease"],
                    description="Patient must NOT have chronic kidney disease",
                ),
            ],
        )

    def test_full_screening_round(self, sample_criteria: ScreeningCriteria):
        """Run one full Flower round with MedGemma screening."""
        from backend.app.federated.central_server import CentralServer

        server = CentralServer()
        result = server.run_screening(sample_criteria)

        assert result.trial_name == "BREAST-ONCO-2026"
        assert result.status == "completed"
        assert len(result.site_results) == 2
        assert result.aggregate_total_patients > 0

        # Each site should have reported
        site_ids = {r.site_id for r in result.site_results}
        assert "site_a" in site_ids
        assert "site_b" in site_ids

        # MedGemma should have screened every patient
        for site_result in result.site_results:
            assert site_result.total_patients > 0
            assert site_result.eligible_patients <= site_result.total_patients
