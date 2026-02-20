"""
Tests for the federated client screening logic.

Uses mock MedGemma responses so tests run without a live Ollama instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.schema.screening_schema import (
    Criterion,
    CriterionCategory,
    Operator,
    ScreeningCriteria,
    SiteScreeningResult,
)
from backend.app.federated.federated_client import (
    ScreeningClient,
    criteria_to_ndarrays,
    ndarrays_to_criteria,
    result_to_ndarrays,
    ndarrays_to_result,
    _parse_decision,
    _format_criteria_text,
)


# Helper to create a minimal FHIR bundle
def _make_fhir_bundle(
    patient_id: str,
    birth_date: str = "1990-01-01",
    gender: str = "female",
    conditions: list[str] | None = None,
) -> dict:
    entries = [{
        "fullUrl": f"urn:uuid:{patient_id}",
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "gender": gender,
            "birthDate": birth_date,
            "name": [{"family": "Test", "given": ["Patient"]}],
        },
    }]
    for cond in (conditions or []):
        entries.append({
            "resource": {
                "resourceType": "Condition",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {"text": cond},
                "clinicalStatus": {
                    "coding": [{"code": "active"}],
                },
            },
        })
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def _write_bundle(directory: Path, patient_id: str, bundle: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"patient_{patient_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


@pytest.fixture()
def fhir_site(tmp_path: Path) -> Path:
    """Create a site directory with 5 FHIR patient bundles."""
    _write_bundle(tmp_path, "P001", _make_fhir_bundle(
        "P001", "1990-05-15", "female",
        ["Malignant neoplasm of breast"],
    ))
    _write_bundle(tmp_path, "P002", _make_fhir_bundle(
        "P002", "2010-03-20", "male",
        ["Malignant neoplasm of breast"],
    ))
    _write_bundle(tmp_path, "P003", _make_fhir_bundle(
        "P003", "1975-11-03", "female",
        ["Essential hypertension"],
    ))
    _write_bundle(tmp_path, "P004", _make_fhir_bundle(
        "P004", "1960-07-22", "male",
        ["Malignant neoplasm of breast", "Chronic kidney disease"],
    ))
    _write_bundle(tmp_path, "P005", _make_fhir_bundle(
        "P005", "1988-01-10", "female",
        ["Malignant neoplasm of breast"],
    ))
    return tmp_path


class TestDecisionParsing:
    """Test MedGemma response parsing."""

    def test_eligible(self):
        response = "DECISION: ELIGIBLE\nREASON: Patient meets all criteria."
        is_eligible, reason = _parse_decision(response)
        assert is_eligible is True
        assert "meets all" in reason

    def test_ineligible(self):
        response = "DECISION: INELIGIBLE\nREASON: Patient is under 18."
        is_eligible, reason = _parse_decision(response)
        assert is_eligible is False
        assert "under 18" in reason

    def test_ineligible_contains_eligible_word(self):
        """INELIGIBLE should not be parsed as eligible."""
        response = "DECISION: INELIGIBLE\nREASON: Not eligible."
        is_eligible, _ = _parse_decision(response)
        assert is_eligible is False

    def test_unparseable_defaults_to_ineligible(self):
        response = "I'm not sure about this patient."
        is_eligible, reason = _parse_decision(response)
        assert is_eligible is False
        assert reason == ""

    def test_extra_whitespace(self):
        response = "  DECISION:   ELIGIBLE  \n  REASON:  Good match.  "
        is_eligible, reason = _parse_decision(response)
        assert is_eligible is True
        assert "Good match" in reason


class TestCriteriaFormatting:
    """Test criteria -> human-readable text."""

    def test_formats_inclusion_and_exclusion(self):
        criteria = ScreeningCriteria(
            trial_name="TEST",
            inclusion=[
                Criterion(
                    category=CriterionCategory.DEMOGRAPHIC,
                    field="age", operator=Operator.GTE, value=18,
                    description="Patient must be 18 or older",
                ),
            ],
            exclusion=[
                Criterion(
                    category=CriterionCategory.CONDITION,
                    field="condition", operator=Operator.IN,
                    value=["CKD"],
                    description="No chronic kidney disease",
                ),
            ],
        )
        text = _format_criteria_text(criteria)
        assert "18 or older" in text
        assert "chronic kidney disease" in text
        assert "Inclusion" in text
        assert "Exclusion" in text

    def test_formats_natural_language_criteria(self):
        criteria = ScreeningCriteria(
            trial_name="NL-TEST",
            natural_language_criteria="Patient must be a smoker.",
            inclusion=[
                Criterion(
                    category=CriterionCategory.DEMOGRAPHIC,
                    field="age", operator=Operator.GTE, value=18,
                ),
            ],
        )
        text = _format_criteria_text(criteria)
        assert "must be a smoker" in text
        assert "Inclusion" in text
        assert "Natural Language" in text


class TestScreeningClientLogic:
    """Test the ScreeningClient with mocked MedGemma."""

    @patch("backend.app.llm.medgemma_client.MedGemmaClient")
    def test_screens_all_patients(self, MockClient, fhir_site: Path):
        """Each patient gets a MedGemma call."""
        mock_instance = MagicMock()
        mock_instance.chat.return_value = (
            "DECISION: ELIGIBLE\nREASON: Meets criteria."
        )
        MockClient.get_instance.return_value = mock_instance

        criteria = ScreeningCriteria(trial_name="TEST")
        client = ScreeningClient("test_site", fhir_site)
        bundles = client._load_fhir_bundles()
        assert len(bundles) == 5

        # Simulate fit
        from backend.app.federated.federated_client import criteria_to_ndarrays
        params = criteria_to_ndarrays(criteria)
        result_arrays, num_patients, metrics = client.fit(params, {})

        assert num_patients == 5
        assert mock_instance.chat.call_count == 5 

    @patch("backend.app.llm.medgemma_client.MedGemmaClient")
    def test_eligible_count(self, MockClient, fhir_site: Path):
        """Mix of eligible/ineligible responses."""
        mock_instance = MagicMock()
        # First 3 eligible, last 2 ineligible
        mock_instance.chat.side_effect = [
            "DECISION: ELIGIBLE\nREASON: OK.",
            "DECISION: ELIGIBLE\nREASON: OK.",
            "DECISION: ELIGIBLE\nREASON: OK.",
            "DECISION: INELIGIBLE\nREASON: Under 18.",
            "DECISION: INELIGIBLE\nREASON: Has CKD.",
        ]
        MockClient.get_instance.return_value = mock_instance

        criteria = ScreeningCriteria(trial_name="TEST")
        client = ScreeningClient("test_site", fhir_site)
        params = criteria_to_ndarrays(criteria)
        result_arrays, num_patients, metrics = client.fit(params, {})

        result = ndarrays_to_result(result_arrays)
        assert result.total_patients == 5
        assert result.eligible_patients == 3

    @patch("backend.app.llm.medgemma_client.MedGemmaClient")
    def test_llm_error_is_non_fatal(self, MockClient, fhir_site: Path):
        """If MedGemma throws on one patient, the rest still get screened."""
        mock_instance = MagicMock()
        mock_instance.chat.side_effect = [
            "DECISION: ELIGIBLE\nREASON: OK.",
            Exception("Ollama timeout"),
            "DECISION: ELIGIBLE\nREASON: OK.",
            "DECISION: INELIGIBLE\nREASON: No.",
            "DECISION: ELIGIBLE\nREASON: OK.",
        ]
        MockClient.get_instance.return_value = mock_instance

        criteria = ScreeningCriteria(trial_name="TEST")
        client = ScreeningClient("test_site", fhir_site)
        params = criteria_to_ndarrays(criteria)
        result_arrays, _, _ = client.fit(params, {})

        result = ndarrays_to_result(result_arrays)
        assert result.total_patients == 5
        assert result.eligible_patients == 3
        assert any("Failed to screen" in e for e in result.errors)

    def test_empty_site(self, tmp_path: Path):
        """Site with no FHIR bundles."""
        criteria = ScreeningCriteria(trial_name="TEST")
        client = ScreeningClient("empty", tmp_path)
        params = criteria_to_ndarrays(criteria)
        result_arrays, num_patients, _ = client.fit(params, {})

        result = ndarrays_to_result(result_arrays)
        assert result.total_patients == 0
        assert any("No patient" in e for e in result.errors)


class TestSerializationHelpers:
    """Test the NumPy serialization/deserialization used by Flower."""

    def test_criteria_round_trip(self):
        criteria = ScreeningCriteria(
            trial_name="SER-TEST",
            inclusion=[
                Criterion(
                    category=CriterionCategory.DEMOGRAPHIC,
                    field="age", operator=Operator.GTE, value=21,
                ),
            ],
        )
        arrays = criteria_to_ndarrays(criteria)
        restored = ndarrays_to_criteria(arrays)

        assert restored.trial_name == "SER-TEST"
        assert len(restored.inclusion) == 1
        assert restored.inclusion[0].value == 21

    def test_result_round_trip(self):
        result = SiteScreeningResult(
            site_id="site_x",
            total_patients=200,
            eligible_patients=75,
        )
        arrays = result_to_ndarrays(result)
        restored = ndarrays_to_result(arrays)

        assert restored.site_id == "site_x"
        assert restored.eligible_patients == 75
