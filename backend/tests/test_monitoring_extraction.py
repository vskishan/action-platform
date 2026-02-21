"""
Tests for the MedGemma clinical-note extraction pipeline.

Tests cover:
- Loading clinical notes from files
- Parsing MedGemma's extraction JSON response
- Full extraction->aggregation pipeline with mocked MedGemma
- Fallback behaviour when no clinical notes exist
- Error handling for malformed extraction responses
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.app.federated.monitoring_client import (
    MonitoringClient,
    ndarrays_to_query,
    query_to_ndarrays,
)
from backend.app.schema.monitoring_schema import (
    MonitoringQuery,
    MonitoringQueryType,
    PatientClinicalNotes,
)


# Helpers

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _make_clinical_notes(patient_id: str, num_visits: int = 3) -> dict:
    """Create a minimal clinical notes file for testing."""
    documents = []
    for i in range(1, num_visits + 1):
        if i == num_visits and num_visits > 2:
            # Make last visit a missed visit
            documents.append({
                "visit_number": i,
                "date": f"2026-0{i}-01",
                "type": "missed_visit_note",
                "text": (
                    f"--- Visit {i} | Date: 2026-0{i}-01 ---\n\n"
                    f"Patient {patient_id} did not present for scheduled visit {i}. "
                    "Phone contact attempted — no answer.\n"
                ),
            })
        else:
            documents.append({
                "visit_number": i,
                "date": f"2026-0{i}-01",
                "type": "progress_note",
                "text": (
                    f"--- Visit {i} | Date: 2026-0{i}-01 ---\n\n"
                    f"Routine visit for patient {patient_id}.\n"
                    f"Vitals: BP 120/80, HR 72, Weight 75.0 kg.\n"
                    f"Patient reports mild nausea since last visit, Grade 1.\n"
                    f"Labs: PSA {8.0 - i * 0.5} ng/mL [ref 0.0-4.0], "
                    f"ALT {25 + i} U/L [ref 7.0-56.0].\n"
                    f"Plan: Continue protocol.\n"
                ),
            })
    return {"patient_id": patient_id, "documents": documents}


def _make_extraction_response(patient_id: str) -> dict:
    """A realistic MedGemma extraction response for testing."""
    return {
        "patient_id": patient_id,
        "visits": [
            {"visit_number": 1, "visit_date": "2026-01-01", "status": "completed",
             "vitals": {"blood_pressure_systolic": 120, "blood_pressure_diastolic": 80,
                        "heart_rate": 72, "weight_kg": 75.0}},
            {"visit_number": 2, "visit_date": "2026-02-01", "status": "completed",
             "vitals": {"blood_pressure_systolic": 118, "blood_pressure_diastolic": 78,
                        "heart_rate": 70, "weight_kg": 74.5}},
            {"visit_number": 3, "visit_date": "2026-03-01", "status": "missed"},
        ],
        "adverse_events": [
            {"ae_term": "Nausea", "ae_category": "Gastrointestinal", "grade": 1,
             "severity": "mild", "serious": False, "onset_day": 5,
             "resolution_day": 12, "outcome": "resolved",
             "relatedness": "probable", "action_taken": "none"},
        ],
        "responses": [
            {"assessment_visit": 2, "assessment_date": "2026-02-01",
             "response_category": "SD", "biomarker_name": "PSA",
             "biomarker_value": 7.0, "biomarker_change_pct": -12.5},
        ],
        "lab_results": [
            {"visit_number": 1, "lab_date": "2026-01-01", "lab_name": "PSA",
             "lab_value": 8.0, "lab_unit": "ng/mL",
             "reference_low": 0.0, "reference_high": 4.0},
            {"visit_number": 1, "lab_date": "2026-01-01", "lab_name": "ALT",
             "lab_value": 26.0, "lab_unit": "U/L",
             "reference_low": 7.0, "reference_high": 56.0},
            {"visit_number": 2, "lab_date": "2026-02-01", "lab_name": "PSA",
             "lab_value": 7.0, "lab_unit": "ng/mL",
             "reference_low": 0.0, "reference_high": 4.0},
            {"visit_number": 2, "lab_date": "2026-02-01", "lab_name": "ALT",
             "lab_value": 27.0, "lab_unit": "U/L",
             "reference_low": 7.0, "reference_high": 56.0},
        ],
    }


# Fixtures

@pytest.fixture()
def notes_site(tmp_path: Path) -> Path:
    """Create a site directory with clinical notes for 3 patients."""
    for pid in ["P001", "P002", "P003"]:
        _write_json(
            tmp_path / "clinical_notes" / f"{pid}.json",
            _make_clinical_notes(pid),
        )
    return tmp_path


@pytest.fixture()
def structured_site(tmp_path: Path) -> Path:
    """Create a site with pre-structured JSON (no clinical notes)."""
    _write_json(tmp_path / "visits.json", [
        {"patient_id": "P001", "visit_number": 1, "status": "completed"},
        {"patient_id": "P001", "visit_number": 2, "status": "completed"},
    ])
    _write_json(tmp_path / "adverse_events.json", [])
    _write_json(tmp_path / "responses.json", [])
    _write_json(tmp_path / "lab_results.json", [])
    return tmp_path


# _load_clinical_notes tests

class TestLoadClinicalNotes:
    """Test loading clinical-note files from disk."""

    def test_loads_all_patient_files(self, notes_site: Path):
        client = MonitoringClient("test_site", notes_site)
        notes = client._load_clinical_notes()

        assert len(notes) == 3
        patient_ids = {n.patient_id for n in notes}
        assert patient_ids == {"P001", "P002", "P003"}

    def test_each_patient_has_documents(self, notes_site: Path):
        client = MonitoringClient("test_site", notes_site)
        notes = client._load_clinical_notes()

        for pn in notes:
            assert len(pn.documents) == 3
            assert pn.documents[0].visit_number == 1

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        client = MonitoringClient("test_site", tmp_path)
        notes = client._load_clinical_notes()

        assert notes == []
        assert any("not found" in e for e in client._errors)

    def test_malformed_json_skipped(self, tmp_path: Path):
        notes_dir = tmp_path / "clinical_notes"
        notes_dir.mkdir(parents=True)
        # Write valid + invalid files
        _write_json(notes_dir / "P001.json", _make_clinical_notes("P001"))
        with open(notes_dir / "P002.json", "w") as f:
            f.write("NOT VALID JSON {{{")

        client = MonitoringClient("test_site", tmp_path)
        notes = client._load_clinical_notes()

        assert len(notes) == 1
        assert notes[0].patient_id == "P001"
        assert any("P002" in e for e in client._errors)

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        (tmp_path / "clinical_notes").mkdir(parents=True)
        client = MonitoringClient("test_site", tmp_path)
        notes = client._load_clinical_notes()

        assert notes == []


# _extract_from_clinical_notes tests

class TestExtractFromClinicalNotes:
    """Test MedGemma extraction with mocked LLM responses."""

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_successful_extraction(self, MockClient, notes_site: Path):
        """MedGemma returns valid JSON → structured data extracted."""
        extraction = _make_extraction_response("P001")
        mock_instance = MagicMock()
        mock_instance.chat.return_value = json.dumps(extraction)
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        notes = client._load_clinical_notes()
        result = client._extract_from_clinical_notes(notes[0])

        assert result["patient_id"] == "P001"
        assert len(result["visits"]) == 3
        assert len(result["adverse_events"]) == 1
        assert len(result["responses"]) == 1
        assert len(result["lab_results"]) == 4

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_extraction_with_markdown_fences(self, MockClient, notes_site: Path):
        """MedGemma wraps JSON in ```json ... ``` → still parsed correctly."""
        extraction = _make_extraction_response("P001")
        wrapped = f"```json\n{json.dumps(extraction)}\n```"
        mock_instance = MagicMock()
        mock_instance.chat.return_value = wrapped
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        notes = client._load_clinical_notes()
        result = client._extract_from_clinical_notes(notes[0])

        assert result["patient_id"] == "P001"
        assert len(result["visits"]) == 3

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_extraction_invalid_json_returns_empty(self, MockClient, notes_site: Path):
        """MedGemma returns garbage → empty dict, error logged."""
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "I cannot process this request"
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        notes = client._load_clinical_notes()
        result = client._extract_from_clinical_notes(notes[0])

        assert result == {}
        assert any("invalid JSON" in e for e in client._errors)

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_extraction_exception_returns_empty(self, MockClient, notes_site: Path):
        """MedGemma client raises → empty dict, error logged."""
        mock_instance = MagicMock()
        mock_instance.chat.side_effect = ConnectionError("Ollama down")
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        notes = client._load_clinical_notes()
        result = client._extract_from_clinical_notes(notes[0])

        assert result == {}
        assert any("extraction failed" in e for e in client._errors)

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_extraction_prompt_sent_correctly(self, MockClient, notes_site: Path):
        """Verify that the correct system prompt and patient text are sent."""
        extraction = _make_extraction_response("P001")
        mock_instance = MagicMock()
        mock_instance.chat.return_value = json.dumps(extraction)
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        notes = client._load_clinical_notes()
        client._extract_from_clinical_notes(notes[0])

        call_args = mock_instance.chat.call_args
        assert "clinical data extraction" in call_args.kwargs["system"].lower()
        # prompt is the first positional arg to chat()
        user_prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "P001" in user_prompt
        assert call_args.kwargs["temperature"] == 0.1


# _load_monitoring_data_with_extraction tests

class TestLoadMonitoringDataWithExtraction:
    """Test the full extraction pipeline (notes→MedGemma→structured data)."""

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_full_pipeline_merges_patients(self, MockClient, notes_site: Path):
        """Extraction from 3 patients merges into flat lists."""
        mock_instance = MagicMock()

        def side_effect(prompt, **kwargs):
            # Determine patient from prompt
            for pid in ["P001", "P002", "P003"]:
                if pid in prompt:
                    return json.dumps(_make_extraction_response(pid))
            return "{}"

        mock_instance.chat.side_effect = side_effect
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        data = client._load_monitoring_data_with_extraction()

        assert data["patient_count"] == 3
        # 3 visits per patient × 3 patients = 9
        assert len(data["visits"]) == 9
        # 1 AE per patient × 3 = 3
        assert len(data["adverse_events"]) == 3
        # 1 response per patient × 3 = 3
        assert len(data["responses"]) == 3
        # 4 labs per patient × 3 = 12
        assert len(data["lab_results"]) == 12

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_patient_ids_stamped_on_records(self, MockClient, notes_site: Path):
        """Each merged record has the correct patient_id."""
        mock_instance = MagicMock()

        def side_effect(prompt, **kwargs):
            for pid in ["P001", "P002", "P003"]:
                if pid in prompt:
                    return json.dumps(_make_extraction_response(pid))
            return "{}"

        mock_instance.chat.side_effect = side_effect
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        data = client._load_monitoring_data_with_extraction()

        visit_pids = {v["patient_id"] for v in data["visits"]}
        assert visit_pids == {"P001", "P002", "P003"}

        ae_pids = {ae["patient_id"] for ae in data["adverse_events"]}
        assert ae_pids == {"P001", "P002", "P003"}

    def test_no_notes_falls_back_to_json(self, structured_site: Path):
        """If clinical_notes/ doesn't exist, fallback to pre-structured JSON."""
        client = MonitoringClient("test_site", structured_site)
        data = client._load_monitoring_data_with_extraction()

        # Should have loaded the structured JSON
        assert data["patient_count"] == 1
        assert len(data["visits"]) == 2

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_partial_extraction_failure(self, MockClient, notes_site: Path):
        """If extraction fails for one patient, others still succeed."""
        call_count = 0
        mock_instance = MagicMock()

        def side_effect(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if "P002" in prompt:
                raise ConnectionError("LLM timeout")
            for pid in ["P001", "P003"]:
                if pid in prompt:
                    return json.dumps(_make_extraction_response(pid))
            return "{}"

        mock_instance.chat.side_effect = side_effect
        MockClient.get_instance.return_value = mock_instance

        client = MonitoringClient("test_site", notes_site)
        data = client._load_monitoring_data_with_extraction()

        # Only 2 patients should succeed
        assert data["patient_count"] == 2
        assert len(data["visits"]) == 6  # 3 per patient × 2
        assert any("P002" in e for e in client._errors)


# Evaluate with use_extraction flag

class TestEvaluateWithExtraction:
    """Test that evaluate() routes correctly based on use_extraction flag."""

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_extraction_flag_triggers_extraction_path(
        self, MockClient, notes_site: Path
    ):
        """use_extraction=True → _load_monitoring_data_with_extraction called."""
        mock_instance = MagicMock()

        def side_effect(prompt, **kwargs):
            for pid in ["P001", "P002", "P003"]:
                if pid in prompt:
                    return json.dumps(_make_extraction_response(pid))
            return "{}"

        mock_instance.chat.side_effect = side_effect
        MockClient.get_instance.return_value = mock_instance

        query = MonitoringQuery(
            trial_name="TEST-TRIAL",
            query_type=MonitoringQueryType.ADVERSE_EVENTS,
            use_extraction=True,
        )
        arrays = query_to_ndarrays(query)

        client = MonitoringClient("test_site", notes_site)
        loss, count, metrics = client.evaluate(arrays, {})

        assert count == 3
        assert metrics["query_type"] == "adverse_events"

    def test_no_extraction_uses_json_path(self, structured_site: Path):
        """use_extraction=False → _load_monitoring_data called (default)."""
        query = MonitoringQuery(
            trial_name="TEST-TRIAL",
            query_type=MonitoringQueryType.VISIT_PROGRESS,
            use_extraction=False,
        )
        arrays = query_to_ndarrays(query)

        client = MonitoringClient("test_site", structured_site)
        loss, count, metrics = client.evaluate(arrays, {})

        assert count == 1
        assert metrics["query_type"] == "visit_progress"

    @patch("backend.app.federated.monitoring_client.MedGemmaClient")
    def test_extraction_ae_handler_produces_correct_result(
        self, MockClient, notes_site: Path
    ):
        """Full flow: extraction → AE handler → valid aggregate result."""
        mock_instance = MagicMock()

        def side_effect(prompt, **kwargs):
            for pid in ["P001", "P002", "P003"]:
                if pid in prompt:
                    return json.dumps(_make_extraction_response(pid))
            return "{}"

        mock_instance.chat.side_effect = side_effect
        MockClient.get_instance.return_value = mock_instance

        query = MonitoringQuery(
            trial_name="TEST-TRIAL",
            query_type=MonitoringQueryType.ADVERSE_EVENTS,
            use_extraction=True,
        )
        arrays = query_to_ndarrays(query)
        client = MonitoringClient("test_site", notes_site)
        loss, count, metrics = client.evaluate(arrays, {})

        # Decode the result
        from backend.app.federated.monitoring_client import monitoring_metrics_to_result
        result = monitoring_metrics_to_result(metrics)

        assert result.total_patients_monitored == 3
        assert result.result_data["total_ae_count"] == 3  # 1 AE × 3 patients


# Query serialisation with use_extraction

class TestExtractionQuerySerialization:
    """Verify use_extraction flag survives Flower transport."""

    def test_use_extraction_true_round_trip(self):
        query = MonitoringQuery(
            trial_name="TRL",
            query_type=MonitoringQueryType.LAB_TRENDS,
            use_extraction=True,
        )
        arrays = query_to_ndarrays(query)
        restored = ndarrays_to_query(arrays)
        assert restored.use_extraction is True

    def test_use_extraction_false_round_trip(self):
        query = MonitoringQuery(
            trial_name="TRL",
            query_type=MonitoringQueryType.LAB_TRENDS,
            use_extraction=False,
        )
        arrays = query_to_ndarrays(query)
        restored = ndarrays_to_query(arrays)
        assert restored.use_extraction is False

    def test_use_extraction_default_is_false(self):
        query = MonitoringQuery(
            trial_name="TRL",
            query_type=MonitoringQueryType.OVERALL_PROGRESS,
        )
        assert query.use_extraction is False
