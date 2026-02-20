"""
Tests for the federated monitoring client logic.

Uses synthetic monitoring data so tests run without Flower or Ollama.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.federated.monitoring_client import (
    MonitoringClient,
    ndarrays_to_query,
    query_to_ndarrays,
    ndarrays_to_monitoring_result,
    result_to_ndarrays,
    result_to_monitoring_metrics,
    monitoring_metrics_to_result,
)
from backend.app.schema.monitoring_schema import (
    MonitoringQuery,
    MonitoringQueryType,
    SiteMonitoringResult,
)


# ── Fixtures ──

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.fixture()
def monitoring_site(tmp_path: Path) -> Path:
    """Create a site directory with minimal monitoring data."""
    visits = [
        {"patient_id": "P001", "visit_number": 1, "visit_date": "2026-01-01", "status": "completed", "vitals": {}},
        {"patient_id": "P001", "visit_number": 2, "visit_date": "2026-01-22", "status": "completed", "vitals": {}},
        {"patient_id": "P001", "visit_number": 3, "visit_date": "2026-02-12", "status": "missed", "vitals": None},
        {"patient_id": "P002", "visit_number": 1, "visit_date": "2026-01-01", "status": "completed", "vitals": {}},
        {"patient_id": "P002", "visit_number": 2, "visit_date": "2026-01-22", "status": "completed", "vitals": {}},
        {"patient_id": "P002", "visit_number": 3, "visit_date": "2026-02-12", "status": "completed", "vitals": {}},
        {"patient_id": "P003", "visit_number": 1, "visit_date": "2026-01-01", "status": "completed", "vitals": {}},
        {"patient_id": "P003", "visit_number": 2, "visit_date": "2026-01-22", "status": "missed", "vitals": None, "dropout_reason": "adverse_event"},
        {"patient_id": "P003", "visit_number": 3, "visit_date": "2026-02-12", "status": "missed", "vitals": None},
    ]
    aes = [
        {"patient_id": "P001", "ae_term": "Nausea", "ae_category": "Gastrointestinal", "grade": 1, "severity": "mild", "serious": False, "onset_day": 5, "resolution_day": 10, "outcome": "resolved", "relatedness": "probable", "action_taken": "none"},
        {"patient_id": "P001", "ae_term": "Fatigue", "ae_category": "General", "grade": 2, "severity": "moderate", "serious": False, "onset_day": 10, "resolution_day": 20, "outcome": "resolved", "relatedness": "possible", "action_taken": "none"},
        {"patient_id": "P002", "ae_term": "Neutropenia", "ae_category": "Hematologic", "grade": 3, "severity": "severe", "serious": True, "onset_day": 14, "resolution_day": 28, "outcome": "resolved", "relatedness": "probable", "action_taken": "dose_reduced"},
        {"patient_id": "P003", "ae_term": "Nausea", "ae_category": "Gastrointestinal", "grade": 1, "severity": "mild", "serious": False, "onset_day": 3, "resolution_day": 7, "outcome": "resolved", "relatedness": "unlikely", "action_taken": "none"},
    ]
    responses = [
        {"patient_id": "P001", "assessment_date": "2026-01-22", "assessment_visit": 2, "response_category": "PR", "biomarker_name": "PSA", "biomarker_value": 5.0, "biomarker_change_pct": -25.0},
        {"patient_id": "P002", "assessment_date": "2026-01-22", "assessment_visit": 2, "response_category": "SD", "biomarker_name": "PSA", "biomarker_value": 7.0, "biomarker_change_pct": -5.0},
        {"patient_id": "P003", "assessment_date": "2026-01-22", "assessment_visit": 2, "response_category": "PD", "biomarker_name": "PSA", "biomarker_value": 10.0, "biomarker_change_pct": 20.0},
    ]
    labs = [
        {"patient_id": "P001", "visit_number": 1, "lab_date": "2026-01-01", "lab_name": "PSA", "lab_value": 8.0, "lab_unit": "ng/mL", "reference_low": 0.0, "reference_high": 4.0},
        {"patient_id": "P001", "visit_number": 2, "lab_date": "2026-01-22", "lab_name": "PSA", "lab_value": 6.5, "lab_unit": "ng/mL", "reference_low": 0.0, "reference_high": 4.0},
        {"patient_id": "P002", "visit_number": 1, "lab_date": "2026-01-01", "lab_name": "PSA", "lab_value": 10.0, "lab_unit": "ng/mL", "reference_low": 0.0, "reference_high": 4.0},
        {"patient_id": "P002", "visit_number": 2, "lab_date": "2026-01-22", "lab_name": "PSA", "lab_value": 9.0, "lab_unit": "ng/mL", "reference_low": 0.0, "reference_high": 4.0},
    ]

    _write_json(tmp_path / "visits.json", visits)
    _write_json(tmp_path / "adverse_events.json", aes)
    _write_json(tmp_path / "responses.json", responses)
    _write_json(tmp_path / "lab_results.json", labs)

    return tmp_path


# ── Serialisation tests ──

class TestSerializationHelpers:
    """Test MonitoringQuery / SiteMonitoringResult NumPy round-trips."""

    def test_query_round_trip(self):
        query = MonitoringQuery(
            trial_name="TRIAL-001",
            query_type=MonitoringQueryType.ADVERSE_EVENTS,
            parameters={"grade_threshold": 3},
            natural_language_query="Grade 3+ AEs?",
        )
        arrays = query_to_ndarrays(query)
        restored = ndarrays_to_query(arrays)

        assert restored.trial_name == "TRIAL-001"
        assert restored.query_type == MonitoringQueryType.ADVERSE_EVENTS
        assert restored.parameters == {"grade_threshold": 3}

    def test_result_round_trip(self):
        result = SiteMonitoringResult(
            site_id="site_x",
            query_type=MonitoringQueryType.VISIT_PROGRESS,
            total_patients_monitored=50,
            result_data={"adherence_rate_pct": 87.5},
            data_as_of="2026-02-15T10:00:00",
        )
        arrays = result_to_ndarrays(result)
        restored = ndarrays_to_monitoring_result(arrays)

        assert restored.site_id == "site_x"
        assert restored.total_patients_monitored == 50
        assert restored.result_data["adherence_rate_pct"] == 87.5

    def test_metrics_round_trip(self):
        result = SiteMonitoringResult(
            site_id="site_y",
            query_type=MonitoringQueryType.LAB_TRENDS,
            total_patients_monitored=30,
            data_as_of="2026-02-15T10:00:00",
        )
        metrics = result_to_monitoring_metrics(result)
        restored = monitoring_metrics_to_result(metrics)

        assert restored.site_id == "site_y"
        assert restored.total_patients_monitored == 30


# ── Adverse-event handler tests ──

class TestAdverseEventsHandler:
    """Test MonitoringClient._compute_adverse_events."""

    def test_basic_ae_computation(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_adverse_events(data, {})

        assert result["total_ae_count"] == 4
        assert result["patients_with_any_ae"] == 3  # P001, P002, P003
        assert result["sae_count"] == 1  # Only P002's Neutropenia is serious
        assert result["sae_patients"] == 1
        assert "Nausea" in result["top_adverse_events"]

    def test_grade_threshold_filtering(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_adverse_events(data, {"grade_threshold": 3})

        # Only Grade 3+ AEs: Neutropenia (grade 3)
        assert result["filtered_ae_count"] == 1
        assert result["patients_with_any_ae"] == 1  # P002

    def test_severity_distribution(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_adverse_events(data, {})

        assert result["by_severity"]["mild"] == 2
        assert result["by_severity"]["moderate"] == 1
        assert result["by_severity"]["severe"] == 1

    def test_dose_modifications(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_adverse_events(data, {})

        assert result["dose_modifications"]["dose_reduced"] == 1


# ── Visit-progress handler tests ──

class TestVisitProgressHandler:
    """Test MonitoringClient._compute_visit_progress."""

    def test_basic_visit_progress(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_visit_progress(data, {})

        # 9 total visits: 6 completed, 3 missed
        assert result["total_scheduled_visits"] == 9
        assert result["completed_visits"] == 6
        assert result["missed_visits"] == 3
        assert 0 < result["adherence_rate_pct"] < 100

    def test_perfect_adherence_count(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_visit_progress(data, {})

        # Only P002 completed all 3 visits
        assert result["patients_with_100pct_adherence"] == 1

    def test_empty_visits(self, tmp_path: Path):
        _write_json(tmp_path / "visits.json", [])
        _write_json(tmp_path / "adverse_events.json", [])
        _write_json(tmp_path / "responses.json", [])
        _write_json(tmp_path / "lab_results.json", [])

        client = MonitoringClient("test_site", tmp_path)
        data = client._load_monitoring_data()
        result = client._compute_visit_progress(data, {})

        assert result["adherence_rate_pct"] == 0.0


# ── Response-summary handler tests ──

class TestResponseSummaryHandler:
    """Test MonitoringClient._compute_response_summary."""

    def test_response_distribution(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_response_summary(data, {})

        assert result["assessed_patients"] == 3
        assert result["response_distribution"]["PR"] == 1
        assert result["response_distribution"]["SD"] == 1
        assert result["response_distribution"]["PD"] == 1
        # ORR = (CR+PR) / assessed = 1/3
        assert result["overall_response_rate_pct"] == pytest.approx(33.33, abs=0.1)

    def test_disease_control_rate(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_response_summary(data, {})

        # DCR = (CR+PR+SD) / assessed = 2/3
        assert result["disease_control_rate_pct"] == pytest.approx(66.67, abs=0.1)


# ── Dropout-summary handler tests ──

class TestDropoutSummaryHandler:
    """Test MonitoringClient._compute_dropout_summary."""

    def test_dropout_detection(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_dropout_summary(data, {})

        # P001 missed visit 3, P003 missed visits 2-3 → 2 dropouts
        assert result["dropout_count"] >= 1
        assert result["total_patients"] == 3
        assert result["retention_rate_pct"] > 0

    def test_dropout_reasons_reported(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_dropout_summary(data, {})

        # P003 has dropout_reason="adverse_event"
        assert "by_reason" in result
        if result["dropout_count"] > 0:
            assert isinstance(result["by_reason"], dict)


# ── Lab-trends handler tests ──

class TestLabTrendsHandler:
    """Test MonitoringClient._compute_lab_trends."""

    def test_lab_trend_computation(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_lab_trends(data, {})

        assert "PSA" in result["labs_reported"]
        psa_points = result["lab_trends"]["PSA"]
        assert len(psa_points) == 2  # visits 1 and 2
        assert psa_points[0]["visit"] == 1
        assert psa_points[1]["visit"] == 2

    def test_lab_name_filter(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_lab_trends(data, {"lab_name": "PSA"})

        assert result["labs_reported"] == ["PSA"]

    def test_nonexistent_lab_filter(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_lab_trends(data, {"lab_name": "XYZ"})

        assert result["lab_trends"] == {}


# ── Overall-progress handler tests ──

class TestOverallProgressHandler:
    """Test MonitoringClient._compute_overall_progress."""

    def test_overall_dashboard(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()
        result = client._compute_overall_progress(data, {})

        assert result["enrolled_patients"] == 3
        assert "ae_rate_pct" in result
        assert "visit_adherence_pct" in result
        assert "overall_response_rate_pct" in result
        assert "retention_rate_pct" in result


# ── Data loading tests ──

class TestDataLoading:
    """Test MonitoringClient._load_monitoring_data."""

    def test_loads_all_data_types(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        data = client._load_monitoring_data()

        assert data["patient_count"] == 3
        assert len(data["visits"]) == 9
        assert len(data["adverse_events"]) == 4
        assert len(data["responses"]) == 3
        assert len(data["lab_results"]) == 4

    def test_missing_directory(self, tmp_path: Path):
        nonexistent = tmp_path / "nonexistent"
        client = MonitoringClient("test_site", nonexistent)
        data = client._load_monitoring_data()

        assert data["patient_count"] == 0
        assert len(client._errors) > 0

    def test_partial_data(self, tmp_path: Path):
        """Site with only visits.json."""
        _write_json(tmp_path / "visits.json", [
            {"patient_id": "P001", "visit_number": 1, "status": "completed"},
        ])
        client = MonitoringClient("test_site", tmp_path)
        data = client._load_monitoring_data()

        assert data["patient_count"] == 1
        assert len(data["visits"]) == 1
        assert len(data["adverse_events"]) == 0


# ── Flower evaluate interface test ──

class TestEvaluateInterface:
    """Test the Flower evaluate() method end-to-end locally."""

    def test_evaluate_adverse_events(self, monitoring_site: Path):
        client = MonitoringClient("test_site", monitoring_site)
        query = MonitoringQuery(
            trial_name="TEST",
            query_type=MonitoringQueryType.ADVERSE_EVENTS,
            parameters={"grade_threshold": 1},
        )
        params = query_to_ndarrays(query)

        loss, num_examples, metrics = client.evaluate(params, {})

        assert loss == 0.0
        assert num_examples == 3
        assert metrics["site_id"] == "test_site"
        assert metrics["query_type"] == "adverse_events"

        # Decode the full result
        result = monitoring_metrics_to_result(metrics)
        assert result.total_patients_monitored == 3
        assert result.result_data["total_ae_count"] == 4

    def test_evaluate_empty_site(self, tmp_path: Path):
        client = MonitoringClient("empty_site", tmp_path)
        query = MonitoringQuery(
            trial_name="TEST",
            query_type=MonitoringQueryType.VISIT_PROGRESS,
        )
        params = query_to_ndarrays(query)

        loss, num_examples, metrics = client.evaluate(params, {})

        assert num_examples == 0
        result = monitoring_metrics_to_result(metrics)
        assert result.total_patients_monitored == 0
        assert any("No monitoring data" in e for e in result.errors)
