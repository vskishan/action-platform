"""
Flower-based Federated Monitoring Client

Each clinical site runs a MonitoringClient that processes monitoring
queries locally and returns only aggregate metrics.

Lifecycle (one Flower "evaluate" round)
---------------------------------------
1. Server sends a MonitoringQuery encoded as NumPy arrays.
2. Client decodes the query and loads local monitoring data.
3. Client computes LOCAL aggregates based on the query type.
4. Client encodes and returns aggregate metrics.

**No patient-level data ever crosses site boundaries.**
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import flwr as fl
from flwr.common import (
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Status,
    Code,
    ndarrays_to_parameters,
)

from backend.app.schema.monitoring_schema import (
    MonitoringQuery,
    MonitoringQueryType,
    PatientClinicalNotes,
    SiteMonitoringResult,
)
from backend.app.llm.prompts import CLINICAL_NOTE_EXTRACTION_SYSTEM
from backend.app.llm.medgemma_client import MedGemmaClient

logger = logging.getLogger(__name__)


# Serialisation helpers (query / result <-> NumPy)
def query_to_ndarrays(query: MonitoringQuery) -> NDArrays:
    """Encode a MonitoringQuery as a list of NumPy arrays for Flower transport."""
    json_bytes = query.model_dump_json().encode("utf-8")
    return [np.frombuffer(json_bytes, dtype=np.uint8)]


def ndarrays_to_query(ndarrays: NDArrays) -> MonitoringQuery:
    """Decode a MonitoringQuery from the NumPy transport."""
    json_bytes = ndarrays[0].tobytes()
    return MonitoringQuery.model_validate_json(json_bytes)


def result_to_ndarrays(result: SiteMonitoringResult) -> NDArrays:
    """Encode a SiteMonitoringResult as NumPy arrays."""
    json_bytes = result.model_dump_json().encode("utf-8")
    return [np.frombuffer(json_bytes, dtype=np.uint8)]


def ndarrays_to_monitoring_result(ndarrays: NDArrays) -> SiteMonitoringResult:
    """Decode a SiteMonitoringResult from NumPy arrays."""
    json_bytes = ndarrays[0].tobytes()
    return SiteMonitoringResult.model_validate_json(json_bytes)


def result_to_monitoring_metrics(result: SiteMonitoringResult) -> dict[str, Any]:
    """Encode a site result into Flower evaluate metrics."""
    return {
        "site_id": result.site_id,
        "patient_count": int(result.total_patients_monitored),
        "query_type": result.query_type.value,
        "site_result_json": result.model_dump_json(),
    }


def monitoring_metrics_to_result(metrics: dict[str, Any]) -> SiteMonitoringResult:
    """Decode a site result from Flower evaluate metrics."""
    payload = metrics.get("site_result_json", "")
    if not isinstance(payload, str) or not payload:
        raise ValueError("Missing site_result_json in evaluation metrics")
    return SiteMonitoringResult.model_validate_json(payload)


# Flower Monitoring Client
class MonitoringClient(fl.client.NumPyClient):
    """Flower client that computes monitoring aggregates locally.

    Uses the ``evaluate()`` round to:
    1. Receive the MonitoringQuery from the server.
    2. Load local monitoring data (visits, AEs, responses, labs).
    3. Compute LOCAL aggregates based on the query type.
    4. Return only aggregated metrics — **no patient-level data**.
    """

    def __init__(self, site_id: str, monitoring_data_dir: Path) -> None:
        super().__init__()
        self.site_id = site_id
        self.monitoring_data_dir = Path(monitoring_data_dir)
        self._errors: list[str] = []

    # Flower interface

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Return client metadata."""
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="OK"),
            properties={"site_id": self.site_id},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Not used — no model parameters to share."""
        return GetParametersRes(
            status=Status(code=Code.OK, message="No model parameters"),
            parameters=ndarrays_to_parameters([]),
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Execute a monitoring query locally and return aggregated metrics.

        Returns aggregate counts only — no patient data leaves the site.
        """
        logger.info("[%s] Received monitoring query.", self.site_id)

        # Decode the query
        query = ndarrays_to_query(parameters)
        logger.info(
            "[%s] Query type: '%s' for trial '%s'.",
            self.site_id,
            query.query_type.value,
            query.trial_name,
        )

        # Load local monitoring data — extraction or pre-structured
        if query.use_extraction:
            logger.info("[%s] Extraction mode: using MedGemma to parse clinical notes.", self.site_id)
            data = self._load_monitoring_data_with_extraction()
        else:
            data = self._load_monitoring_data()
        patient_count = data.get("patient_count", 0)

        if patient_count == 0:
            result = SiteMonitoringResult(
                site_id=self.site_id,
                query_type=query.query_type,
                total_patients_monitored=0,
                data_as_of=datetime.now().isoformat(),
                errors=self._errors + ["No monitoring data found."],
            )
            return 0.0, 0, result_to_monitoring_metrics(result)

        # Route to the appropriate handler
        handler_map = {
            MonitoringQueryType.ADVERSE_EVENTS:   self._compute_adverse_events,
            MonitoringQueryType.VISIT_PROGRESS:   self._compute_visit_progress,
            MonitoringQueryType.RESPONSE_SUMMARY: self._compute_response_summary,
            MonitoringQueryType.DROPOUT_SUMMARY:  self._compute_dropout_summary,
            MonitoringQueryType.LAB_TRENDS:       self._compute_lab_trends,
            MonitoringQueryType.OVERALL_PROGRESS: self._compute_overall_progress,
        }

        handler = handler_map.get(query.query_type)
        if handler is None:
            result_data = {"error": f"Unknown query type: {query.query_type}"}
        else:
            try:
                result_data = handler(data, query.parameters)
            except Exception as exc:
                msg = f"Failed to compute {query.query_type.value}: {exc}"
                self._errors.append(msg)
                logger.error("[%s] %s", self.site_id, msg)
                result_data = {"error": msg}

        # Build aggregate result
        result = SiteMonitoringResult(
            site_id=self.site_id,
            query_type=query.query_type,
            total_patients_monitored=patient_count,
            result_data=result_data,
            data_as_of=datetime.now().isoformat(),
            errors=self._errors,
        )

        summary = (
            f"[{self.site_id}] Monitoring query '{query.query_type.value}' "
            f"completed for {patient_count} patients."
        )
        logger.info(summary)
        print(summary, flush=True)

        metrics = result_to_monitoring_metrics(result)

        return 0.0, patient_count, metrics

    def fit(self, parameters, config):
        """Not used in monitoring workflow."""
        return [], 0, {}

    # Query handlers

    def _compute_adverse_events(
        self, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute adverse-event aggregate metrics."""
        aes = data.get("adverse_events", [])
        patient_count = data["patient_count"]
        grade_threshold = params.get("grade_threshold", 1)

        if not aes:
            return {
                "total_ae_count": 0,
                "patients_with_any_ae": 0,
                "ae_rate_pct": 0.0,
            }

        # Filter by grade threshold if specified
        filtered_aes = [ae for ae in aes if ae.get("grade", 0) >= grade_threshold]

        patients_with_ae = len(set(ae["patient_id"] for ae in filtered_aes))

        # Grade distribution
        by_grade: dict[str, int] = {}
        for ae in aes:  # Full distribution (not filtered)
            grade = str(ae.get("grade", 0))
            by_grade[grade] = by_grade.get(grade, 0) + 1

        # Severity distribution
        by_severity: dict[str, int] = {}
        for ae in aes:
            sev = ae.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # SAE count
        sae_count = sum(1 for ae in aes if ae.get("serious", False))
        sae_patients = len(set(
            ae["patient_id"] for ae in aes if ae.get("serious", False)
        ))

        # Top AEs by frequency
        ae_freq: dict[str, int] = {}
        for ae in aes:
            term = ae.get("ae_term", "Unknown")
            ae_freq[term] = ae_freq.get(term, 0) + 1
        top_aes = dict(sorted(ae_freq.items(), key=lambda x: -x[1])[:5])

        # By relatedness
        by_relatedness: dict[str, int] = {}
        for ae in aes:
            rel = ae.get("relatedness", "unknown")
            by_relatedness[rel] = by_relatedness.get(rel, 0) + 1

        # Dose modifications
        dose_mods: dict[str, int] = {}
        for ae in aes:
            action = ae.get("action_taken", "none")
            if action != "none":
                dose_mods[action] = dose_mods.get(action, 0) + 1

        # By category
        by_category: dict[str, int] = {}
        for ae in aes:
            cat = ae.get("ae_category", "Unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_ae_count": len(aes),
            "filtered_ae_count": len(filtered_aes),
            "grade_threshold_applied": grade_threshold,
            "patients_with_any_ae": patients_with_ae,
            "ae_rate_pct": round(100 * patients_with_ae / patient_count, 2) if patient_count > 0 else 0.0,
            "by_grade": by_grade,
            "by_severity": by_severity,
            "sae_count": sae_count,
            "sae_patients": sae_patients,
            "sae_rate_pct": round(100 * sae_patients / patient_count, 2) if patient_count > 0 else 0.0,
            "top_adverse_events": top_aes,
            "by_relatedness": by_relatedness,
            "dose_modifications": dose_mods,
            "by_category": by_category,
        }

    def _compute_visit_progress(
        self, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute visit adherence and completion metrics."""
        visits = data.get("visits", [])
        patient_count = data["patient_count"]

        if not visits:
            return {
                "total_scheduled_visits": 0,
                "completed_visits": 0,
                "missed_visits": 0,
                "adherence_rate_pct": 0.0,
            }

        total = len(visits)
        completed = sum(1 for v in visits if v.get("status") == "completed")
        missed = sum(1 for v in visits if v.get("status") == "missed")

        # Visits per patient
        visits_per_patient: dict[str, int] = {}
        for v in visits:
            pid = v.get("patient_id", "")
            if v.get("status") == "completed":
                visits_per_patient[pid] = visits_per_patient.get(pid, 0) + 1

        mean_visits = (
            sum(visits_per_patient.values()) / len(visits_per_patient)
            if visits_per_patient else 0.0
        )

        # Count patients whose completed visits equal their total scheduled visits
        total_visits_per_patient: dict[str, int] = {}
        for v in visits:
            pid = v.get("patient_id", "")
            total_visits_per_patient[pid] = total_visits_per_patient.get(pid, 0) + 1

        patients_with_perfect_adherence = sum(
            1 for pid, completed_count in visits_per_patient.items()
            if completed_count == total_visits_per_patient.get(pid, 0)
        )

        return {
            "total_scheduled_visits": total,
            "completed_visits": completed,
            "missed_visits": missed,
            "adherence_rate_pct": round(100 * completed / total, 2) if total > 0 else 0.0,
            "mean_completed_visits_per_patient": round(mean_visits, 2),
            "patients_with_100pct_adherence": patients_with_perfect_adherence,
        }

    def _compute_response_summary(
        self, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute treatment response summary (RECIST-like categories)."""
        responses = data.get("responses", [])
        patient_count = data["patient_count"]

        if not responses:
            return {"message": "No response assessments available."}

        # Latest response per patient
        latest_by_patient: dict[str, dict] = {}
        for r in responses:
            pid = r.get("patient_id", "")
            visit = r.get("assessment_visit", 0)
            if pid not in latest_by_patient or visit > latest_by_patient[pid].get("assessment_visit", 0):
                latest_by_patient[pid] = r

        # Response category distribution (latest per patient)
        by_category: dict[str, int] = {}
        for r in latest_by_patient.values():
            cat = r.get("response_category", "Unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

        assessed_patients = len(latest_by_patient)
        cr = by_category.get("CR", 0)
        pr = by_category.get("PR", 0)
        sd = by_category.get("SD", 0)
        pd_count = by_category.get("PD", 0)

        # Overall Response Rate = CR + PR
        orr = cr + pr
        # Disease Control Rate = CR + PR + SD
        dcr = cr + pr + sd

        # Biomarker changes
        psa_changes = [
            r.get("biomarker_change_pct", 0)
            for r in latest_by_patient.values()
            if r.get("biomarker_name") == "PSA"
        ]

        return {
            "assessed_patients": assessed_patients,
            "response_distribution": by_category,
            "overall_response_rate_pct": round(100 * orr / assessed_patients, 2) if assessed_patients > 0 else 0.0,
            "disease_control_rate_pct": round(100 * dcr / assessed_patients, 2) if assessed_patients > 0 else 0.0,
            "progressive_disease_rate_pct": round(100 * pd_count / assessed_patients, 2) if assessed_patients > 0 else 0.0,
            "psa_change_stats": {
                "count": len(psa_changes),
                "mean_change_pct": round(sum(psa_changes) / len(psa_changes), 2) if psa_changes else 0.0,
                "min_change_pct": round(min(psa_changes), 2) if psa_changes else 0.0,
                "max_change_pct": round(max(psa_changes), 2) if psa_changes else 0.0,
            },
        }

    def _compute_dropout_summary(
        self, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute dropout / retention metrics."""
        visits = data.get("visits", [])
        patient_count = data["patient_count"]

        if not visits:
            return {"message": "No visit data available."}

        # Identify dropouts: patients whose later visits are all 'missed'
        patient_visits: dict[str, list[dict]] = {}
        for v in visits:
            pid = v.get("patient_id", "")
            patient_visits.setdefault(pid, []).append(v)

        dropouts: list[dict[str, Any]] = []
        active_patients = 0

        for pid, pvs in patient_visits.items():
            pvs_sorted = sorted(pvs, key=lambda x: x.get("visit_number", 0))
            # Find if there's a tail of missed visits
            last_completed = 0
            for v in pvs_sorted:
                if v.get("status") == "completed":
                    last_completed = v.get("visit_number", 0)

            total_visits = len(pvs_sorted)
            if last_completed < total_visits and last_completed > 0:
                # Patient dropped out after visit `last_completed`
                # Look for explicit dropout_reason on the first missed visit
                reason = "unknown"
                for v in pvs_sorted:
                    if v.get("dropout_reason"):
                        reason = v["dropout_reason"]
                        break
                dropouts.append({
                    "dropout_after_visit": last_completed,
                    "total_visits": total_visits,
                    "reason": reason,
                })
            elif last_completed == total_visits:
                active_patients += 1
            elif last_completed == 0:
                # Never completed any visit — immediate dropout
                reason = "unknown"
                for v in pvs_sorted:
                    if v.get("dropout_reason"):
                        reason = v["dropout_reason"]
                        break
                dropouts.append({
                    "dropout_after_visit": 0,
                    "total_visits": total_visits,
                    "reason": reason,
                })

        dropout_count = len(dropouts)
        retention_rate = round(
            100 * (patient_count - dropout_count) / patient_count, 2
        ) if patient_count > 0 else 0.0

        avg_dropout_visit = (
            round(
                sum(d["dropout_after_visit"] for d in dropouts) / dropout_count, 1
            ) if dropout_count > 0 else None
        )

        # Aggregate dropout reasons
        by_reason: dict[str, int] = {}
        for d in dropouts:
            r = d.get("reason", "unknown")
            by_reason[r] = by_reason.get(r, 0) + 1

        return {
            "total_patients": patient_count,
            "active_patients": active_patients,
            "dropout_count": dropout_count,
            "dropout_rate_pct": round(100 * dropout_count / patient_count, 2) if patient_count > 0 else 0.0,
            "retention_rate_pct": retention_rate,
            "avg_dropout_after_visit": avg_dropout_visit,
            "by_reason": by_reason,
        }

    def _compute_lab_trends(
        self, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute aggregated lab value trends over time."""
        labs = data.get("lab_results", [])
        patient_count = data["patient_count"]
        target_lab = params.get("lab_name", None)  # Filter to specific lab if provided

        if not labs:
            return {"message": "No lab data available."}

        # Group by lab_name and visit_number
        lab_by_name_visit: dict[str, dict[int, list[float]]] = {}
        for lab in labs:
            name = lab.get("lab_name", "Unknown")
            if target_lab and name.lower() != target_lab.lower():
                continue
            visit = lab.get("visit_number", 0)
            value = lab.get("lab_value", 0.0)
            lab_by_name_visit.setdefault(name, {}).setdefault(visit, []).append(value)

        # Compute per-timepoint aggregates
        trends: dict[str, list[dict[str, Any]]] = {}
        for lab_name, visit_data in lab_by_name_visit.items():
            trend_points = []
            for visit_num in sorted(visit_data.keys()):
                values = visit_data[visit_num]
                n = len(values)
                mean_val = sum(values) / n
                sorted_vals = sorted(values)
                median_val = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

                trend_points.append({
                    "visit": visit_num,
                    "count": n,
                    "mean": round(mean_val, 2),
                    "median": round(median_val, 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                })
            trends[lab_name] = trend_points

        return {
            "lab_trends": trends,
            "labs_reported": list(trends.keys()),
            "patients_with_labs": patient_count,
        }

    def _compute_overall_progress(
        self, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute a high-level trial progress dashboard."""
        # Combine key metrics from other handlers
        ae_summary = self._compute_adverse_events(data, {})
        visit_summary = self._compute_visit_progress(data, {})
        response_summary = self._compute_response_summary(data, {})
        dropout_summary = self._compute_dropout_summary(data, {})

        return {
            "enrolled_patients": data["patient_count"],
            "active_patients": dropout_summary.get("active_patients", 0),
            "dropout_count": dropout_summary.get("dropout_count", 0),
            "retention_rate_pct": dropout_summary.get("retention_rate_pct", 0.0),
            "visit_adherence_pct": visit_summary.get("adherence_rate_pct", 0.0),
            "ae_rate_pct": ae_summary.get("ae_rate_pct", 0.0),
            "sae_rate_pct": ae_summary.get("sae_rate_pct", 0.0),
            "overall_response_rate_pct": response_summary.get("overall_response_rate_pct", 0.0),
            "disease_control_rate_pct": response_summary.get("disease_control_rate_pct", 0.0),
        }

    # MedGemma extraction pipeline

    def _load_clinical_notes(self) -> list[PatientClinicalNotes]:
        """Load per-patient clinical-note JSON files from ``clinical_notes/``.

        Returns a list of ``PatientClinicalNotes`` objects, one per patient.
        Returns an empty list if the directory does not exist or contains no
        files.
        """
        notes_dir = self.monitoring_data_dir / "clinical_notes"
        if not notes_dir.exists():
            self._errors.append(
                f"Clinical notes directory not found: {notes_dir}"
            )
            logger.warning("[%s] %s", self.site_id, self._errors[-1])
            return []

        patient_notes: list[PatientClinicalNotes] = []
        for filepath in sorted(notes_dir.glob("*.json")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                patient_notes.append(PatientClinicalNotes.model_validate(raw))
            except Exception as exc:
                msg = f"Failed to load clinical notes {filepath.name}: {exc}"
                self._errors.append(msg)
                logger.warning("[%s] %s", self.site_id, msg)

        logger.info(
            "[%s] Loaded clinical notes for %d patients.",
            self.site_id, len(patient_notes),
        )
        return patient_notes

    def _extract_from_clinical_notes(
        self, patient_notes: PatientClinicalNotes
    ) -> dict[str, Any]:
        """Use MedGemma to extract structured monitoring data from a single
        patient's clinical notes.

        Sends all of the patient's visit notes in one prompt and expects
        a JSON object with ``visits``, ``adverse_events``, ``responses``,
        and ``lab_results`` arrays — the same schema the existing aggregate
        handlers consume.

        Returns
        -------
        dict
            Parsed extraction result, or an empty dict on failure.
        """
        # Build a single user prompt from all documents
        note_texts = []
        for doc in patient_notes.documents:
            note_texts.append(doc.text)
        combined = "\n\n".join(note_texts)

        user_prompt = (
            f"Patient ID: {patient_notes.patient_id}\n\n"
            f"=== Clinical Notes ({len(patient_notes.documents)} documents) ===\n\n"
            f"{combined}"
        )

        try:
            client = MedGemmaClient.get_instance()
            raw_response = client.chat(
                prompt=user_prompt,
                system=CLINICAL_NOTE_EXTRACTION_SYSTEM,
                temperature=0.1,  # Low temp for deterministic extraction
            )

            # Strip markdown fences if the model wraps the JSON
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                # Remove opening ```json or ``` and closing ```
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            extracted = json.loads(cleaned)
            logger.info(
                "[%s] MedGemma extracted data for patient %s: "
                "%d visits, %d AEs, %d responses, %d labs.",
                self.site_id,
                patient_notes.patient_id,
                len(extracted.get("visits", [])),
                len(extracted.get("adverse_events", [])),
                len(extracted.get("responses", [])),
                len(extracted.get("lab_results", [])),
            )
            return extracted

        except json.JSONDecodeError as exc:
            msg = (
                f"MedGemma returned invalid JSON for {patient_notes.patient_id}: {exc}"
            )
            self._errors.append(msg)
            logger.warning("[%s] %s", self.site_id, msg)
            return {}
        except Exception as exc:
            msg = (
                f"MedGemma extraction failed for {patient_notes.patient_id}: {exc}"
            )
            self._errors.append(msg)
            logger.warning("[%s] %s", self.site_id, msg)
            return {}

    def _load_monitoring_data_with_extraction(self) -> dict[str, Any]:
        """Load monitoring data by extracting from clinical notes via MedGemma.

        This is the *extraction pipeline* alternative to
        ``_load_monitoring_data()``.  It:

        1. Reads per-patient clinical-note files from ``clinical_notes/``.
        2. Sends each patient's notes to MedGemma for structured extraction.
        3. Merges all per-patient extractions into the same flat-list format
           that the existing aggregate handlers expect (``visits``,
           ``adverse_events``, ``responses``, ``lab_results``).

        Falls back to the pre-structured JSON path if no clinical notes are
        found.
        """
        patient_notes_list = self._load_clinical_notes()
        if not patient_notes_list:
            logger.info(
                "[%s] No clinical notes found — falling back to structured JSON.",
                self.site_id,
            )
            return self._load_monitoring_data()

        # Aggregate all per-patient extractions into flat lists
        all_visits: list[dict] = []
        all_aes: list[dict] = []
        all_responses: list[dict] = []
        all_labs: list[dict] = []
        patient_ids: set[str] = set()

        for pn in patient_notes_list:
            extracted = self._extract_from_clinical_notes(pn)
            if not extracted:
                continue

            pid = extracted.get("patient_id", pn.patient_id)
            patient_ids.add(pid)

            # Merge visits — stamp patient_id on each record
            for v in extracted.get("visits", []):
                v["patient_id"] = pid
                all_visits.append(v)

            # Merge adverse events
            for ae in extracted.get("adverse_events", []):
                ae["patient_id"] = pid
                all_aes.append(ae)

            # Merge responses
            for r in extracted.get("responses", []):
                r["patient_id"] = pid
                all_responses.append(r)

            # Merge lab results
            for lab in extracted.get("lab_results", []):
                lab["patient_id"] = pid
                all_labs.append(lab)

        result = {
            "visits": all_visits,
            "adverse_events": all_aes,
            "responses": all_responses,
            "lab_results": all_labs,
            "patient_count": len(patient_ids),
        }

        logger.info(
            "[%s] Extraction complete: %d patients, %d visits, %d AEs, "
            "%d responses, %d labs.",
            self.site_id,
            result["patient_count"],
            len(all_visits),
            len(all_aes),
            len(all_responses),
            len(all_labs),
        )
        return result

    # Data loading (structured JSON fallback)

    def _load_monitoring_data(self) -> dict[str, Any]:
        """Load all monitoring JSON files from the site directory.

        Returns a dict with keys: visits, adverse_events, responses,
        lab_results, patient_count.
        """
        result: dict[str, Any] = {
            "visits": [],
            "adverse_events": [],
            "responses": [],
            "lab_results": [],
            "patient_count": 0,
        }

        if not self.monitoring_data_dir.exists():
            self._errors.append(f"Directory not found: {self.monitoring_data_dir}")
            return result

        file_keys = ["visits", "adverse_events", "responses", "lab_results"]

        for key in file_keys:
            filepath = self.monitoring_data_dir / f"{key}.json"
            if filepath.exists():
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        result[key] = json.load(f)
                    logger.info(
                        "[%s] Loaded %d %s records.",
                        self.site_id, len(result[key]), key,
                    )
                except Exception as exc:
                    msg = f"Failed to load {filepath.name}: {exc}"
                    self._errors.append(msg)
                    logger.warning("[%s] %s", self.site_id, msg)
            else:
                logger.info("[%s] No %s file found.", self.site_id, key)

        # Count unique patients across all data sources
        patient_ids: set[str] = set()
        for key in file_keys:
            for record in result[key]:
                pid = record.get("patient_id")
                if pid:
                    patient_ids.add(pid)

        result["patient_count"] = len(patient_ids)
        logger.info(
            "[%s] Total unique patients: %d",
            self.site_id, result["patient_count"],
        )

        return result
