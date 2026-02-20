"""
Monitoring Data Generator

Generates realistic synthetic monitoring data for the treatment arm.
Each site gets visit records, adverse events, treatment responses,
and lab results for its enrolled patients.

Usage::

    python -m data.monitoring.generate_monitoring_data
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Output directory
_OUTPUT_DIR = Path(__file__).resolve().parent

# Configuration
NUM_PATIENTS_PER_SITE = {"site_a": 10, "site_b": 8}
NUM_VISITS = 8  # visits per patient (over ~6 months)
VISIT_INTERVAL_DAYS = 21  # every 3 weeks

# Seed for reproducibility
random.seed(42)

# AE reference data (realistic clinical trial AEs)
AE_TERMS = [
    ("Nausea", "Gastrointestinal", [1, 1, 1, 2, 2, 3]),
    ("Fatigue", "General", [1, 1, 1, 1, 2, 2, 3]),
    ("Anemia", "Hematologic", [1, 2, 2, 2, 3, 3]),
    ("Neutropenia", "Hematologic", [2, 2, 3, 3, 3, 4]),
    ("Diarrhea", "Gastrointestinal", [1, 1, 1, 2, 2]),
    ("Headache", "Neurological", [1, 1, 1, 2]),
    ("Alopecia", "Dermatologic", [1, 1, 2]),
    ("Arthralgia", "Musculoskeletal", [1, 1, 2, 2]),
    ("Peripheral neuropathy", "Neurological", [1, 2, 2, 3]),
    ("Hepatotoxicity", "Hepatic", [2, 2, 3, 3, 4]),
    ("Rash", "Dermatologic", [1, 1, 1, 2]),
    ("Vomiting", "Gastrointestinal", [1, 1, 2, 2, 3]),
    ("Thrombocytopenia", "Hematologic", [1, 2, 2, 3, 4]),
    ("Elevated ALT", "Hepatic", [1, 1, 2, 2, 3]),
    ("Mucositis", "Gastrointestinal", [1, 2, 2, 3]),
]

SEVERITY_MAP = {
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "life-threatening",
    5: "fatal",
}

RELATEDNESS_OPTIONS = ["definite", "probable", "possible", "unlikely", "unrelated"]
RELATEDNESS_WEIGHTS = [0.05, 0.25, 0.35, 0.25, 0.10]

ACTION_OPTIONS = ["none", "dose_reduced", "dose_interrupted", "discontinued"]
ACTION_WEIGHTS = [0.50, 0.25, 0.15, 0.10]

OUTCOME_OPTIONS = ["resolved", "ongoing", "resolved_with_sequelae"]
OUTCOME_WEIGHTS = [0.65, 0.25, 0.10]

DROPOUT_REASONS = [
    "adverse_event",
    "disease_progression",
    "patient_withdrawal",
    "protocol_violation",
    "lost_to_follow_up",
    "physician_decision",
]
DROPOUT_REASON_WEIGHTS = [0.25, 0.20, 0.20, 0.10, 0.15, 0.10]

# RECIST response categories and their rough probabilities
RESPONSE_CATEGORIES = ["CR", "PR", "SD", "PD"]
RESPONSE_WEIGHTS = [0.10, 0.30, 0.35, 0.25]

# Lab definitions: (name, unit, baseline_mean, baseline_std, treatment_effect_per_visit)
LAB_DEFINITIONS = [
    ("PSA", "ng/mL", 8.0, 4.0, -0.3),         # PSA should decrease with treatment
    ("ALT", "U/L", 25.0, 10.0, 0.8),           # Liver enzymes may rise
    ("AST", "U/L", 22.0, 8.0, 0.5),            # Liver enzymes may rise
    ("Hemoglobin", "g/dL", 13.5, 1.5, -0.15),  # May decrease (anemia)
    ("WBC", "10^3/uL", 7.0, 2.0, -0.2),        # May decrease (neutropenia)
    ("Creatinine", "mg/dL", 1.0, 0.2, 0.02),   # Renal function may decrease
]


def _generate_patient_id(site_id: str, index: int) -> str:
    """Generate a patient ID like 'SA-P001'."""
    prefix = site_id.replace("site_", "S").upper()
    return f"{prefix}-P{index:03d}"


def _random_date(start: datetime, end: datetime) -> datetime:
    """Random date between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def _generate_visits(
    patient_id: str,
    enrollment_date: datetime,
    is_dropout: bool,
    dropout_visit: int,
    dropout_reason: str | None = None,
) -> list[dict]:
    """Generate visit records for a single patient."""
    visits = []
    for v in range(1, NUM_VISITS + 1):
        visit_date = enrollment_date + timedelta(days=(v - 1) * VISIT_INTERVAL_DAYS)

        # If patient dropped out, mark remaining visits as missed
        if is_dropout and v > dropout_visit:
            status = "missed"
        elif random.random() < 0.08:  # 8% chance of missing a visit
            status = "missed"
        else:
            status = "completed"

        visit: dict[str, Any] = {
            "patient_id": patient_id,
            "visit_number": v,
            "visit_date": visit_date.strftime("%Y-%m-%d"),
            "status": status,
            "vitals": {
                "blood_pressure_systolic": round(random.gauss(130, 15)),
                "blood_pressure_diastolic": round(random.gauss(82, 10)),
                "heart_rate": round(random.gauss(74, 10)),
                "weight_kg": round(random.gauss(78, 12), 1),
            } if status == "completed" else None,
        }

        # Tag the first missed visit after dropout with the reason
        if is_dropout and v == dropout_visit + 1 and dropout_reason:
            visit["dropout_reason"] = dropout_reason

        visits.append(visit)

    return visits


def _generate_adverse_events(patient_id: str, enrollment_date: datetime, max_visit: int) -> list[dict]:
    """Generate adverse events for a patient."""
    events = []
    # Each patient has 0-4 AEs
    num_aes = random.choices([0, 1, 2, 3, 4], weights=[0.15, 0.30, 0.25, 0.20, 0.10])[0]

    for _ in range(num_aes):
        ae_term, ae_category, grade_pool = random.choice(AE_TERMS)
        grade = random.choice(grade_pool)
        onset_day = random.randint(1, max_visit * VISIT_INTERVAL_DAYS)
        resolution_day = onset_day + random.randint(3, 30) if random.random() < 0.75 else None
        outcome = random.choices(OUTCOME_OPTIONS, weights=OUTCOME_WEIGHTS)[0] if resolution_day else "ongoing"
        relatedness = random.choices(RELATEDNESS_OPTIONS, weights=RELATEDNESS_WEIGHTS)[0]
        action = random.choices(ACTION_OPTIONS, weights=ACTION_WEIGHTS)[0]

        event = {
            "patient_id": patient_id,
            "ae_term": ae_term,
            "ae_category": ae_category,
            "grade": grade,
            "severity": SEVERITY_MAP.get(grade, "unknown"),
            "serious": grade >= 3,
            "onset_day": onset_day,
            "resolution_day": resolution_day,
            "outcome": outcome,
            "relatedness": relatedness,
            "action_taken": action,
        }
        events.append(event)

    return events


def _generate_responses(patient_id: str, enrollment_date: datetime, max_visit: int) -> list[dict]:
    """Generate treatment response assessments (every 2 visits)."""
    responses = []
    baseline_psa = max(0.5, random.gauss(8.0, 4.0))

    for v in range(2, max_visit + 1, 2):  # Assessment every 2 visits
        assessment_date = enrollment_date + timedelta(days=(v - 1) * VISIT_INTERVAL_DAYS)
        response_cat = random.choices(RESPONSE_CATEGORIES, weights=RESPONSE_WEIGHTS)[0]

        # PSA change based on response category
        psa_change_map = {"CR": -80, "PR": -40, "SD": -5, "PD": 30}
        base_change = psa_change_map[response_cat]
        psa_change_pct = base_change + random.gauss(0, 10)
        current_psa = max(0.1, baseline_psa * (1 + psa_change_pct / 100))

        response = {
            "patient_id": patient_id,
            "assessment_date": assessment_date.strftime("%Y-%m-%d"),
            "assessment_visit": v,
            "response_category": response_cat,
            "biomarker_name": "PSA",
            "biomarker_value": round(current_psa, 2),
            "biomarker_change_pct": round(psa_change_pct, 1),
        }
        responses.append(response)

    return responses


def _generate_lab_results(patient_id: str, enrollment_date: datetime, max_visit: int) -> list[dict]:
    """Generate lab results for each visit."""
    labs = []

    for lab_name, lab_unit, base_mean, base_std, effect_per_visit in LAB_DEFINITIONS:
        baseline = max(0.1, random.gauss(base_mean, base_std))

        for v in range(1, max_visit + 1):
            lab_date = enrollment_date + timedelta(days=(v - 1) * VISIT_INTERVAL_DAYS)
            # Value drifts with treatment effect + noise
            value = baseline + (v - 1) * effect_per_visit + random.gauss(0, base_std * 0.15)
            value = max(0.01, round(value, 2))

            # Reference ranges (simplified)
            ref_ranges = {
                "PSA": (0.0, 4.0),
                "ALT": (7.0, 56.0),
                "AST": (10.0, 40.0),
                "Hemoglobin": (12.0, 17.5),
                "WBC": (4.5, 11.0),
                "Creatinine": (0.7, 1.3),
            }

            ref_low, ref_high = ref_ranges.get(lab_name, (0, 100))

            lab = {
                "patient_id": patient_id,
                "visit_number": v,
                "lab_date": lab_date.strftime("%Y-%m-%d"),
                "lab_name": lab_name,
                "lab_value": value,
                "lab_unit": lab_unit,
                "reference_low": ref_low,
                "reference_high": ref_high,
            }
            labs.append(lab)

    return labs


def generate_site_data(site_id: str, num_patients: int) -> dict[str, list[dict]]:
    """Generate all monitoring data for a single site."""
    all_visits: list[dict] = []
    all_aes: list[dict] = []
    all_responses: list[dict] = []
    all_labs: list[dict] = []

    enrollment_start = datetime(2025, 6, 1)

    for i in range(1, num_patients + 1):
        patient_id = _generate_patient_id(site_id, i)
        enrollment_date = _random_date(
            enrollment_start,
            enrollment_start + timedelta(days=30),
        )

        # Determine if patient drops out (15% chance)
        is_dropout = random.random() < 0.15
        dropout_visit = random.randint(2, NUM_VISITS - 1) if is_dropout else NUM_VISITS
        max_visit = dropout_visit if is_dropout else NUM_VISITS
        dropout_reason: str | None = None
        if is_dropout:
            dropout_reason = random.choices(
                DROPOUT_REASONS, weights=DROPOUT_REASON_WEIGHTS
            )[0]

        all_visits.extend(
            _generate_visits(
                patient_id, enrollment_date, is_dropout,
                dropout_visit, dropout_reason,
            )
        )
        all_aes.extend(
            _generate_adverse_events(patient_id, enrollment_date, max_visit)
        )
        all_responses.extend(
            _generate_responses(patient_id, enrollment_date, max_visit)
        )
        all_labs.extend(
            _generate_lab_results(patient_id, enrollment_date, max_visit)
        )

    return {
        "visits": all_visits,
        "adverse_events": all_aes,
        "responses": all_responses,
        "lab_results": all_labs,
    }


def main() -> None:
    """Generate monitoring data for all sites."""
    for site_id, num_patients in NUM_PATIENTS_PER_SITE.items():
        print(f"Generating monitoring data for {site_id} ({num_patients} patients)...")

        site_dir = _OUTPUT_DIR / site_id
        site_dir.mkdir(parents=True, exist_ok=True)

        data = generate_site_data(site_id, num_patients)

        for data_type, records in data.items():
            filepath = site_dir / f"{data_type}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"  {data_type}: {len(records)} records -> {filepath.name}")

        print()

    print("Done!")


if __name__ == "__main__":
    main()
