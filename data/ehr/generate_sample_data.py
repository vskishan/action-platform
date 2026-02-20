"""
Generate Sample EHR Data

Creates small, realistic-looking CSV files in the site_a and site_b
directories so the federated screening simulation can be tested
end-to-end.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# Paths
_DATA_DIR = Path(__file__).resolve().parent
_SITE_A = _DATA_DIR / "site_a"
_SITE_B = _DATA_DIR / "site_b"


# Helpers
def _write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  [OK] {path}  ({len(rows)} rows)")


# Data generators

_GENDERS = ["Male", "Female"]
_RACES = ["White", "Black", "Asian", "Hispanic", "Other"]
_CONDITIONS = [
    ("C50",  "Breast Cancer"),
    ("C61",  "Prostate Cancer"),
    ("E11",  "Type 2 Diabetes"),
    ("I10",  "Hypertension"),
    ("J45",  "Asthma"),
    ("N18",  "Chronic Kidney Disease"),
]
_LABS = [
    ("PSA",         "ng/mL",   0.5,  15.0),
    ("Hemoglobin",  "g/dL",    8.0,  17.0),
    ("Creatinine",  "mg/dL",   0.5,   3.0),
    ("ALT",         "U/L",    10.0,  80.0),
    ("WBC",         "K/uL",    3.0,  15.0),
]
_MEDICATIONS = [
    "Metformin",
    "Lisinopril",
    "Amlodipine",
    "Tamoxifen",
    "Enzalutamide",
    "Aspirin",
    "Atorvastatin",
    "Omeprazole",
]


def _generate_site(site_dir: Path, n_patients: int, seed: int) -> None:
    """Generate all four CSV files for one site."""
    rng = random.Random(seed)
    print(f"\nGenerating data for {site_dir.name} ({n_patients} patients) ...")

    # patients.csv
    patients_rows = []
    for i in range(1, n_patients + 1):
        patients_rows.append([
            f"P{seed:02d}{i:04d}",
            rng.randint(18, 85),
            rng.choice(_GENDERS),
            rng.choice(_RACES),
        ])
    _write_csv(
        site_dir / "patients.csv",
        ["patient_id", "age", "gender", "race"],
        patients_rows,
    )

    # conditions.csv
    conditions_rows = []
    for patient in patients_rows:
        pid = patient[0]
        # Each patient gets 1-3 random conditions
        for code, name in rng.sample(_CONDITIONS, k=rng.randint(1, 3)):
            year  = rng.randint(2018, 2025)
            month = rng.randint(1, 12)
            conditions_rows.append([pid, code, name, f"{year}-{month:02d}-01"])
    _write_csv(
        site_dir / "conditions.csv",
        ["patient_id", "condition_code", "condition_name", "onset_date"],
        conditions_rows,
    )

    # labs.csv
    labs_rows = []
    for patient in patients_rows:
        pid = patient[0]
        for lab_name, unit, lo, hi in rng.sample(_LABS, k=rng.randint(2, 4)):
            value = round(rng.uniform(lo, hi), 2)
            year  = rng.randint(2022, 2025)
            month = rng.randint(1, 12)
            labs_rows.append([pid, lab_name, value, unit, f"{year}-{month:02d}-15"])
    _write_csv(
        site_dir / "labs.csv",
        ["patient_id", "lab_name", "lab_value", "lab_unit", "result_date"],
        labs_rows,
    )

    # medications.csv
    meds_rows = []
    for patient in patients_rows:
        pid = patient[0]
        for med in rng.sample(_MEDICATIONS, k=rng.randint(0, 3)):
            start_year = rng.randint(2020, 2024)
            meds_rows.append([
                pid,
                med,
                f"{start_year}-01-01",
                f"{start_year + rng.randint(1, 3)}-12-31",
            ])
    _write_csv(
        site_dir / "medications.csv",
        ["patient_id", "medication_name", "start_date", "end_date"],
        meds_rows,
    )

if __name__ == "__main__":
    _generate_site(_SITE_A, n_patients=50, seed=42)
    _generate_site(_SITE_B, n_patients=40, seed=99)
    print("\n[DONE] Sample EHR data generated successfully.")
