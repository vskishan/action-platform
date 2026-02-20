"""
Generate Synthetic FHIR Patient Bundles

Creates realistic FHIR R4 Bundle JSON files (one per patient)
in site_a/ and site_b/ directories for the federated screening pipeline.

Each bundle contains:
- Patient resource (demographics, birth date, race)
- Condition resources (SNOMED CT + ICD-10 coded diagnoses)
- Observation resources (LOINC-coded lab results)
- MedicationStatement resources (RxNorm-coded medications)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Paths
_DATA_DIR = Path(__file__).resolve().parent
_SITE_A = _DATA_DIR / "site_a"
_SITE_B = _DATA_DIR / "site_b"

# Clinical data pools with proper coding systems

_CONDITIONS = [
    {"snomed": "254837009", "icd10": "C50",  "display": "Malignant neoplasm of breast"},
    {"snomed": "399068003", "icd10": "C61",  "display": "Malignant neoplasm of prostate"},
    {"snomed": "44054006",  "icd10": "E11",  "display": "Type 2 diabetes mellitus"},
    {"snomed": "38341003",  "icd10": "I10",  "display": "Essential hypertension"},
    {"snomed": "195967001", "icd10": "J45",  "display": "Asthma"},
    {"snomed": "709044004", "icd10": "N18",  "display": "Chronic kidney disease"},
]

_LABS = [
    {"name": "Prostate-Specific Antigen", "loinc": "2857-1",  "unit": "ng/mL",   "lo": 0.5,  "hi": 15.0},
    {"name": "Hemoglobin",                "loinc": "718-7",   "unit": "g/dL",    "lo": 8.0,  "hi": 17.0},
    {"name": "Creatinine",                "loinc": "2160-0",  "unit": "mg/dL",   "lo": 0.5,  "hi": 3.0},
    {"name": "Alanine aminotransferase",  "loinc": "1742-6",  "unit": "U/L",     "lo": 10.0, "hi": 80.0},
    {"name": "White blood cell count",    "loinc": "6690-2",  "unit": "10*3/uL", "lo": 3.0,  "hi": 15.0},
]

_MEDICATIONS = [
    {"name": "Metformin",      "rxnorm": "6809"},
    {"name": "Lisinopril",     "rxnorm": "29046"},
    {"name": "Amlodipine",     "rxnorm": "17767"},
    {"name": "Tamoxifen",      "rxnorm": "10324"},
    {"name": "Enzalutamide",   "rxnorm": "1313988"},
    {"name": "Aspirin",        "rxnorm": "1191"},
    {"name": "Atorvastatin",   "rxnorm": "83367"},
    {"name": "Omeprazole",     "rxnorm": "7646"},
]

_GIVEN_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert",
    "Jennifer", "Michael", "Linda", "David", "Elizabeth",
]
_FAMILY_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Wilson", "Anderson",
]
_GENDERS = ["male", "female"]
_RACES = [
    "White", "Black or African American", "Asian",
    "Hispanic or Latino", "Other",
]


def _random_date(rng: random.Random, year_lo: int, year_hi: int) -> str:
    y = rng.randint(year_lo, year_hi)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _build_patient_bundle(patient_id: str, rng: random.Random) -> dict:
    """Build a FHIR R4 Bundle for a single patient."""
    age = rng.randint(16, 85)
    birth_year = 2026 - age
    birth_date = f"{birth_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    gender = rng.choice(_GENDERS)
    race = rng.choice(_RACES)

    entries: list[dict] = []

    # Patient resource
    entries.append({
        "fullUrl": f"urn:uuid:{patient_id}",
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "name": [{
                "use": "official",
                "family": rng.choice(_FAMILY_NAMES),
                "given": [rng.choice(_GIVEN_NAMES)],
            }],
            "gender": gender,
            "birthDate": birth_date,
            "extension": [{
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                "extension": [{
                    "url": "text",
                    "valueString": race,
                }],
            }],
        },
    })

    # Condition resources (1-3 per patient)
    conditions = rng.sample(_CONDITIONS, k=rng.randint(1, 3))
    for cond in conditions:
        entries.append({
            "resource": {
                "resourceType": "Condition",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": cond["snomed"],
                            "display": cond["display"],
                        },
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": cond["icd10"],
                            "display": cond["display"],
                        },
                    ],
                    "text": cond["display"],
                },
                "clinicalStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active",
                    }],
                },
                "onsetDateTime": _random_date(rng, 2018, 2025),
            },
        })

    # Observation resources (2-4 labs per patient)
    labs = rng.sample(_LABS, k=rng.randint(2, 4))
    for lab in labs:
        entries.append({
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": lab["loinc"],
                        "display": lab["name"],
                    }],
                    "text": lab["name"],
                },
                "valueQuantity": {
                    "value": round(rng.uniform(lab["lo"], lab["hi"]), 2),
                    "unit": lab["unit"],
                    "system": "http://unitsofmeasure.org",
                },
                "effectiveDateTime": _random_date(rng, 2023, 2025),
            },
        })

    # MedicationStatement resources (0-3 per patient)
    meds = rng.sample(_MEDICATIONS, k=rng.randint(0, 3))
    for med in meds:
        entries.append({
            "resource": {
                "resourceType": "MedicationStatement",
                "status": "active",
                "subject": {"reference": f"Patient/{patient_id}"},
                "medicationCodeableConcept": {
                    "coding": [{
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": med["rxnorm"],
                        "display": med["name"],
                    }],
                    "text": med["name"],
                },
                "effectivePeriod": {
                    "start": _random_date(rng, 2020, 2024),
                    "end": _random_date(rng, 2025, 2027),
                },
            },
        })

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }


def _generate_site(site_dir: Path, n_patients: int, seed: int) -> None:
    """Generate FHIR bundles for one site."""
    rng = random.Random(seed)
    site_dir.mkdir(parents=True, exist_ok=True)

    # Clean old files
    for old_file in site_dir.glob("patient_*.json"):
        old_file.unlink()
    for old_file in site_dir.glob("*.csv"):
        old_file.unlink()

    print(f"\nGenerating FHIR bundles for {site_dir.name} ({n_patients} patients) ...")

    for i in range(1, n_patients + 1):
        pid = f"P{seed:02d}{i:04d}"
        bundle = _build_patient_bundle(pid, rng)
        filepath = site_dir / f"patient_{pid}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

    print(f"  [OK] {n_patients} FHIR bundles written to {site_dir}")


if __name__ == "__main__":
    _generate_site(_SITE_A, n_patients=10, seed=42)
    _generate_site(_SITE_B, n_patients=8, seed=99)
    print("\n[DONE] FHIR patient bundles generated successfully.")
