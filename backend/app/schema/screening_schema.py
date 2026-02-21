"""
Screening Schema

Pydantic models defining inclusion / exclusion criteria that the
central server broadcasts and the aggregate results that each
federated client reports back.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Criterion building blocks
class CriterionCategory(str, Enum):
    """The domain a single criterion applies to."""
    DEMOGRAPHIC = "demographic"   # age, gender, race
    CONDITION   = "condition"     # ICD codes, diagnosis names
    LAB         = "lab"           # lab test value ranges
    MEDICATION  = "medication"    # current / past medications

# Comparison operators for numeric criteria
class Operator(str, Enum):
    EQ  = "eq"   # ==
    NEQ = "neq"  # !=
    GT  = "gt"   # >
    GTE = "gte"  # >=
    LT  = "lt"   # <
    LTE = "lte"  # <=
    IN  = "in"   # value is a member of a list
    NIN = "nin"  # value is NOT a member of a list

# Inclusion or exclusion rule
class Criterion(BaseModel):
    """A single inclusion or exclusion rule.

    Examples
    --------
    - Age >= 18:
        Criterion(category="demographic", field="age", operator="gte", value=18)
    - Diagnosis contains "breast cancer":
        Criterion(category="condition", field="condition_name",
                  operator="in", value=["breast cancer"])
    - PSA < 10:
        Criterion(category="lab", field="lab_value",
                  lab_name="PSA", operator="lt", value=10)
    """

    category: CriterionCategory
    field: str = Field(
        ..., description="Column name in the EHR data to evaluate."
    )
    operator: Operator
    value: Any = Field(
        ..., description="Reference value (number, string, or list)."
    )
    # Extra context (e.g. which lab test this criterion applies to)
    lab_name: Optional[str] = Field(
        None,
        description="When category is 'lab', the specific lab test name.",
    )
    description: Optional[str] = Field(
        None,
        description="Human-readable explanation of the criterion.",
    )

# Full set of criteria the central server distributes
class ScreeningCriteria(BaseModel):
    """Full set of criteria the central server distributes."""

    trial_name: str = Field(..., description="Name or ID of the clinical trial.")
    inclusion: list[Criterion] = Field(
        default_factory=list, description="Patients MUST satisfy all of these."
    )
    exclusion: list[Criterion] = Field(
        default_factory=list, description="Patients matching ANY of these are excluded."
    )
    natural_language_criteria: Optional[str] = Field(
        None,
        description="Optional free-text description of eligibility criteria (MedGemma will interpret this directly).",
    )


# ── Per-patient screening detail (self-correcting screening) ─────────────

class PatientAuditDetail(BaseModel):
    """Audit trail for a single patient's screening decision."""

    patient_id: str = Field(..., description="The patient identifier.")
    initial_decision: str = Field(
        ..., description="First-pass decision: ELIGIBLE or INELIGIBLE."
    )
    initial_reason: str = Field(
        ..., description="First-pass reasoning."
    )
    final_decision: str = Field(
        ..., description="Final decision after audit and possible reflection."
    )
    final_reason: str = Field(
        ..., description="Final reasoning."
    )
    confidence: str = Field(
        "medium", description="Confidence level: high, medium, or low."
    )
    was_corrected: bool = Field(
        False, description="Whether the auditor caused the decision to change."
    )
    screening_passes: int = Field(
        1, description="Number of screening passes (1=single, 2=re-screened)."
    )
    flagged_for_review: bool = Field(
        False, description="Whether this patient was flagged for human review."
    )
    audit_issues: list[str] = Field(
        default_factory=list,
        description="Issues identified by the auditor.",
    )


# Aggregate counts a single site reports back (no patient-level data)
class SiteScreeningResult(BaseModel):
    """Aggregate counts a single site reports back (no patient-level data)."""

    site_id: str = Field(..., description="Unique identifier for this site.")
    total_patients: int = Field(
        ..., description="Total patients in the site's EHR."
    )
    eligible_patients: int = Field(
        ..., description="Patients satisfying all inclusion AND no exclusion criteria."
    )
    inclusion_pass_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-criterion count of patients passing each inclusion rule.",
    )
    exclusion_hit_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-criterion count of patients hitting each exclusion rule.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during screening.",
    )
    # ── Self-correcting screening fields ─────────────────────────────
    patient_audit_details: list[PatientAuditDetail] = Field(
        default_factory=list,
        description="Per-patient audit trail (no PHI, just decisions).",
    )
    high_confidence_count: int = Field(
        0, description="Patients screened with HIGH confidence."
    )
    medium_confidence_count: int = Field(
        0, description="Patients screened with MEDIUM confidence."
    )
    low_confidence_count: int = Field(
        0, description="Patients screened with LOW confidence."
    )
    corrected_count: int = Field(
        0, description="Patients whose decision was changed by the auditor."
    )
    flagged_for_review_count: int = Field(
        0, description="Patients flagged for human review."
    )


# Combined response returned by the API after all sites report
class FederatedScreeningResponse(BaseModel):
    """Combined response returned by the API after all sites report."""

    trial_name: str
    criteria: ScreeningCriteria
    site_results: list[SiteScreeningResult] = Field(default_factory=list)
    aggregate_total_patients: int = 0
    aggregate_eligible_patients: int = 0
    status: str = Field(
        "completed", description="Overall status: 'completed' | 'partial' | 'error'."
    )
    message: str = ""
    # ── Aggregate audit metrics ──────────────────────────────────────
    aggregate_corrected_count: int = Field(
        0,
        description="Total patients whose decision was corrected by the auditor.",
    )
    aggregate_flagged_for_review: int = Field(
        0,
        description="Total patients flagged for human review.",
    )
    aggregate_high_confidence: int = Field(
        0, description="Total patients with HIGH confidence decisions."
    )
    aggregate_low_confidence: int = Field(
        0, description="Total patients with LOW confidence decisions."
    )
