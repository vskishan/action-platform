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
