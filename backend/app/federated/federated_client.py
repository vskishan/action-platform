"""
Flower-based Federated Client

Each simulated hospital site runs as a Flower NumPyClient.

Lifecycle (one Flower "fit" round)
----------------------------------
1. Server sends criteria encoded as NumPy arrays (via FitIns).
2. Client decodes the criteria JSON.
3. Client loads FHIR R4 patient bundles from its local directory.
4. For each patient, MedGemma evaluates eligibility against the
   trial criteria using the full FHIR record.
5. Client encodes the aggregate result as NumPy arrays and returns
   them (via FitRes).  **No patient-level data leaves the site.**
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Parameters,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from backend.app.schema.screening_schema import (
    Criterion,
    CriterionCategory,
    Operator,
    PatientAuditDetail,
    ScreeningCriteria,
    SiteScreeningResult,
)

from backend.app.llm.prompts import SCREENING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# Serialization helpers (criteria <-> NumPy for Flower transport)

def criteria_to_ndarrays(criteria: ScreeningCriteria) -> NDArrays:
    """Encode a ScreeningCriteria object as a list of NumPy arrays."""
    json_bytes = criteria.model_dump_json().encode("utf-8")
    return [np.frombuffer(json_bytes, dtype=np.uint8)]


def ndarrays_to_criteria(ndarrays: NDArrays) -> ScreeningCriteria:
    """Decode a ScreeningCriteria from the NumPy transport."""
    json_bytes = ndarrays[0].tobytes()
    return ScreeningCriteria.model_validate_json(json_bytes)


def result_to_ndarrays(result: SiteScreeningResult) -> NDArrays:
    """Encode a SiteScreeningResult as NumPy arrays."""
    json_bytes = result.model_dump_json().encode("utf-8")
    return [np.frombuffer(json_bytes, dtype=np.uint8)]


def ndarrays_to_result(ndarrays: NDArrays) -> SiteScreeningResult:
    """Decode a SiteScreeningResult from NumPy arrays."""
    json_bytes = ndarrays[0].tobytes()
    return SiteScreeningResult.model_validate_json(json_bytes)





def _format_criteria_text(criteria: ScreeningCriteria) -> str:
    """Format screening criteria as human-readable text for MedGemma."""
    lines: list[str] = []

    # If natural language criteria is provided, it takes precedence or is prepended.
    if criteria.natural_language_criteria:
        lines.append("## Study Eligibility Criteria (Natural Language)")
        lines.append(criteria.natural_language_criteria)
        lines.append("")

    if criteria.inclusion:
        lines.append("Inclusion Criteria (patient MUST meet ALL):")
        for i, c in enumerate(criteria.inclusion, 1):
            desc = c.description or f"{c.field} {c.operator.value} {c.value}"
            lines.append(f"  {i}. {desc}")

    if criteria.exclusion:
        if lines:
            lines.append("")
        lines.append("Exclusion Criteria (patient must NOT have ANY):")
        for i, c in enumerate(criteria.exclusion, 1):
            desc = c.description or f"{c.field} {c.operator.value} {c.value}"
            lines.append(f"  {i}. {desc}")

    # When only natural language criteria are provided (no structured lists),
    # add a clear boundary statement to prevent the model from inventing criteria.
    if criteria.natural_language_criteria and not criteria.inclusion and not criteria.exclusion:
        lines.append("")
        lines.append(
            "NOTE: The ONLY inclusion and exclusion criteria are those "
            "described in the natural language text above. Do NOT add any "
            "additional criteria beyond what is explicitly stated above. "
            "Any medical condition, medication, or finding NOT mentioned "
            "above should be IGNORED for eligibility purposes."
        )

    return "\n".join(lines)


def _build_screening_prompt(
    trial_name: str,
    criteria_text: str,
    fhir_bundle: dict,
) -> str:
    """Build the user prompt for a single patient screening call."""
    bundle_json = json.dumps(fhir_bundle, indent=2)
    return (
        f"## Clinical Trial: {trial_name}\n\n"
        f"{criteria_text}\n\n"
        f"## Patient Medical Record (FHIR R4 Bundle):\n"
        f"```json\n{bundle_json}\n```\n\n"
        f"IMPORTANT REMINDER: Evaluate this patient ONLY against the "
        f"criteria listed above. Do NOT exclude the patient for any "
        f"condition, medication, or finding that is NOT explicitly "
        f"mentioned in the exclusion criteria. Do NOT add your own "
        f"medical judgment about what should be excluded."
    )


def _parse_decision(response: str) -> tuple[bool, str]:
    """Parse MedGemma's DECISION/REASON response.

    Returns (is_eligible, reason).
    """
    is_eligible = False
    reason = ""
    criteria_used = ""

    for line in response.strip().split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("CRITERIA_USED:"):
            criteria_used = stripped.split(":", 1)[1].strip()

        if upper.startswith("DECISION:"):
            decision = upper.split(":", 1)[1].strip()
            is_eligible = "ELIGIBLE" in decision and "INELIGIBLE" not in decision

        if upper.startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()

    if criteria_used:
        logger.debug("Criteria model reported using: %s", criteria_used)

    return is_eligible, reason


# Flower client

class ScreeningClient(fl.client.NumPyClient):
    """Flower client that screens patients using MedGemma.

    For each patient in the site's FHIR directory, MedGemma reads the
    full FHIR Bundle and makes a holistic eligibility decision. Only
    aggregate counts are returned to the central server.
    """

    def __init__(self, site_id: str, ehr_data_dir: Path) -> None:
        super().__init__()
        self.site_id = site_id
        self.ehr_data_dir = Path(ehr_data_dir)
        self._errors: list[str] = []

    # Flower interface

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Return client metadata."""
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="OK"),
            properties={"site_id": self.site_id},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Not used for screening (no model weights to share)."""
        return GetParametersRes(
            status=Status(code=Code.OK, message="No model parameters"),
            parameters=ndarrays_to_parameters([]),
        )

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Any],
    ) -> tuple[NDArrays, int, dict[str, Any]]:
        """Screen each patient's FHIR record using MedGemma with self-correction.

        Uses the ScreeningAuditorAgent for a two-pass screen → audit → reflect
        pipeline.  Each patient decision includes confidence scoring and an
        audit trail.  Returns aggregate counts only — no patient data leaves
        the site.
        """
        logger.info("[%s] Received fit request from server.", self.site_id)

        # Decode criteria
        criteria = ndarrays_to_criteria(parameters)
        criteria_text = _format_criteria_text(criteria)
        logger.info(
            "[%s] Trial '%s' — %d inclusion, %d exclusion rules.",
            self.site_id,
            criteria.trial_name,
            len(criteria.inclusion),
            len(criteria.exclusion),
        )

        # Load FHIR bundles
        bundles = self._load_fhir_bundles()
        total = len(bundles)

        if total == 0:
            result = SiteScreeningResult(
                site_id=self.site_id,
                total_patients=0,
                eligible_patients=0,
                errors=self._errors + ["No FHIR patient bundles found."],
            )
            return result_to_ndarrays(result), 0, {"site_id": self.site_id}

        # Initialise the ScreeningAuditorAgent for self-correcting screening
        from backend.app.llm.screening_auditor import ScreeningAuditorAgent
        auditor_agent = ScreeningAuditorAgent()

        # Screen each patient with audit
        eligible_count = 0
        patient_audit_details: list[PatientAuditDetail] = []
        high_conf = 0
        medium_conf = 0
        low_conf = 0
        corrected_count = 0
        flagged_count = 0

        for idx, (patient_id, bundle) in enumerate(bundles.items(), 1):
            progress = f"[{self.site_id}] Screening patient {idx}/{total} ({patient_id}) [with audit]"
            logger.info(progress)
            print(progress, flush=True)

            try:
                # Run the full screen → audit → (reflect) pipeline
                decision = auditor_agent.screen_and_audit(
                    patient_id=patient_id,
                    fhir_bundle=bundle,
                    criteria_text=criteria_text,
                    trial_name=criteria.trial_name,
                )

                if decision.final_decision == "ELIGIBLE":
                    eligible_count += 1

                # Track confidence counts
                if decision.confidence.value == "high":
                    high_conf += 1
                elif decision.confidence.value == "low":
                    low_conf += 1
                else:
                    medium_conf += 1

                if decision.was_corrected:
                    corrected_count += 1
                if decision.flagged_for_review:
                    flagged_count += 1

                # Build audit detail (no PHI — just decisions)
                audit_detail = PatientAuditDetail(
                    patient_id=patient_id,
                    initial_decision=decision.initial_decision,
                    initial_reason=decision.initial_reason,
                    final_decision=decision.final_decision,
                    final_reason=decision.final_reason,
                    confidence=decision.confidence.value,
                    was_corrected=decision.was_corrected,
                    screening_passes=decision.screening_passes,
                    flagged_for_review=decision.flagged_for_review,
                    audit_issues=(
                        decision.audit_result.issues
                        if decision.audit_result else []
                    ),
                )
                patient_audit_details.append(audit_detail)

                decision_str = decision.final_decision
                conf_str = decision.confidence.value.upper()
                corrected_str = " [CORRECTED]" if decision.was_corrected else ""
                flagged_str = " [FLAGGED]" if decision.flagged_for_review else ""
                log_msg = (
                    f"[{self.site_id}] {patient_id} -> {decision_str} "
                    f"(confidence={conf_str}, passes={decision.screening_passes})"
                    f"{corrected_str}{flagged_str}"
                )
                logger.info(log_msg)
                print(
                    f"  -> {decision_str}: {decision.final_reason} "
                    f"[{conf_str}]{corrected_str}{flagged_str}",
                    flush=True,
                )

            except Exception as exc:
                msg = f"Failed to screen {patient_id}: {exc}"
                self._errors.append(msg)
                logger.warning("[%s] %s", self.site_id, msg)
                print(f"  -> ERROR: {exc}", flush=True)

        summary = (
            f"[{self.site_id}] Done - {eligible_count}/{total} eligible "
            f"({eligible_count / total * 100:.0f}%) | "
            f"Confidence: H={high_conf} M={medium_conf} L={low_conf} | "
            f"Corrected: {corrected_count} | Flagged: {flagged_count}"
        )
        logger.info(summary)
        print(summary, flush=True)

        # Build aggregate result with audit metadata
        result = SiteScreeningResult(
            site_id=self.site_id,
            total_patients=total,
            eligible_patients=eligible_count,
            errors=self._errors,
            patient_audit_details=patient_audit_details,
            high_confidence_count=high_conf,
            medium_confidence_count=medium_conf,
            low_confidence_count=low_conf,
            corrected_count=corrected_count,
            flagged_for_review_count=flagged_count,
        )

        metrics = {
            "site_id": self.site_id,
            "total_patients": total,
            "eligible_patients": eligible_count,
            "high_confidence": high_conf,
            "medium_confidence": medium_conf,
            "low_confidence": low_conf,
            "corrected": corrected_count,
            "flagged_for_review": flagged_count,
        }

        return result_to_ndarrays(result), total, metrics

    def evaluate(self, parameters, config):
        """Not used in screening workflow."""
        return 0.0, 0, {}

    # Private helpers

    def _load_fhir_bundles(self) -> dict[str, dict]:
        """Load all patient_*.json FHIR bundles from the site directory."""
        bundles: dict[str, dict] = {}

        if not self.ehr_data_dir.exists():
            self._errors.append(f"Directory not found: {self.ehr_data_dir}")
            return bundles

        json_files = sorted(self.ehr_data_dir.glob("patient_*.json"))

        if not json_files:
            self._errors.append(
                f"No patient_*.json files found in {self.ehr_data_dir}"
            )
            return bundles

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                patient_id = filepath.stem.replace("patient_", "")
                bundles[patient_id] = bundle
            except Exception as exc:
                msg = f"Failed to load {filepath.name}: {exc}"
                self._errors.append(msg)
                logger.warning("[%s] %s", self.site_id, msg)

        logger.info(
            "[%s] Loaded %d FHIR bundles from %s",
            self.site_id, len(bundles), self.ehr_data_dir,
        )
        return bundles
