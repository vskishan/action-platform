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
        """Screen each patient's FHIR record using MedGemma.

        Returns aggregate counts only — no patient data leaves the site.
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

        # Get MedGemma client
        from backend.app.llm.medgemma_client import MedGemmaClient
        medgemma = MedGemmaClient.get_instance()

        # Screen each patient
        eligible_count = 0
        reasons: list[str] = []

        for idx, (patient_id, bundle) in enumerate(bundles.items(), 1):
            progress = f"[{self.site_id}] Screening patient {idx}/{total} ({patient_id})"
            logger.info(progress)
            print(progress, flush=True)

            try:
                prompt = _build_screening_prompt(
                    criteria.trial_name, criteria_text, bundle,
                )
                response = medgemma.chat(
                    prompt=prompt,
                    system=SCREENING_SYSTEM_PROMPT,
                    temperature=0.0,
                )

                is_eligible, reason = _parse_decision(response)

                if is_eligible:
                    eligible_count += 1

                if reason:
                    reasons.append(reason)

                decision_str = "ELIGIBLE" if is_eligible else "INELIGIBLE"
                log_msg = f"[{self.site_id}] {patient_id} -> {decision_str} ({reason})"
                logger.info(log_msg)
                print(f"  -> {decision_str}: {reason}", flush=True)

            except Exception as exc:
                msg = f"Failed to screen {patient_id}: {exc}"
                self._errors.append(msg)
                logger.warning("[%s] %s", self.site_id, msg)
                print(f"  -> ERROR: {exc}", flush=True)

        summary = (
            f"[{self.site_id}] Done - {eligible_count}/{total} eligible "
            f"({eligible_count / total * 100:.0f}%)"
        )
        logger.info(summary)
        print(summary, flush=True)

        # Build aggregate result
        result = SiteScreeningResult(
            site_id=self.site_id,
            total_patients=total,
            eligible_patients=eligible_count,
            errors=self._errors,
        )

        metrics = {
            "site_id": self.site_id,
            "total_patients": total,
            "eligible_patients": eligible_count,
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
