"""
Screening Auditor Agent — Self-correcting Patient Screening

Implements a two-pass screen → audit → (optional re-screen) pattern
that dramatically improves screening accuracy.

Architecture
------------
1. **Initial Screen** — Standard MedGemma screening call (existing).
2. **Audit** — A second MedGemma call reviews the decision, reasoning,
   and patient data to assess whether the decision is correct.
3. **Confidence Scoring** — The auditor assigns HIGH / MEDIUM / LOW
   confidence based on reasoning quality and data clarity.
4. **Reflection Loop** — If the auditor DISAGREES with the initial
   decision, a third MedGemma call re-screens the patient with the
   auditor's feedback incorporated.

This creates a multi-agent dynamic (screener + auditor) that catches
reasoning errors (e.g. applying criteria the trial doesn't have,
ignoring relevant patient data) without relying on human review for
every patient.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.app.llm.medgemma_client import MedGemmaClient
from backend.app.llm.prompts import (
    SCREENING_AUDIT_SYSTEM_PROMPT,
    SCREENING_REFLECTION_PROMPT,
    SCREENING_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """Confidence in a screening decision."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AuditVerdict(str, Enum):
    """Whether the auditor agrees with the initial decision."""

    AGREE = "agree"
    DISAGREE = "disagree"


class ScreeningAuditResult(BaseModel):
    """Result from the auditor's review of a screening decision."""

    verdict: AuditVerdict = Field(
        ..., description="Whether the auditor agrees with the initial decision."
    )
    confidence: ConfidenceLevel = Field(
        ..., description="Auditor's confidence in the decision correctness."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Issues the auditor identified with the initial decision.",
    )
    corrected_decision: str | None = Field(
        None, description="Corrected decision if the auditor disagrees."
    )
    corrected_reason: str | None = Field(
        None, description="Corrected reasoning if the auditor disagrees."
    )


class PatientScreeningDecision(BaseModel):
    """Complete screening decision for a single patient, including audit."""

    patient_id: str
    initial_decision: str = Field(
        ..., description="ELIGIBLE or INELIGIBLE from the first pass."
    )
    initial_reason: str = Field(
        ..., description="Reasoning from the first pass."
    )
    audit_result: ScreeningAuditResult | None = Field(
        None, description="Result of the auditor's review."
    )
    final_decision: str = Field(
        ..., description="The final ELIGIBLE/INELIGIBLE after audit+reflection."
    )
    final_reason: str = Field(
        ..., description="Final reasoning."
    )
    confidence: ConfidenceLevel = Field(
        ConfidenceLevel.MEDIUM,
        description="Overall confidence in the final decision.",
    )
    was_corrected: bool = Field(
        False, description="Whether the auditor caused a decision change."
    )
    screening_passes: int = Field(
        1, description="Number of screening passes (1=single, 2=re-screened)."
    )
    flagged_for_review: bool = Field(
        False,
        description="Whether this patient should be flagged for human review.",
    )


# ── Screening Auditor Agent ──────────────────────────────────────────────

class ScreeningAuditorAgent:
    """Multi-agent screening with auditor-driven self-correction.

    Usage::

        auditor = ScreeningAuditorAgent()
        decision = auditor.screen_and_audit(
            patient_id="P420001",
            fhir_bundle=bundle,
            criteria_text="Inclusion: Age >= 18 ...",
            trial_name="PROSTATE-CANCER",
        )
        print(decision.final_decision, decision.confidence)
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._client = MedGemmaClient.get_instance(model=model_name)

    # ── Public API ────────────────────────────────────────────────────

    def screen_and_audit(
        self,
        patient_id: str,
        fhir_bundle: dict[str, Any],
        criteria_text: str,
        trial_name: str,
    ) -> PatientScreeningDecision:
        """Run the full screen → audit → (reflect) pipeline.

        Parameters
        ----------
        patient_id : str
            Identifier for the patient being screened.
        fhir_bundle : dict
            Full FHIR R4 Bundle for the patient.
        criteria_text : str
            Human-readable criteria text.
        trial_name : str
            Name of the clinical trial.

        Returns
        -------
        PatientScreeningDecision
            Complete decision with confidence score and audit trail.
        """

        # ── Pass 1: Initial Screening ────────────────────────────────
        initial_decision, initial_reason = self._screen_patient(
            fhir_bundle=fhir_bundle,
            criteria_text=criteria_text,
            trial_name=trial_name,
        )

        logger.info(
            "[Auditor] %s — Initial: %s (%s)",
            patient_id, initial_decision, initial_reason,
        )

        # ── Pass 2: Audit ────────────────────────────────────────────
        audit_result = self._audit_decision(
            patient_id=patient_id,
            fhir_bundle=fhir_bundle,
            criteria_text=criteria_text,
            initial_decision=initial_decision,
            initial_reason=initial_reason,
        )

        logger.info(
            "[Auditor] %s — Audit: %s (confidence=%s, issues=%s)",
            patient_id,
            audit_result.verdict.value,
            audit_result.confidence.value,
            audit_result.issues,
        )

        # ── Pass 3: Reflection (only if auditor disagrees) ───────────
        final_decision = initial_decision
        final_reason = initial_reason
        was_corrected = False
        screening_passes = 1

        if audit_result.verdict == AuditVerdict.DISAGREE:
            logger.info(
                "[Auditor] %s — Auditor disagrees. Running reflection pass.",
                patient_id,
            )
            final_decision, final_reason = self._reflect_and_rescreen(
                fhir_bundle=fhir_bundle,
                criteria_text=criteria_text,
                trial_name=trial_name,
                issues="; ".join(audit_result.issues) if audit_result.issues else "Auditor disagreed",
                corrected_decision=audit_result.corrected_decision or "UNKNOWN",
                corrected_reason=audit_result.corrected_reason or "N/A",
            )
            was_corrected = (final_decision != initial_decision)
            screening_passes = 2

            logger.info(
                "[Auditor] %s — Reflection result: %s (corrected=%s)",
                patient_id, final_decision, was_corrected,
            )

        # ── Determine confidence & flag status ───────────────────────
        confidence = audit_result.confidence

        # Flag for human review if: low confidence, or the decision
        # was corrected, or there were issues even with AGREE
        flagged = (
            confidence == ConfidenceLevel.LOW
            or (was_corrected and confidence != ConfidenceLevel.HIGH)
            or (audit_result.verdict == AuditVerdict.AGREE
                and len(audit_result.issues) > 0
                and confidence != ConfidenceLevel.HIGH)
        )

        return PatientScreeningDecision(
            patient_id=patient_id,
            initial_decision=initial_decision,
            initial_reason=initial_reason,
            audit_result=audit_result,
            final_decision=final_decision,
            final_reason=final_reason,
            confidence=confidence,
            was_corrected=was_corrected,
            screening_passes=screening_passes,
            flagged_for_review=flagged,
        )

    # ── Private Methods ───────────────────────────────────────────────

    def _screen_patient(
        self,
        fhir_bundle: dict[str, Any],
        criteria_text: str,
        trial_name: str,
    ) -> tuple[str, str]:
        """Run a single screening pass using MedGemma.

        Returns (decision, reason).
        """
        bundle_json = json.dumps(fhir_bundle, indent=2)
        prompt = (
            f"## Clinical Trial: {trial_name}\n\n"
            f"{criteria_text}\n\n"
            f"## Patient Medical Record (FHIR R4 Bundle):\n"
            f"```json\n{bundle_json}\n```\n\n"
            f"IMPORTANT REMINDER: Evaluate this patient ONLY against the "
            f"criteria listed above. Do NOT exclude the patient for any "
            f"condition, medication, or finding that is NOT explicitly "
            f"mentioned in the exclusion criteria."
        )

        response = self._client.chat(
            prompt=prompt,
            system=SCREENING_SYSTEM_PROMPT,
            temperature=0.0,
        )

        return self._parse_decision(response)

    def _audit_decision(
        self,
        patient_id: str,
        fhir_bundle: dict[str, Any],
        criteria_text: str,
        initial_decision: str,
        initial_reason: str,
    ) -> ScreeningAuditResult:
        """Run the auditor agent to review a screening decision."""

        # Build a compact patient summary (avoid sending full FHIR twice)
        patient_summary = self._summarise_fhir(fhir_bundle)

        audit_prompt = (
            f"## Trial Eligibility Criteria\n"
            f"{criteria_text}\n\n"
            f"## Patient Summary\n"
            f"Patient ID: {patient_id}\n"
            f"{patient_summary}\n\n"
            f"## Initial Screening Decision\n"
            f"DECISION: {initial_decision}\n"
            f"REASON: {initial_reason}\n\n"
            f"Review this decision against the criteria and patient data."
        )

        response = self._client.chat(
            prompt=audit_prompt,
            system=SCREENING_AUDIT_SYSTEM_PROMPT,
            temperature=0.0,
        )

        return self._parse_audit_response(response)

    def _reflect_and_rescreen(
        self,
        fhir_bundle: dict[str, Any],
        criteria_text: str,
        trial_name: str,
        issues: str,
        corrected_decision: str,
        corrected_reason: str,
    ) -> tuple[str, str]:
        """Re-screen a patient incorporating audit feedback.

        Returns (decision, reason).
        """
        bundle_json = json.dumps(fhir_bundle, indent=2)
        prompt = SCREENING_REFLECTION_PROMPT.format(
            issues=issues,
            corrected_decision=corrected_decision,
            corrected_reason=corrected_reason,
            criteria_text=criteria_text,
            fhir_bundle=bundle_json,
        )

        response = self._client.chat(
            prompt=prompt,
            system=SCREENING_SYSTEM_PROMPT,
            temperature=0.0,
        )

        return self._parse_decision(response)

    # ── Parsing Helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_decision(response: str) -> tuple[str, str]:
        """Parse a DECISION/REASON response from MedGemma.

        Returns (decision_str, reason_str).
        """
        decision = "INELIGIBLE"
        reason = ""

        for line in response.strip().split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("DECISION:"):
                dec = upper.split(":", 1)[1].strip()
                if "ELIGIBLE" in dec and "INELIGIBLE" not in dec:
                    decision = "ELIGIBLE"
                else:
                    decision = "INELIGIBLE"

            if upper.startswith("REASON:"):
                reason = stripped.split(":", 1)[1].strip()

        return decision, reason

    @staticmethod
    def _parse_audit_response(response: str) -> ScreeningAuditResult:
        """Parse the auditor's structured response."""
        verdict = AuditVerdict.AGREE
        confidence = ConfidenceLevel.MEDIUM
        issues: list[str] = []
        corrected_decision: str | None = None
        corrected_reason: str | None = None

        for line in response.strip().split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("AUDIT_DECISION:"):
                val = upper.split(":", 1)[1].strip()
                if "DISAGREE" in val:
                    verdict = AuditVerdict.DISAGREE
                else:
                    verdict = AuditVerdict.AGREE

            elif upper.startswith("CONFIDENCE:"):
                val = upper.split(":", 1)[1].strip()
                if "HIGH" in val:
                    confidence = ConfidenceLevel.HIGH
                elif "LOW" in val:
                    confidence = ConfidenceLevel.LOW
                else:
                    confidence = ConfidenceLevel.MEDIUM

            elif upper.startswith("ISSUES:"):
                val = stripped.split(":", 1)[1].strip()
                if val.lower() != "none" and val:
                    issues = [i.strip() for i in val.split(",") if i.strip()]

            elif upper.startswith("CORRECTED_DECISION:"):
                val = stripped.split(":", 1)[1].strip()
                if val.upper() not in ("UNCHANGED", "N/A", ""):
                    corrected_decision = val.upper()

            elif upper.startswith("CORRECTED_REASON:"):
                val = stripped.split(":", 1)[1].strip()
                if val.upper() not in ("N/A", ""):
                    corrected_reason = val

        return ScreeningAuditResult(
            verdict=verdict,
            confidence=confidence,
            issues=issues,
            corrected_decision=corrected_decision,
            corrected_reason=corrected_reason,
        )

    @staticmethod
    def _summarise_fhir(bundle: dict[str, Any]) -> str:
        """Create a compact text summary of a FHIR bundle for the auditor.

        This avoids sending the full JSON twice, reducing token usage
        while providing enough context for the audit.
        """
        lines: list[str] = []
        entries = bundle.get("entry", [])

        for entry in entries:
            resource = entry.get("resource", {})
            res_type = resource.get("resourceType", "Unknown")

            if res_type == "Patient":
                gender = resource.get("gender", "unknown")
                birth = resource.get("birthDate", "unknown")
                lines.append(f"- Patient: gender={gender}, birthDate={birth}")

            elif res_type == "Condition":
                code = resource.get("code", {})
                text = code.get("text", "")
                if not text:
                    codings = code.get("coding", [])
                    text = codings[0].get("display", "unknown") if codings else "unknown"
                status = resource.get("clinicalStatus", {})
                status_text = ""
                if isinstance(status, dict):
                    codings = status.get("coding", [])
                    status_text = codings[0].get("code", "") if codings else ""
                lines.append(f"- Condition: {text} (status={status_text})")

            elif res_type == "MedicationStatement":
                med_ref = resource.get("medicationCodeableConcept", {})
                med_text = med_ref.get("text", "")
                if not med_text:
                    codings = med_ref.get("coding", [])
                    med_text = codings[0].get("display", "unknown") if codings else "unknown"
                med_status = resource.get("status", "unknown")
                lines.append(f"- Medication: {med_text} (status={med_status})")

            elif res_type == "Observation":
                obs_code = resource.get("code", {})
                obs_text = obs_code.get("text", "")
                if not obs_text:
                    codings = obs_code.get("coding", [])
                    obs_text = codings[0].get("display", "unknown") if codings else "unknown"
                value = resource.get("valueQuantity", {})
                val_str = ""
                if value:
                    val_str = f"{value.get('value', '?')} {value.get('unit', '')}"
                else:
                    val_str = str(resource.get("valueString", ""))
                lines.append(f"- Observation: {obs_text} = {val_str}")

            elif res_type == "Procedure":
                proc_code = resource.get("code", {})
                proc_text = proc_code.get("text", "")
                if not proc_text:
                    codings = proc_code.get("coding", [])
                    proc_text = codings[0].get("display", "unknown") if codings else "unknown"
                lines.append(f"- Procedure: {proc_text}")

        if not lines:
            lines.append("- No structured resources found in bundle")

        return "\n".join(lines)
