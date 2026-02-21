"""
Workflow Orchestration Agent — Autonomous Workflow Intelligence

An agentic layer that analyses stage results after each workflow step
completes and autonomously decides whether to proceed, adjust
parameters, or flag for human review.

Architecture
------------
After a workflow stage completes (screening, cohort formation, or
cohort monitoring), the orchestrator:

1. **Analyses Stage Results** — Feeds the stage output into MedGemma
   with a structured analysis prompt.
2. **Generates a Recommendation** — PROCEED / ADJUST / REVIEW / ALERT
   with quality scoring and rationale.
3. **Auto-Advances** — If the recommendation is PROCEED and
   ``auto_advance=True``, the orchestrator automatically advances
   the workflow to the next stage.
4. **Records Focus Areas** — Stores focus areas for the next stage
   in workflow metadata so downstream stages can use them.

This transforms the workflow from a passive state machine into an
intelligent, self-driving pipeline where the AI proactively guides
the trial through its lifecycle.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from backend.app.engine.workflow_engine import WorkflowEngine
from backend.app.engine.workflow_store import WorkflowStore
from backend.app.llm.medgemma_client import MedGemmaClient
from backend.app.llm.prompts import WORKFLOW_STAGE_ANALYSIS_PROMPT
from backend.app.schema.workflow_schema import (
    STAGE_ORDER,
    RecommendationAction,
    Workflow,
    WorkflowRecommendation,
    WorkflowStage,
)

logger = logging.getLogger(__name__)


class WorkflowOrchestrationAgent:
    """AI agent that autonomously analyses and orchestrates workflow stages.

    Usage::

        orchestrator = WorkflowOrchestrationAgent()
        recommendation = orchestrator.analyse_and_recommend(
            workflow_id="abc-123",
            completed_stage=WorkflowStage.PATIENT_SCREENING,
        )
        print(recommendation.recommendation)    # PROCEED / ADJUST / ...
        print(recommendation.quality_score)      # 0.0 – 1.0
        print(recommendation.focus_areas)        # ["Monitor PSA ..."]
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._client = MedGemmaClient.get_instance(model=model_name)
        self._engine = WorkflowEngine()
        self._store = WorkflowStore()

    # Public API

    def analyse_and_recommend(
        self,
        workflow_id: str,
        completed_stage: WorkflowStage,
        auto_advance: bool = False,
    ) -> tuple[WorkflowRecommendation, bool]:
        """Analyse a completed stage and generate a recommendation.

        Parameters
        ----------
        workflow_id : str
            The workflow to analyse.
        completed_stage : WorkflowStage
            The stage that just completed.
        auto_advance : bool
            If ``True`` and the recommendation is PROCEED, the
            orchestrator will automatically advance the workflow.

        Returns
        -------
        tuple[WorkflowRecommendation, bool]
            (recommendation, was_auto_advanced)
        """
        workflow = self._store.get(workflow_id)
        if workflow is None:
            raise KeyError(f"Workflow '{workflow_id}' not found.")

        stage_result = workflow.stages.get(completed_stage)
        if stage_result is None:
            raise ValueError(
                f"Stage '{completed_stage}' not found in workflow."
            )

        # Determine next stage
        next_stage = self._next_stage(completed_stage)

        # Build analysis prompt
        stage_results_json = json.dumps(
            stage_result.output_data or {},
            indent=2,
            default=str,
        )

        prompt = WORKFLOW_STAGE_ANALYSIS_PROMPT.format(
            workflow_name=workflow.name,
            trial_name=workflow.trial_name,
            completed_stage=completed_stage.value,
            next_stage=next_stage.value if next_stage else "None (final stage)",
            stage_results_json=stage_results_json,
        )

        logger.info(
            "Orchestrator analysing stage '%s' for workflow '%s'.",
            completed_stage.value,
            workflow_id,
        )

        # Call MedGemma for analysis
        response = self._client.chat(
            prompt=prompt,
            system=(
                "You are an autonomous clinical-trial workflow coordinator. "
                "Analyse stage results and return ONLY valid JSON."
            ),
            temperature=0.1,
        )

        # Parse the recommendation
        recommendation = self._parse_recommendation(
            response=response,
            workflow_id=workflow_id,
            completed_stage=completed_stage,
            next_stage=next_stage,
        )

        logger.info(
            "Orchestrator recommendation for '%s' stage '%s': %s "
            "(quality=%.2f, anomalies=%d, focus_areas=%d)",
            workflow_id,
            completed_stage.value,
            recommendation.recommendation.value,
            recommendation.quality_score,
            len(recommendation.anomalies),
            len(recommendation.focus_areas),
        )

        # Store focus areas in workflow metadata for downstream stages
        self._store_focus_areas(workflow, recommendation)

        # Auto-advance if recommended
        was_advanced = False
        if (
            auto_advance
            and recommendation.recommendation == RecommendationAction.PROCEED
            and next_stage is not None
        ):
            try:
                self._engine.advance_workflow(workflow_id)
                was_advanced = True
                logger.info(
                    "Orchestrator auto-advanced workflow '%s' to stage '%s'.",
                    workflow_id,
                    next_stage.value,
                )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Orchestrator could not auto-advance workflow '%s': %s",
                    workflow_id,
                    exc,
                )

        return recommendation, was_advanced

    # Private Methods

    def _parse_recommendation(
        self,
        response: str,
        workflow_id: str,
        completed_stage: WorkflowStage,
        next_stage: Optional[WorkflowStage],
    ) -> WorkflowRecommendation:
        """Parse MedGemma's JSON recommendation response."""

        # Try to extract JSON from the response
        parsed = self._extract_json(response)

        if parsed is None:
            # Fallback: conservative recommendation
            logger.warning(
                "Could not parse orchestrator response as JSON. "
                "Falling back to REVIEW recommendation."
            )
            return WorkflowRecommendation(
                workflow_id=workflow_id,
                completed_stage=completed_stage,
                next_stage=next_stage,
                quality_score=0.5,
                recommendation=RecommendationAction.REVIEW,
                rationale=(
                    "The AI orchestrator could not parse its own analysis. "
                    "Human review is recommended before proceeding."
                ),
                stage_summary="Analysis inconclusive.",
            )

        # Map recommendation string to enum
        rec_str = parsed.get("recommendation", "REVIEW").upper()
        try:
            rec_action = RecommendationAction(rec_str.lower())
        except ValueError:
            rec_action = RecommendationAction.REVIEW

        # Validate quality score
        quality = parsed.get("quality_score", 0.5)
        if not isinstance(quality, (int, float)):
            quality = 0.5
        quality = max(0.0, min(1.0, float(quality)))

        return WorkflowRecommendation(
            workflow_id=workflow_id,
            completed_stage=completed_stage,
            next_stage=next_stage,
            quality_score=quality,
            recommendation=rec_action,
            rationale=parsed.get("rationale", ""),
            anomalies=parsed.get("anomalies", []),
            focus_areas=parsed.get("focus_areas", []),
            suggested_adjustments=parsed.get("suggested_adjustments", {}),
            stage_summary=parsed.get("stage_summary", ""),
        )

    def _store_focus_areas(
        self,
        workflow: Workflow,
        recommendation: WorkflowRecommendation,
    ) -> None:
        """Persist focus areas and recommendation in workflow metadata."""
        stage_key = recommendation.completed_stage.value

        workflow.metadata[f"{stage_key}_recommendation"] = {
            "recommendation": recommendation.recommendation.value,
            "quality_score": recommendation.quality_score,
            "anomalies": recommendation.anomalies,
            "focus_areas": recommendation.focus_areas,
            "stage_summary": recommendation.stage_summary,
            "suggested_adjustments": recommendation.suggested_adjustments,
        }

        # Store focus areas for the next stage
        if recommendation.next_stage and recommendation.focus_areas:
            next_key = recommendation.next_stage.value
            workflow.metadata[f"{next_key}_focus_areas"] = (
                recommendation.focus_areas
            )

        self._store.save(workflow)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from LLM output (handles code fences)."""

        # Try: direct JSON parse
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Try: extract from code fence
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try: find first { to last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace : last_brace + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    @staticmethod
    def _next_stage(current: WorkflowStage) -> Optional[WorkflowStage]:
        """Return the stage after *current*, or ``None`` if at the end."""
        idx = STAGE_ORDER.index(current)
        if idx + 1 < len(STAGE_ORDER):
            return STAGE_ORDER[idx + 1]
        return None
