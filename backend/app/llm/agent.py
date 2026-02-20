"""
Agent (Orchestrator)

The central orchestrator that:
1. Receives a user query (natural language).
2. Uses the IntentClassifier (MedGemma) to determine engine + intent.
3. Routes to the correct engine and executes the query.
4. Uses MedGemma to format the raw result into a human-readable response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.app.engine.analytics_engine import AnalyticsEngine
from backend.app.engine.survival_engine import SurvivalAnalysisEngine
from backend.app.llm.intent_classifier import IntentClassifier
from backend.app.llm.medgemma_client import MedGemmaClient
from backend.app.llm.prompts import RESPONSE_FORMATTING_SYSTEM

logger = logging.getLogger(__name__)


class Agent:
    """Orchestrator that connects the LLM classifier with domain engines.

    Usage::

        agent = Agent()
        response = agent.handle("What is the average survival rate of patients with the age of 60 and above?")
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._client = MedGemmaClient.get_instance(model=model_name)

        # Initialise the intent classifier.
        self._classifier = IntentClassifier(model_name=model_name)

        # Initialise and load engines.
        logger.info("Initialising engines...")
        self._survival_engine = SurvivalAnalysisEngine().run()
        self._analytics_engine = AnalyticsEngine().run()

        logger.info("Agent is ready")

    def handle(self, user_query: str) -> dict[str, Any]:
        """Process a natural-language query end-to-end.

        Returns
        -------
        dict
            {
                "query": <original query>,
                "response": <formatted natural-language answer>,
            }
        """
        # Step 1: Classify intent.
        classification = self._classifier.classify(user_query)
        engine_name = classification.get("engine", "none")
        intent = classification.get("intent", "unknown")
        parameters = classification.get("parameters", {})

        logger.info(
            "Query: '%s' -> engine=%s, intent=%s",
            user_query,
            engine_name,
            intent,
        )

        # Step 2: Route to the correct engine.
        raw_result = self._route(engine_name, intent, parameters)

        # Step 3: Format the response via MedGemma.
        formatted_response = self._format_response(user_query, raw_result)

        return {
            "query": user_query,
            "response": formatted_response,
        }

    # Internal routing

    def _route(
        self,
        engine_name: str,
        intent: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch to the appropriate engine."""

        if engine_name == "survival":
            return self._handle_survival(intent, parameters)
        elif engine_name == "analytics":
            return self._analytics_engine.query(intent, parameters)
        else:
            return {
                "error": (
                    "I couldn't determine how to answer that question. "
                    "Try asking about patient demographics, survival "
                    "predictions, lab values, or disease progression."
                )
            }

    def _handle_survival(
        self,
        intent: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle survival-engine intents."""

        if intent == "model_summary":
            model = self._survival_engine.fitted_model
            summary_df = model.summary

            # Convert the summary DataFrame to a readable dict.
            coefficients = []
            for idx, row in summary_df.iterrows():
                coefficients.append({
                    "covariate": str(idx),
                    "coefficient": round(float(row["coef"]), 4),
                    "hazard_ratio": round(float(row["exp(coef)"]), 4),
                    "p_value": round(float(row["p"]), 4),
                    "significance": "significant" if float(row["p"]) < 0.05 else "not significant",
                })

            return {
                "intent": "model_summary",
                "concordance_index": round(
                    self._survival_engine.concordance_index, 4
                ),
                "num_observations": int(model.summary.shape[0]),
                "coefficients": coefficients,
            }

        elif intent == "risk_factors":
            model = self._survival_engine.fitted_model
            summary_df = model.summary.copy()

            # Sort by absolute coefficient to find most influential factors.
            summary_df["abs_coef"] = summary_df["coef"].abs()
            summary_df = summary_df.sort_values("abs_coef", ascending=False)

            risk_factors = []
            for idx, row in summary_df.iterrows():
                direction = "increases" if float(row["coef"]) > 0 else "decreases"
                risk_factors.append({
                    "factor": str(idx),
                    "hazard_ratio": round(float(row["exp(coef)"]), 4),
                    "direction": direction,
                    "p_value": round(float(row["p"]), 4),
                    "significant": bool(float(row["p"]) < 0.05),
                })

            return {
                "intent": "risk_factors",
                "risk_factors": risk_factors,
                "concordance_index": round(
                    self._survival_engine.concordance_index, 4
                ),
            }

        elif intent == "survival_prediction":
            return {
                "intent": "survival_prediction",
                "message": (
                    "Survival prediction requires specific patient features. "
                    "Please provide: age group, Gleason score, bone metastasis "
                    "status (Y/N), and positive node status (Y/N)."
                ),
                "parameters_received": parameters,
            }

        else:
            return {
                "error": f"Unknown survival intent: '{intent}'",
                "supported": ["model_summary", "risk_factors", "survival_prediction"],
            }

    # Response formatting

    def _format_response(
        self,
        user_query: str,
        raw_result: dict[str, Any],
    ) -> str:
        """Use MedGemma to convert raw JSON into a readable answer."""
        try:
            prompt = (
                f"User question: {user_query}\n\n"
                f"Raw data from the engine:\n"
                f"{json.dumps(raw_result, indent=2, default=str)}\n\n"
                f"Please provide a clear, natural-language answer."
            )

            return self._client.chat(
                prompt,
                system=RESPONSE_FORMATTING_SYSTEM,
                temperature=0.3,
            )

        except Exception as exc:
            logger.warning("Response formatting failed: %s", exc)
            # Fall back to raw JSON if the LLM is unavailable.
            return json.dumps(raw_result, indent=2, default=str)
