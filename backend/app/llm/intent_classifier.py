"""
Intent Classifier

Uses MedGemma model to classify a user's natural-language query
into a structured intent that can be routed to the correct engine.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.app.llm.medgemma_client import MedGemmaClient
from backend.app.llm.prompts import INTENT_CLASSIFICATION_SYSTEM

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify user queries into engine + intent pairs via MedGemma."""

    def __init__(self, model_name: str | None = None) -> None:
        self._client = MedGemmaClient.get_instance(model=model_name)

        # Verify that the model is available
        if self._client.is_available():
            logger.info("MedGemma model '%s' is available.", self._client.model)
        else:
            logger.warning("Model '%s' not found.", self._client.model)

    def classify(self, user_query: str) -> dict[str, Any]:
        """Return ``{"engine": ..., "intent": ..., "parameters": ...}``.

        Falls back to a safe default on LLM or parsing failure.
        """
        try:
            raw_text = self._client.chat(
                user_query,
                system=INTENT_CLASSIFICATION_SYSTEM,
                temperature=0.0,
            )

            # Strip markdown code fences if the model wraps output.
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                raw_text = raw_text.rsplit("```", 1)[0].strip()

            parsed = json.loads(raw_text)
            logger.info("Classified intent: %s", parsed)
            return parsed

        except json.JSONDecodeError as exc:
            logger.warning(
                "Intent classification failed to parse JSON: %s — raw: %s",
                exc,
                raw_text if "raw_text" in dir() else "N/A",
            )
            return {
                "engine": "none",
                "intent": "unknown",
                "parameters": {},
                "error": f"Failed to parse LLM response: {exc}",
            }
        except Exception as exc:
            logger.error("Intent classification error: %s", exc)
            return {
                "engine": "none",
                "intent": "unknown",
                "parameters": {},
                "error": str(exc),
            }
