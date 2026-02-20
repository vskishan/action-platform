"""
MedGemma Client


A centralised wrapper around the Ollama API for the MedGemma model.

This gives us a single place to:

* Configure the model name and default parameters.
* Validate that Ollama is reachable and the model is pulled.
* Swap out the backend later (e.g. HuggingFace, vLLM) without touching every consumer.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import ollama

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "alibayram/medgemma"


class MedGemmaClient:
    """Thin, reusable wrapper over the Ollama chat API.

    Parameters
    ----------
    model : str | None
        Ollama model tag.  Defaults to ``DEFAULT_MODEL``.
    default_temperature : float
        Temperature used when the caller does not specify one.

    Usage
    -----
    ::

        client = MedGemmaClient()
        answer = client.chat("What is breast cancer?")
    """

    # Class-level singleton so the entire app shares one instance.
    _instance: Optional["MedGemmaClient"] = None

    def __init__(
        self,
        model: str | None = None,
        default_temperature: float = 0.3,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.default_temperature = default_temperature

    # Factory / singleton 
    @classmethod
    def get_instance(cls, **kwargs) -> "MedGemmaClient":
        """Return (and optionally create) the shared singleton."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    # Health checks
    def is_available(self) -> bool:
        """Return ``True`` if Ollama is reachable and the model exists."""
        try:
            ollama.show(self.model)
            return True
        except Exception:
            return False

    def check_ready(self) -> None:
        """Raise ``RuntimeError`` if Ollama / model is not available.

        Designed to be called once at application startup.
        """
        # 1. Is Ollama running?
        try:
            models = ollama.list()
            logger.info(
                "Ollama is running.  %d model(s) available.", len(models.models)
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot connect to Ollama.  Make sure the Ollama service "
                f"is running.  Error: {exc}"
            ) from exc

        # 2. Is the required model pulled?
        available = [m.model for m in models.models]
        if not any(m.startswith(self.model) for m in available):
            raise RuntimeError(
                f"Model '{self.model}' is not available in Ollama.\n"
                f"  Run:  ollama pull {self.model}\n"
                f"  Available models: {available}"
            )

        logger.info("Required model '%s' is available.", self.model)

    # Core chat method
    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        raw: bool = False,
    ) -> str:
        """Send a single-turn chat message and return the response text.

        Parameters
        ----------
        prompt : str
            The user message.
        system : str | None
            Optional system prompt.
        temperature : float | None
            Sampling temperature (overrides ``default_temperature``).
        raw : bool
            If ``True``, return the complete Ollama ``ChatResponse``
            object instead of just the text.

        Returns
        -------
        str
            The assistant's reply (stripped of leading/trailing whitespace).
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature or self.default_temperature},
        )

        if raw:
            return response                # type: ignore[return-value]
        return response.message.content.strip()

    # Multi-turn convenience
    def chat_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        """Send an arbitrary message list and return the response text."""
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature or self.default_temperature},
        )
        return response.message.content.strip()
