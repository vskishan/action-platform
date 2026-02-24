"""
MedGemma Client

A centralised wrapper around the Ollama API for the MedGemma model.

This gives us a single place to:

* Configure the model name and default parameters.
* Validate that Ollama is reachable and the model is pulled.
* Swap out the backend later (e.g. HuggingFace, vLLM) without
  touching every consumer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import ollama

logger = logging.getLogger(__name__)

# Model name is resolved at import time from the environment.
DEFAULT_MODEL = os.environ.get("MEDGEMMA_MODEL")

# Hard cap on a single Ollama inference call (seconds).
# The timeout is set on the underlying httpx client so when Ollama stalls
# the connection itself raises an exception — no background thread needed.
DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "180"))


class MedGemmaClient:
    """Thin, reusable wrapper over the Ollama chat API.

    Parameters
    ----------
    model : str | None
        Ollama model tag.  Defaults to ``DEFAULT_MODEL``.
    default_temperature : float
        Temperature used when the caller does not specify one.
    request_timeout : float
        Per-request HTTP timeout in seconds.  When Ollama stalls
        mid-inference this raises an ``ollama.ResponseError`` /
        ``httpx.TimeoutException`` that propagates to the caller so the
        patient loop can skip the hung patient and continue.

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
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.default_temperature = default_temperature
        self.request_timeout = request_timeout

        # Use a persistent ollama.Client so the timeout is applied at the
        # HTTP layer (httpx) rather than requiring a separate thread.
        # This means a stalled ollama.chat() call will raise a real exception
        # after `request_timeout` seconds, which propagates naturally through
        # screen_and_audit() back to the caller's try/except block.
        self._ollama = ollama.Client(timeout=request_timeout)
        logger.info(
            "MedGemmaClient initialised — model=%s, timeout=%.0fs",
            self.model,
            self.request_timeout,
        )

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
            self._ollama.show(self.model)
            return True
        except Exception:
            return False

    def check_ready(self) -> None:
        """Raise ``RuntimeError`` if Ollama / model is not available.

        Designed to be called once at application startup.
        """
        # 1. Is Ollama running?
        try:
            models = self._ollama.list()
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
        max_tokens: int | None = None,
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
        max_tokens : int | None
            Maximum tokens to generate (maps to Ollama's ``num_predict``).
            Use this to cap short-format responses (e.g. DECISION/REASON)
            and avoid the model generating verbose preamble.
        raw : bool
            If ``True``, return the complete Ollama ``ChatResponse``
            object instead of just the text.

        Returns
        -------
        str
            The assistant's reply (stripped of leading/trailing whitespace).

        Raises
        ------
        ollama.ResponseError / httpx.TimeoutException
            If the Ollama server stalls and the per-request timeout fires.
            The caller (federated_client patient loop) catches this and
            skips the hung patient rather than blocking indefinitely.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict = {"temperature": temperature or self.default_temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # self._ollama is an ollama.Client with timeout set at the httpx layer.
        # When Ollama stalls mid-token-generation this call raises an exception
        # after self.request_timeout seconds — no separate thread required.
        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
            keep_alive="5m",
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
        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature or self.default_temperature},
            keep_alive="5m",
        )
        return response.message.content.strip()
