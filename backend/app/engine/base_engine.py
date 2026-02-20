"""
Base Engine
===========

Abstract base class for all domain-specific analysis engines.

Every engine must be able to:
1. Initialise / load itself               (``run``)
2. Answer a structured query              (``query``)
3. Describe its own capabilities          (``capabilities``)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Common contract that every analysis engine must honour."""

    @abstractmethod
    def run(self) -> "BaseEngine":
        """Initialise the engine (load data / model).

        Returns ``self`` so callers can chain.
        """

    @abstractmethod
    def query(self, intent: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute a domain query and return a JSON-serialisable result.

        Parameters
        ----------
        intent : str
            The fine-grained intent label determined by the classifier
            (e.g. ``"survival_prediction"``, ``"cohort_summary"``).
        parameters : dict
            Extracted parameters relevant to the intent.

        Returns
        -------
        dict
            Result payload that the agent will format for the user.
        """

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        """Return a list of intent strings this engine can handle."""
