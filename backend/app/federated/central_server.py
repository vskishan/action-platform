"""
Central Server for Federated Patient Screening

Orchestrates federated patient screening across simulated hospital sites.

Each site's ``ScreeningClient.fit()`` is invoked directly in-process
(one thread per site).  The Flower serialisation contract is honoured:
criteria are encoded as NumPy arrays before being handed to the client,
and only aggregate counts come back.  **No patient-level data crosses
site boundaries.**
"""

import logging
import threading
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from backend.app.federated.federated_client import (
    ScreeningClient,
    criteria_to_ndarrays,
    ndarrays_to_result,
)
from backend.app.schema.screening_schema import (
    FederatedScreeningResponse,
    ScreeningCriteria,
    SiteScreeningResult,
)

logger = logging.getLogger(__name__)


# Site registry
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SITE_REGISTRY: dict[str, Path] = {
    "site_a": _PROJECT_ROOT / "data" / "ehr" / "site_a",
    "site_b": _PROJECT_ROOT / "data" / "ehr" / "site_b",
}

# Custom flower strategy for federated patient screening.
class PatientScreeningStrategy(Strategy):
    """Flower strategy for federated patient screening.

    The strategy holds the screening criteria and collects aggregate
    results from every client.

    Parameters
    ----------
    criteria : ScreeningCriteria
        The inclusion / exclusion rules to distribute.
    num_clients : int
        Expected number of client sites.
    """

    def __init__(
        self,
        criteria: ScreeningCriteria,
        num_clients: int = 2,
    ) -> None:
        super().__init__()
        self._criteria = criteria
        self._num_clients = num_clients
        self._criteria_arrays = criteria_to_ndarrays(criteria)

        # Results collected after the round
        self.site_results: list[SiteScreeningResult] = []

    # Flower strategy interface (only fit-related methods are used)

    def initialize_parameters(self, client_manager):
        """Return initial 'parameters' — not used for screening."""
        return ndarrays_to_parameters(self._criteria_arrays)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Send criteria to every connected client."""
        # Wait until all expected clients have connected
        client_manager.wait_for(self._num_clients)
        clients = list(client_manager.all().values())

        logger.info(
            "Round %d: distributing criteria to %d client(s).",
            server_round,
            len(clients),
        )

        # Pack criteria into FitIns
        fit_ins = FitIns(
            parameters=ndarrays_to_parameters(self._criteria_arrays),
            config={},
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Decode and store each client's aggregate result."""
        logger.info(
            "Round %d: received %d result(s), %d failure(s).",
            server_round,
            len(results),
            len(failures),
        )

        for _, fit_res in results:
            try:
                arrays = parameters_to_ndarrays(fit_res.parameters)
                site_result = ndarrays_to_result(arrays)
                self.site_results.append(site_result)
                logger.info(
                    "  Site '%s': %d / %d eligible.",
                    site_result.site_id,
                    site_result.eligible_patients,
                    site_result.total_patients,
                )
            except Exception as exc:
                logger.error("Failed to decode client result: %s", exc)

        for failure in failures:
            logger.error("Client failure: %s", failure)

        # Return empty params (no global model to maintain)
        return ndarrays_to_parameters([]), {}

    # Unused strategy methods (required by interface)

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Not used — no evaluation round."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Not used."""
        return None, {}

    def evaluate(self, server_round, parameters):
        """Not used."""
        return None

# Central Orchestrator class
class CentralServer:
    """Orchestrate a full federated screening round using Flower.

    Parameters
    ----------
    site_registry : dict[str, Path] | None
        Mapping of ``site_id -> ehr_data_directory``.
    server_address : str
        gRPC address for the Flower server.
    """

    def __init__(
        self,
        site_registry: dict[str, Path] | None = None,
    ) -> None:
        self.site_registry = site_registry or SITE_REGISTRY

    def run_screening(
        self, criteria: ScreeningCriteria
    ) -> FederatedScreeningResponse:
        """Execute one federated screening round.

        Each site's ScreeningClient is invoked directly in-process.
        The Flower ``fit()`` contract is honoured: criteria are encoded
        as NumPy arrays before being handed to each client, and only
        aggregate counts come back.  **No patient-level data is shared
        between sites or returned to the caller.**

        This avoids Flower's gRPC transport layer which is unreliable
        when running server + clients in background threads (deprecated
        ``start_client`` API in Flower ≥ 1.5).
        """
        num_sites = len(self.site_registry)

        logger.info(
            "Central server starting screening for trial '%s' "
            "across %d site(s) (in-process Flower clients).",
            criteria.trial_name,
            num_sites,
        )

        # Encode criteria the same way a Flower server would
        criteria_arrays = criteria_to_ndarrays(criteria)

        # Run each site's client.fit() — optionally in parallel threads
        site_results: list[SiteScreeningResult] = []
        client_errors: list[str] = []

        def _run_site(site_id: str, ehr_dir: Path) -> None:
            """Run screening for one site (thread target)."""
            try:
                client = ScreeningClient(
                    site_id=site_id,
                    ehr_data_dir=ehr_dir,
                )
                result_arrays, num_examples, metrics = client.fit(
                    parameters=criteria_arrays,
                    config={},
                )
                site_result = ndarrays_to_result(result_arrays)
                site_results.append(site_result)
                logger.info(
                    "  Site '%s': %d / %d eligible.",
                    site_result.site_id,
                    site_result.eligible_patients,
                    site_result.total_patients,
                )
            except Exception as exc:
                msg = f"Site '{site_id}' screening failed: {exc}"
                logger.error(msg, exc_info=True)
                client_errors.append(msg)

        # Launch one thread per site so they run concurrently
        threads: list[threading.Thread] = []
        for site_id, ehr_dir in self.site_registry.items():
            t = threading.Thread(
                target=_run_site,
                args=(site_id, ehr_dir),
                name=f"screen-{site_id}",
                daemon=True,
            )
            t.start()
            threads.append(t)
            logger.info("Started screening thread for '%s'.", site_id)

        # Wait for all sites (MedGemma calls can take minutes)
        for t in threads:
            t.join(timeout=900)

        # Aggregate
        aggregate_total = sum(r.total_patients for r in site_results)
        aggregate_eligible = sum(r.eligible_patients for r in site_results)
        any_errors = any(r.errors for r in site_results) or bool(client_errors)

        # Aggregate audit metrics from self-correcting screening
        aggregate_corrected = sum(r.corrected_count for r in site_results)
        aggregate_flagged = sum(r.flagged_for_review_count for r in site_results)
        aggregate_high_conf = sum(r.high_confidence_count for r in site_results)
        aggregate_low_conf = sum(r.low_confidence_count for r in site_results)

        status = "completed"
        if len(site_results) < num_sites:
            status = "partial"
        elif any_errors:
            status = "completed_with_warnings"

        audit_summary = ""
        if aggregate_corrected or aggregate_flagged:
            audit_summary = (
                f" | Self-correction: {aggregate_corrected} decision(s) corrected, "
                f"{aggregate_flagged} flagged for review, "
                f"{aggregate_high_conf} high-confidence, "
                f"{aggregate_low_conf} low-confidence."
            )

        message = (
            f"Screening complete. "
            f"{aggregate_eligible} of {aggregate_total} total patients "
            f"across {len(site_results)} site(s) are eligible for "
            f"trial '{criteria.trial_name}'.{audit_summary}"
        )

        logger.info(message)

        return FederatedScreeningResponse(
            trial_name=criteria.trial_name,
            criteria=criteria,
            site_results=site_results,
            aggregate_total_patients=aggregate_total,
            aggregate_eligible_patients=aggregate_eligible,
            status=status,
            message=message,
            aggregate_corrected_count=aggregate_corrected,
            aggregate_flagged_for_review=aggregate_flagged,
            aggregate_high_confidence=aggregate_high_conf,
            aggregate_low_confidence=aggregate_low_conf,
        )


