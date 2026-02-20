"""
Flower-based Monitoring Server & Orchestrator

Orchestrates federated monitoring queries using the Flower framework.

The server runs for exactly **1 round**.  In that single round:
1.  ``configure_evaluate`` encodes a ``MonitoringQuery`` as NumPy arrays
    and sends it to every connected monitoring client.
2.  Each client runs ``evaluate()`` — loads local monitoring data,
    computes aggregates, and returns aggregate metrics.
3.  ``aggregate_evaluate`` decodes results and stores them.

The ``MonitoringOrchestrator`` wraps the Flower round with:
- LLM-based query classification (natural language -> MonitoringQueryType)
- Cross-site result merging
- LLM-based response formatting

No patient-level data ever crosses site boundaries.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

# Low-level Flower internals
from flwr.compat.server.app import init_defaults, start_grpc_server
from flwr.server.server import run_fl

from backend.app.federated.monitoring_client import (
    MonitoringClient,
    monitoring_metrics_to_result,
    query_to_ndarrays,
)
from backend.app.llm.prompts import (
    MONITORING_QUERY_CLASSIFICATION_SYSTEM,
    MONITORING_RESPONSE_FORMATTING_SYSTEM,
)
from backend.app.schema.monitoring_schema import (
    AggregateMonitoringResult,
    MonitoringQuery,
    MonitoringQueryResponse,
    MonitoringQueryType,
    SiteMonitoringResult,
)

logger = logging.getLogger(__name__)

# Site registry for monitoring data
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

MONITORING_SITE_REGISTRY: dict[str, Path] = {
    "site_a": _PROJECT_ROOT / "data" / "monitoring" / "site_a",
    "site_b": _PROJECT_ROOT / "data" / "monitoring" / "site_b",
}


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Flower Strategy for Monitoring Queries
class MonitoringQueryStrategy(Strategy):
    """Flower strategy that distributes a monitoring query and collects results.

    Uses the evaluate round to send the query (encoded as NumPy arrays)
    to all monitoring clients and collect their aggregate results.

    Parameters
    ----------
    query : MonitoringQuery
        The monitoring query to distribute to all sites.
    num_clients : int
        Expected number of client sites.
    """

    def __init__(
        self,
        query: MonitoringQuery,
        num_clients: int = 2,
    ) -> None:
        super().__init__()
        self._query = query
        self._num_clients = num_clients
        self._query_arrays = query_to_ndarrays(query)

        # Results collected after the round
        self.site_results: list[SiteMonitoringResult] = []

    # Flower strategy interface

    def initialize_parameters(self, client_manager):
        """Return the encoded query as initial parameters."""
        return ndarrays_to_parameters(self._query_arrays)

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Send the monitoring query to every connected client."""
        client_manager.wait_for(self._num_clients)
        clients = list(client_manager.all().values())

        logger.info(
            "Round %d: distributing monitoring query '%s' to %d client(s).",
            server_round,
            self._query.query_type.value,
            len(clients),
        )

        evaluate_ins = EvaluateIns(
            parameters=ndarrays_to_parameters(self._query_arrays),
            config={},
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Decode and store each client's aggregate monitoring result."""
        logger.info(
            "Round %d: received %d result(s), %d failure(s).",
            server_round,
            len(results),
            len(failures),
        )

        for _, evaluate_res in results:
            try:
                site_result = monitoring_metrics_to_result(evaluate_res.metrics)
                self.site_results.append(site_result)
                logger.info(
                    "  Site '%s': %d patients monitored.",
                    site_result.site_id,
                    site_result.total_patients_monitored,
                )
            except Exception as exc:
                logger.error("Failed to decode client result: %s", exc)

        for failure in failures:
            logger.error("Client failure: %s", failure)

        return None, {}

    # Unused strategy methods (required by interface)

    def configure_fit(self, server_round, parameters, client_manager):
        """Not used."""
        return []

    def aggregate_fit(self, server_round, results, failures):
        """Not used."""
        return None, {}

    def evaluate(self, server_round, parameters):
        """Not used."""
        return None


# Monitoring Orchestrator

# Monitoring prompts are centralized in backend.app.llm.prompts.py
class MonitoringOrchestrator:
    """End-to-end orchestrator for federated monitoring queries.

    1. LLM classifies the natural-language query into a MonitoringQueryType.
    2. Builds a MonitoringQuery and runs a Flower federated round.
    3. Merges site-level results into a global aggregate.
    4. LLM formats the final response.

    Parameters
    ----------
    site_registry : dict[str, Path] | None
        Mapping of ``site_id -> monitoring_data_directory``.
    """

    def __init__(
        self,
        site_registry: dict[str, Path] | None = None,
    ) -> None:
        self.site_registry = site_registry or MONITORING_SITE_REGISTRY
        self._port: int = 0

    def query(
        self, trial_name: str, user_query: str, *, use_extraction: bool = False
    ) -> MonitoringQueryResponse:
        """Process a natural-language monitoring query end-to-end."""

        # Step 1: Classify query type via LLM
        query_type, parameters = self._classify_query(user_query)

        logger.info(
            "Monitoring query classified: type='%s', params=%s, extraction=%s",
            query_type.value,
            parameters,
            use_extraction,
        )

        # Step 2: Build the monitoring query
        monitoring_query = MonitoringQuery(
            trial_name=trial_name,
            query_type=query_type,
            parameters=parameters,
            natural_language_query=user_query,
            use_extraction=use_extraction,
        )

        # Step 3: Run federated round
        site_results = self._run_federated_query(monitoring_query)

        # Step 4: Merge results across sites
        global_aggregate = self._merge_results(query_type, site_results)

        # Step 5: Format response via LLM
        response_text = self._format_response(
            user_query, global_aggregate.result_data, site_results
        )

        # Build status
        num_sites = len(self.site_registry)
        any_errors = any(r.errors for r in site_results)
        status = "completed"
        if len(site_results) < num_sites:
            status = "partial"
        elif any_errors:
            status = "completed_with_warnings"

        total_patients = sum(r.total_patients_monitored for r in site_results)
        message = (
            f"Monitoring query '{query_type.value}' completed across "
            f"{len(site_results)} site(s) covering {total_patients} patients."
        )

        return MonitoringQueryResponse(
            trial_name=trial_name,
            query=user_query,
            query_type=query_type,
            site_results=site_results,
            global_result=global_aggregate.result_data,
            response=response_text,
            status=status,
            message=message,
        )

    # Query classification

    def _classify_query(
        self, user_query: str
    ) -> tuple[MonitoringQueryType, dict[str, Any]]:
        """Use MedGemma to classify the monitoring query."""
        from backend.app.llm.medgemma_client import MedGemmaClient

        client = MedGemmaClient.get_instance()

        try:
            raw_text = client.chat(
                user_query,
                system=MONITORING_QUERY_CLASSIFICATION_SYSTEM,
                temperature=0.0,
            )

            # Strip markdown code fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                raw_text = raw_text.rsplit("```", 1)[0].strip()

            parsed = json.loads(raw_text)
            query_type = MonitoringQueryType(parsed.get("query_type", "overall_progress"))
            parameters = parsed.get("parameters", {})

            return query_type, parameters

        except Exception as exc:
            logger.warning(
                "Monitoring query classification failed: %s. "
                "Defaulting to 'overall_progress'.",
                exc,
            )
            return MonitoringQueryType.OVERALL_PROGRESS, {}

    # Federated execution

    def _run_federated_query(
        self, query: MonitoringQuery
    ) -> list[SiteMonitoringResult]:
        """Run a Flower federated round to collect monitoring data."""
        num_sites = len(self.site_registry)
        strategy = MonitoringQueryStrategy(
            query=query,
            num_clients=num_sites,
        )

        logger.info(
            "Starting federated monitoring query '%s' for trial '%s' "
            "across %d site(s).",
            query.query_type.value,
            query.trial_name,
            num_sites,
        )

        # Pick a free port
        self._port = _find_free_port()
        logger.info("Using port %d for Flower gRPC.", self._port)

        # Start Flower server in a thread
        server_thread = threading.Thread(
            target=self._start_server,
            args=(strategy,),
            name="flower-monitoring-server",
            daemon=True,
        )
        server_thread.start()

        # Give the server a moment to bind
        time.sleep(1.5)

        # Start client threads
        client_threads: list[threading.Thread] = []
        for site_id, data_dir in self.site_registry.items():
            t = threading.Thread(
                target=self._start_client,
                args=(site_id, data_dir),
                name=f"flower-monitoring-client-{site_id}",
                daemon=True,
            )
            t.start()
            client_threads.append(t)
            logger.info("Started monitoring client for '%s'.", site_id)

        # Wait for completion
        server_thread.join(timeout=120)  # Monitoring is faster than screening
        for t in client_threads:
            t.join(timeout=120)

        return strategy.site_results

    def _start_server(self, strategy: MonitoringQueryStrategy) -> None:
        """Run the Flower server (blocking)."""
        try:
            config = fl.server.ServerConfig(num_rounds=1)

            server, server_config = init_defaults(
                server=None,
                config=config,
                strategy=strategy,
                client_manager=None,
            )
            logger.info("Flower monitoring server initialised.")

            server_address = f"0.0.0.0:{self._port}"
            grpc_server = start_grpc_server(
                client_manager=server.client_manager(),
                server_address=server_address,
                max_message_length=2_147_483_647,
                certificates=None,
            )
            logger.info(
                "Flower monitoring gRPC server running on %s.", server_address
            )

            run_fl(server=server, config=server_config)
            grpc_server.stop(grace=1)
            logger.info("Flower monitoring server finished.")

        except Exception as exc:
            logger.error("Flower monitoring server error: %s", exc)

    def _start_client(self, site_id: str, data_dir: Path) -> None:
        """Run a Flower monitoring client (blocking) for one site."""
        try:
            client = MonitoringClient(
                site_id=site_id,
                monitoring_data_dir=data_dir,
            )
            client_address = f"127.0.0.1:{self._port}"
            fl.client.start_client(
                server_address=client_address,
                client=client.to_client(),
                insecure=True,
            )
        except Exception as exc:
            logger.error(
                "Flower monitoring client '%s' error: %s", site_id, exc,
            )

    # Cross-site aggregation

    def _merge_results(
        self,
        query_type: MonitoringQueryType,
        site_results: list[SiteMonitoringResult],
    ) -> AggregateMonitoringResult:
        """Merge site-level aggregates into a global result."""
        if not site_results:
            return AggregateMonitoringResult(
                query_type=query_type,
                result_data={"error": "No site results to merge."},
            )

        merge_map = {
            MonitoringQueryType.ADVERSE_EVENTS:   self._merge_adverse_events,
            MonitoringQueryType.VISIT_PROGRESS:   self._merge_visit_progress,
            MonitoringQueryType.RESPONSE_SUMMARY: self._merge_response_summary,
            MonitoringQueryType.DROPOUT_SUMMARY:  self._merge_dropout_summary,
            MonitoringQueryType.LAB_TRENDS:       self._merge_lab_trends,
            MonitoringQueryType.OVERALL_PROGRESS: self._merge_overall_progress,
        }

        merger = merge_map.get(query_type)
        if merger is None:
            return AggregateMonitoringResult(
                query_type=query_type,
                result_data={"error": f"No merger for query type: {query_type}"},
            )

        try:
            result_data = merger(site_results)
            total_patients = sum(r.total_patients_monitored for r in site_results)
            return AggregateMonitoringResult(
                query_type=query_type,
                total_sites=len(site_results),
                total_patients_monitored=total_patients,
                result_data=result_data,
            )
        except Exception as exc:
            logger.error("Merge failed for %s: %s", query_type.value, exc)
            return AggregateMonitoringResult(
                query_type=query_type,
                result_data={"error": f"Merge failed: {exc}"},
            )

    def _merge_adverse_events(
        self, site_results: list[SiteMonitoringResult]
    ) -> dict[str, Any]:
        """Merge adverse event aggregates across sites."""
        total_patients = sum(r.total_patients_monitored for r in site_results)
        total_ae = sum(r.result_data.get("total_ae_count", 0) for r in site_results)
        patients_with_ae = sum(r.result_data.get("patients_with_any_ae", 0) for r in site_results)
        sae_count = sum(r.result_data.get("sae_count", 0) for r in site_results)
        sae_patients = sum(r.result_data.get("sae_patients", 0) for r in site_results)

        # Merge by_grade
        merged_by_grade: dict[str, int] = {}
        for r in site_results:
            for grade, count in r.result_data.get("by_grade", {}).items():
                merged_by_grade[grade] = merged_by_grade.get(grade, 0) + count

        # Merge top AEs
        merged_ae_freq: dict[str, int] = {}
        for r in site_results:
            for ae, count in r.result_data.get("top_adverse_events", {}).items():
                merged_ae_freq[ae] = merged_ae_freq.get(ae, 0) + count
        top_aes = dict(sorted(merged_ae_freq.items(), key=lambda x: -x[1])[:10])

        # Merge by category
        merged_categories: dict[str, int] = {}
        for r in site_results:
            for cat, count in r.result_data.get("by_category", {}).items():
                merged_categories[cat] = merged_categories.get(cat, 0) + count

        # Merge by severity
        merged_severity: dict[str, int] = {}
        for r in site_results:
            for sev, count in r.result_data.get("by_severity", {}).items():
                merged_severity[sev] = merged_severity.get(sev, 0) + count

        return {
            "total_patients": total_patients,
            "num_sites": len(site_results),
            "total_ae_count": total_ae,
            "patients_with_any_ae": patients_with_ae,
            "ae_rate_pct": round(100 * patients_with_ae / total_patients, 2) if total_patients > 0 else 0.0,
            "by_grade": merged_by_grade,
            "by_severity": merged_severity,
            "sae_count": sae_count,
            "sae_patients": sae_patients,
            "sae_rate_pct": round(100 * sae_patients / total_patients, 2) if total_patients > 0 else 0.0,
            "top_adverse_events": top_aes,
            "by_category": merged_categories,
        }

    def _merge_visit_progress(
        self, site_results: list[SiteMonitoringResult]
    ) -> dict[str, Any]:
        """Merge visit progress across sites."""
        total_patients = sum(r.total_patients_monitored for r in site_results)
        total_scheduled = sum(r.result_data.get("total_scheduled_visits", 0) for r in site_results)
        total_completed = sum(r.result_data.get("completed_visits", 0) for r in site_results)
        total_missed = sum(r.result_data.get("missed_visits", 0) for r in site_results)

        return {
            "total_patients": total_patients,
            "num_sites": len(site_results),
            "total_scheduled_visits": total_scheduled,
            "completed_visits": total_completed,
            "missed_visits": total_missed,
            "adherence_rate_pct": round(100 * total_completed / total_scheduled, 2) if total_scheduled > 0 else 0.0,
        }

    def _merge_response_summary(
        self, site_results: list[SiteMonitoringResult]
    ) -> dict[str, Any]:
        """Merge treatment response data across sites."""
        total_patients = sum(r.total_patients_monitored for r in site_results)
        total_assessed = sum(r.result_data.get("assessed_patients", 0) for r in site_results)

        # Merge response distribution
        merged_dist: dict[str, int] = {}
        for r in site_results:
            for cat, count in r.result_data.get("response_distribution", {}).items():
                merged_dist[cat] = merged_dist.get(cat, 0) + count

        cr = merged_dist.get("CR", 0)
        pr = merged_dist.get("PR", 0)
        sd = merged_dist.get("SD", 0)
        pd_count = merged_dist.get("PD", 0)

        orr = cr + pr
        dcr = cr + pr + sd

        return {
            "total_patients": total_patients,
            "num_sites": len(site_results),
            "assessed_patients": total_assessed,
            "response_distribution": merged_dist,
            "overall_response_rate_pct": round(100 * orr / total_assessed, 2) if total_assessed > 0 else 0.0,
            "disease_control_rate_pct": round(100 * dcr / total_assessed, 2) if total_assessed > 0 else 0.0,
            "progressive_disease_rate_pct": round(100 * pd_count / total_assessed, 2) if total_assessed > 0 else 0.0,
        }

    def _merge_dropout_summary(
        self, site_results: list[SiteMonitoringResult]
    ) -> dict[str, Any]:
        """Merge dropout/retention data across sites."""
        total_patients = sum(r.total_patients_monitored for r in site_results)
        total_active = sum(r.result_data.get("active_patients", 0) for r in site_results)
        total_dropouts = sum(r.result_data.get("dropout_count", 0) for r in site_results)

        # Merge dropout reasons across sites
        merged_reasons: dict[str, int] = {}
        for r in site_results:
            for reason, count in r.result_data.get("by_reason", {}).items():
                merged_reasons[reason] = merged_reasons.get(reason, 0) + count

        return {
            "total_patients": total_patients,
            "num_sites": len(site_results),
            "active_patients": total_active,
            "dropout_count": total_dropouts,
            "dropout_rate_pct": round(100 * total_dropouts / total_patients, 2) if total_patients > 0 else 0.0,
            "retention_rate_pct": round(100 * (total_patients - total_dropouts) / total_patients, 2) if total_patients > 0 else 0.0,
            "by_reason": merged_reasons,
        }

    def _merge_lab_trends(
        self, site_results: list[SiteMonitoringResult]
    ) -> dict[str, Any]:
        """Merge lab trend data across sites (weighted means)."""
        total_patients = sum(r.total_patients_monitored for r in site_results)

        # Collect all lab names
        all_labs: set[str] = set()
        for r in site_results:
            trends = r.result_data.get("lab_trends", {})
            all_labs.update(trends.keys())

        merged_trends: dict[str, list[dict[str, Any]]] = {}

        for lab_name in sorted(all_labs):
            # Collect data by visit across sites
            visit_data: dict[int, list[tuple[float, int]]] = {}  # visit -> [(mean, count), ...]
            for r in site_results:
                lab_points = r.result_data.get("lab_trends", {}).get(lab_name, [])
                for point in lab_points:
                    visit = point.get("visit", 0)
                    mean = point.get("mean", 0.0)
                    count = point.get("count", 0)
                    visit_data.setdefault(visit, []).append((mean, count))

            # Compute weighted mean per visit
            trend_points = []
            for visit in sorted(visit_data.keys()):
                entries = visit_data[visit]
                total_count = sum(c for _, c in entries)
                weighted_mean = sum(m * c for m, c in entries) / total_count if total_count > 0 else 0.0
                trend_points.append({
                    "visit": visit,
                    "count": total_count,
                    "mean": round(weighted_mean, 2),
                })
            merged_trends[lab_name] = trend_points

        return {
            "total_patients": total_patients,
            "num_sites": len(site_results),
            "lab_trends": merged_trends,
            "labs_reported": list(merged_trends.keys()),
        }

    def _merge_overall_progress(
        self, site_results: list[SiteMonitoringResult]
    ) -> dict[str, Any]:
        """Merge overall progress dashboards across sites."""
        total_patients = sum(r.total_patients_monitored for r in site_results)
        total_active = sum(r.result_data.get("active_patients", 0) for r in site_results)
        total_dropouts = sum(r.result_data.get("dropout_count", 0) for r in site_results)

        # Weighted averages for rates
        def _weighted_avg(key: str) -> float:
            total_w = 0.0
            total_n = 0
            for r in site_results:
                n = r.total_patients_monitored
                val = r.result_data.get(key, 0.0)
                total_w += val * n
                total_n += n
            return round(total_w / total_n, 2) if total_n > 0 else 0.0

        return {
            "total_patients": total_patients,
            "num_sites": len(site_results),
            "active_patients": total_active,
            "dropout_count": total_dropouts,
            "retention_rate_pct": _weighted_avg("retention_rate_pct"),
            "visit_adherence_pct": _weighted_avg("visit_adherence_pct"),
            "ae_rate_pct": _weighted_avg("ae_rate_pct"),
            "sae_rate_pct": _weighted_avg("sae_rate_pct"),
            "overall_response_rate_pct": _weighted_avg("overall_response_rate_pct"),
            "disease_control_rate_pct": _weighted_avg("disease_control_rate_pct"),
        }

    # LLM response formatting

    def _format_response(
        self,
        user_query: str,
        global_result: dict[str, Any],
        site_results: list[SiteMonitoringResult],
    ) -> str:
        """Use MedGemma to convert raw monitoring data into a readable answer."""
        from backend.app.llm.medgemma_client import MedGemmaClient

        client = MedGemmaClient.get_instance()

        try:
            # Include per-site summaries for context
            site_summaries = []
            for r in site_results:
                site_summaries.append({
                    "site_id": r.site_id,
                    "patients_monitored": r.total_patients_monitored,
                    "data_as_of": r.data_as_of,
                })

            prompt = (
                f"User question: {user_query}\n\n"
                f"Global aggregated data across all sites:\n"
                f"{json.dumps(global_result, indent=2, default=str)}\n\n"
                f"Per-site summary:\n"
                f"{json.dumps(site_summaries, indent=2, default=str)}\n\n"
                f"Please provide a clear, natural-language answer."
            )

            return client.chat(
                prompt,
                system=MONITORING_RESPONSE_FORMATTING_SYSTEM,
                temperature=0.3,
            )

        except Exception as exc:
            logger.warning("Response formatting failed: %s", exc)
            return json.dumps(global_result, indent=2, default=str)
