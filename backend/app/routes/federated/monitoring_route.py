"""
Federated Monitoring API Route

Exposes endpoint(s) for federated treatment-arm monitoring queries.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.federated.monitoring_server import MonitoringOrchestrator
from backend.app.schema.monitoring_schema import (
    MonitoringQueryRequest,
    MonitoringQueryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/federated/monitoring", tags=["monitoring"])


@router.post("/query", response_model=MonitoringQueryResponse)
def query_monitoring(request: MonitoringQueryRequest) -> MonitoringQueryResponse:
    """Query treatment-arm monitoring data across federated sites."""
    try:
        orchestrator = MonitoringOrchestrator()
        return orchestrator.query(
            request.trial_name,
            request.query,
            use_extraction=request.use_extraction,
        )
    except Exception as exc:
        logger.exception("Federated monitoring query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Federated monitoring query failed: {exc}",
        )
