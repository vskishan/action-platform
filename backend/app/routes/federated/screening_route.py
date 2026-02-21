"""
Federated Screening API Routes

Exposes an endpoint that triggers the federated patient-screening.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.schema.screening_schema import (
    FederatedScreeningResponse,
    ScreeningCriteria,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/federated", tags=["federated"])


@router.post("/screen", response_model=FederatedScreeningResponse)
def screen_patients(criteria: ScreeningCriteria) -> FederatedScreeningResponse:
    """Trigger a federated patient-screening round."""
    try:
        from backend.app.federated.central_server import CentralServer

        server = CentralServer()
        result = server.run_screening(criteria)
        return result

    except Exception as exc:
        logger.exception("Federated screening failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Federated screening failed: {exc}",
        )
