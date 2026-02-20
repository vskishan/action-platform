"""Federated route package exports."""

from backend.app.routes.federated.monitoring_route import router as monitoring_router
from backend.app.routes.federated.screening_route import router as screening_router

__all__ = ["monitoring_router", "screening_router"]
