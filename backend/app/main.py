"""
ACTION Platform - Backend Layer

Entry point for the backend server.

"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.llm.medgemma_client import MedGemmaClient
from backend.app.routes.cohort.query_route import router as cohort_router
from backend.app.routes.federated.monitoring_route import router as monitoring_router
from backend.app.routes.federated.screening_route import router as federated_router
from backend.app.routes.workflow.workflow_route import router as workflow_router
from backend.app.routes.workflow.job_route import router as job_router

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Shared MedGemma client (singleton)
_medgemma = MedGemmaClient.get_instance()


# Application lifespan (startup / shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic for the FastAPI application."""
    # --- Startup ---
    logger.info("Starting ACTION Platform...")
    _medgemma.check_ready()
    logger.info("All preflight checks passed.")

    yield  # Application runs here.

    # --- Shutdown ---
    logger.info("Shutting down ACTION Platform.")


# Application
app = FastAPI(
    title="ACTION Platform",
    description=(
        "AI-powered platform for clinical trial research"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the frontend (served from file:// or localhost) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(cohort_router)
app.include_router(federated_router)
app.include_router(monitoring_router)
app.include_router(workflow_router)
app.include_router(job_router)

# Health check
@app.get("/health")
async def health_check():
    """Liveness probe: also verifies Ollama connectivity."""
    model_ready = _medgemma.is_available()

    return {
        "status": "healthy" if model_ready else "degraded",
        "ollama_connected": model_ready,
        "model": _medgemma.model,
    }

# Serve the frontend as static files (fallback, after all API routes).
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logger.info("Serving frontend from %s", _frontend_dir)
