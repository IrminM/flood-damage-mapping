"""FastAPI Application factory for Flood Damage Mapper backend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title="Flood Damage Mapper Backend",
        description=(
            "FastAPI backend and orchestration layer for rapid multi-evidence flood damage mapping, "
            "infrastructure impact assessment, cut-off settlement network analysis, and DEM downstream tracing."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for future frontend/dashboard integrations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers under both /api/v1 and /api
    app.include_router(router, prefix="/api/v1")
    app.include_router(router, prefix="/api")

    # Mount optional frontend static files if packaged alongside backend
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/floodintel", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    @app.get("/", summary="Root API Index")
    def root_index():
        return {
            "name": "Flood Damage Mapper API",
            "version": "1.0.0",
            "documentation": "/docs",
            "health_check": "/api/v1/health",
            "model_a_status": "/api/v1/model-a/status",
            "attention_unet_status": "/api/v1/attention-unet/status",
            "data_readiness": "/api/v1/data-status",
            "submit_analysis": "/api/v1/analyses",
            "analysis_endpoint": "/api/analysis",
            "report_endpoint": "/api/report",
            "layers_endpoint": "/api/layers",
            "frontend_dashboard": "/floodintel/",
        }

    return app


app = create_app()
