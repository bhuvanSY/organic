"""
FastAPI Application Factory for ChemRxn-Intelligence.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chemrxn.api.routes import router
from chemrxn.config.settings import load_config
from chemrxn.utils.device import get_device_info
from chemrxn.utils.logger import get_logger, setup_logging

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup diagnostics
    setup_logging(level="INFO")
    logger.info("Initializing ChemRxn-Intelligence API service...")
    hw_info = get_device_info()
    logger.info(f"Hardware Acceleration Detected: {hw_info}")
    yield
    logger.info("Shutting down ChemRxn-Intelligence API service.")


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    """
    app = FastAPI(
        title="ChemRxn-Intelligence Prediction API",
        description="Organic Reaction Prediction Platform using Reactant SMILES",
        version="0.1.0",
        lifespan=lifespan,
    )

    import os
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Enable CORS for React/Vite frontend (localhost:5173 and Vercel domains)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
        allow_origin_regex=r"https://.*\.vercel\.app" if allowed_origins != ["*"] else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/")
    def root():
        return {
            "name": "ChemRxn-Intelligence API",
            "version": "0.1.0",
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
        }

    @app.get("/health")
    def health_root():
        """Root health check for cloud platform load balancers."""
        return {"status": "ok", "service": "ChemRxn-Intelligence", "version": "0.1.0"}

    return app


app = create_app()
