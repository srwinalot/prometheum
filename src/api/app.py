
"""
FastAPI application for Prometheum NAS Router OS.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, backup
from .dependencies import get_health_monitor
from ..utils.config import load_config

logger = logging.getLogger(__name__)

# Load API configuration
API_CONFIG = load_config("/var/lib/prometheum/api_config.json", {
    "cors_origins": ["*"],
    "debug": False
})

def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Prometheum API",
        description="API for Prometheum NAS Router OS",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CONFIG.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include main router
    app.include_router(router)
    app.include_router(backup.router, prefix="/api/backup", tags=["Backup"])
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("API starting")
        
        # Start health monitoring
        health_monitor = get_health_monitor()
        if health_monitor and not health_monitor.monitoring_active:
            health_monitor.start_monitoring()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("API shutting down")
        
        # Stop health monitoring
        health_monitor = get_health_monitor()
        if health_monitor and health_monitor.monitoring_active:
            health_monitor.stop_monitoring()
    
    return app

