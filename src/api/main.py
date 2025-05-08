"""
Main application entry point for Prometheum API.
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .routes import auth, storage, health, system, backup
from .dependencies import get_health_monitor
from ..utils.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Load API configuration
config_path = "/var/lib/prometheum/api_config.json"
default_config = {
    "cors_origins": ["*"],
    "debug": False,
    "title": "Prometheum NAS Router OS API",
    "version": "0.1.0",
    "description": "API for managing the Prometheum NAS Router OS"
}
config = load_config(config_path, default_config)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title=config.get("title", "Prometheum API"),
        description=config.get("description", "API for Prometheum NAS Router OS"),
        version=config.get("version", "0.1.0"),
        docs_url=None,  # Custom docs URL
        redoc_url=None  # Custom ReDoc URL
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "Validation error", "details": exc.errors()}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error"}
        )
    
    # Mount routers
    app.include_router(
        auth.router,
        prefix="/api/auth",
        tags=["Authentication"]
    )
    
    app.include_router(
        storage.router,
        prefix="/api/storage",
        tags=["Storage"]
    )
    
    app.include_router(
        health.router,
        prefix="/api/health",
        tags=["Health"]
    )
    
    app.include_router(
        system.router,
        prefix="/api/system",
        tags=["System"]
    )
    
    app.include_router(
        backup.router,
        prefix="/api/backup",
        tags=["Backup"]
    )
    
    # API documentation
    @app.get("/api/docs", include_in_schema=False)
    async def get_documentation():
        """Serve Swagger UI documentation."""
        return get_swagger_ui_html(
            openapi_url="/api/openapi.json",
            title=f"{config.get('title')} - API Documentation"
        )
    
    @app.get("/api/openapi.json", include_in_schema=False)
    async def get_openapi_schema():
        """Serve OpenAPI schema."""
        return get_openapi(
            title=config.get("title", "Prometheum API"),
            version=config.get("version", "0.1.0"),
            description=config.get("description", "API for Prometheum NAS Router OS"),
            routes=app.routes
        )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint that redirects to documentation."""
        return {
            "name": config.get("title", "Prometheum API"),
            "version": config.get("version", "0.1.0"),
            "documentation": "/api/docs"
        }
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Run at application startup."""
        logger.info("Starting API server")
        
        # Start health monitoring if enabled
        if config.get("health_monitoring", True):
            health_monitor = get_health_monitor()
            if health_monitor and not health_monitor.monitoring_active:
                health_monitor.start_monitoring()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run at application shutdown."""
        logger.info("Shutting down API server")
        
        # Stop health monitoring
        health_monitor = get_health_monitor()
        if health_monitor and health_monitor.monitoring_active:
            health_monitor.stop_monitoring()
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    # For direct execution during development
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("API_PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=config.get("debug", False)
    )

