from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging

from prometheum.api.routes import storage, backup, health
from prometheum.storage.health import HealthMonitor
from prometheum.api.dependencies import get_health_monitor

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Prometheum NAS API",
    description="API for managing Prometheum NAS Router OS",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(storage.router, prefix="/api/storage", tags=["storage"])
app.include_router(backup.router, prefix="/api/backup", tags=["backup"])
app.include_router(health.router, prefix="/api/health", tags=["health"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
async def init_health_monitoring():
    """Initialize and start health monitoring on application startup."""
    try:
        health_monitor = get_health_monitor()
        if not health_monitor.monitoring_active:
            logger.info("Starting health monitoring service")
            health_monitor.start_monitoring()
    except Exception as e:
        logger.error(f"Failed to initialize health monitoring: {e}")

@app.on_event("shutdown")
async def shutdown_health_monitoring():
    """Stop health monitoring on application shutdown."""
    try:
        health_monitor = get_health_monitor()
        if health_monitor.monitoring_active:
            logger.info("Stopping health monitoring service")
            health_monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Error stopping health monitoring: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

