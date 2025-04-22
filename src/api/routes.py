
"""
Main router for Prometheum API.
"""

from fastapi import APIRouter

from .routes import auth, storage, health, backup, system

# Create main router
router = APIRouter(prefix="/api")

# Include sub-routers
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(storage.router, prefix="/storage", tags=["Storage"])
router.include_router(health.router, prefix="/health", tags=["Health"])
router.include_router(backup.router, prefix="/backup", tags=["Backup"])
router.include_router(system.router, prefix="/system", tags=["System"])

# Root endpoint
@router.get("/", tags=["System"])
async def root():
    """API status endpoint."""
    return {
        "name": "Prometheum NAS Router OS API",
        "version": "0.1.0",
        "status": "online"
    }

