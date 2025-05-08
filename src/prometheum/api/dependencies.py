"""
API dependency injection configuration.

This module provides dependency injection functions for the FastAPI application,
including storage management, health monitoring, and other core services.
"""

from typing import Optional
import logging

from prometheum.storage.pool import StoragePoolManager
from prometheum.storage.volume import VolumeManager
from prometheum.storage.health import HealthMonitor

# Set up logging
logger = logging.getLogger(__name__)

# Singleton instances
_health_monitor: Optional[HealthMonitor] = None
_pool_manager: Optional[StoragePoolManager] = None
_volume_manager: Optional[VolumeManager] = None

def get_storage_pool_manager() -> StoragePoolManager:
    """Get or create the StoragePoolManager singleton instance."""
    global _pool_manager
    if _pool_manager is None:
        logger.info("Initializing StoragePoolManager")
        _pool_manager = StoragePoolManager()
    return _pool_manager

def get_volume_manager() -> VolumeManager:
    """Get or create the VolumeManager singleton instance."""
    global _volume_manager
    if _volume_manager is None:
        logger.info("Initializing VolumeManager")
        _volume_manager = VolumeManager(get_storage_pool_manager())
    return _volume_manager

def get_health_monitor() -> HealthMonitor:
    """Get or create the HealthMonitor singleton instance."""
    global _health_monitor
    if _health_monitor is None:
        logger.info("Initializing HealthMonitor")
        pool_manager = get_storage_pool_manager()
        volume_manager = get_volume_manager()
        _health_monitor = HealthMonitor(
            pool_manager=pool_manager,
            volume_manager=volume_manager
        )
    return _health_monitor

