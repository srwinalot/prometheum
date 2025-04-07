"""
Main cloud management system for Prometheum.

This module provides the central CloudManager class that coordinates
all functionality of the Prometheum personal cloud storage system.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..sync.sync_manager import SyncManager
from ..devices.device_manager import DeviceManager
from ..security.encryption import EncryptionManager
from .config import Config


class CloudManager:
    """
    Main controller class for the Prometheum cloud system.
    
    This class orchestrates all components of the system including
    synchronization, device management, storage, and security.
    """
    
    def __init__(self, storage_path: Union[str, Path], config_file: Optional[str] = None):
        """
        Initialize the cloud manager with a storage location.
        
        Args:
            storage_path: Base directory for all cloud storage
            config_file: Optional path to a custom configuration file
        """
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration
        self.config = Config(config_file)
        
        # Initialize components
        self.sync_manager = SyncManager(self)
        self.device_manager = DeviceManager(self)
        self.encryption = EncryptionManager()
        
        self.logger.info(f"Initialized CloudManager with storage at {self.storage_path}")
    
    def setup(self) -> bool:
        """
        Complete one-time setup of the cloud system.
        
        Returns:
            bool: True if setup was successful
        """
        try:
            self.logger.info("Starting Prometheum cloud setup")
            
            # Create necessary directories
            for directory in ["files", "backups", "devices", "temp"]:
                (self.storage_path / directory).mkdir(exist_ok=True)
            
            # Initialize configuration
            self.config.initialize()
            
            # Set up device management
            self.device_manager.initialize()
            
            # Set up encryption
            self.encryption.initialize()
            
            self.logger.info("Prometheum cloud setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            return False
    
    def add_sync_directory(self, local_path: str, 
                          sync_policy: str = "two-way",
                          auto_backup: bool = False) -> str:
        """
        Add a directory to be synchronized with the cloud.
        
        Args:
            local_path: Path to the local directory to sync
            sync_policy: Synchronization policy ('one-way', 'two-way', etc.)
            auto_backup: Whether to automatically back up this directory
            
        Returns:
            str: ID of the new sync configuration
        """
        return self.sync_manager.add_directory(local_path, sync_policy, auto_backup)
    
    def start_sync_service(self) -> bool:
        """
        Start the background synchronization service.
        
        Returns:
            bool: True if service started successfully
        """
        return self.sync_manager.start_service()
    
    def stop_sync_service(self) -> bool:
        """
        Stop the background synchronization service.
        
        Returns:
            bool: True if service stopped successfully
        """
        return self.sync_manager.stop_service()
    
    def get_status(self) -> Dict:
        """
        Get the current status of the cloud system.
        
        Returns:
            Dict: Status information including sync stats, device info, etc.
        """
        status = {
            "storage": self._get_storage_stats(),
            "sync": self.sync_manager.get_status(),
            "devices": self.device_manager.get_connected_devices(),
            "version": "0.1.0",  # Hardcoded for now
        }
        return status
    
    def _get_storage_stats(self) -> Dict:
        """
        Get statistics about storage usage.
        
        Returns:
            Dict: Storage statistics
        """
        total_space = 0
        used_space = 0
        
        # This is just a placeholder - real implementation would calculate actual disk usage
        return {
            "total_bytes": total_space,
            "used_bytes": used_space,
            "free_bytes": total_space - used_space,
            "usage_percent": (used_space / total_space * 100) if total_space else 0
        }


def setup(storage_path: str, **kwargs) -> CloudManager:
    """
    Convenience function to create and set up a CloudManager.
    
    Args:
        storage_path: Path where cloud data will be stored
        **kwargs: Additional arguments to pass to CloudManager
        
    Returns:
        CloudManager: Initialized cloud manager instance
    """
    manager = CloudManager(storage_path, **kwargs)
    manager.setup()
    return manager

