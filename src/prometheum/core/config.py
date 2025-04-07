"""
Configuration management for Prometheum cloud system.

This module handles loading, saving, and validation of 
configuration settings for all components of the system.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging


class Config:
    """
    Configuration manager for Prometheum.
    
    Handles reading and writing configuration from various sources
    and provides access to configuration values for other components.
    """
    
    DEFAULT_CONFIG = {
        "version": 1,
        "storage": {
            "compression": True,
            "encryption": True,
            "default_quota_mb": 10240  # 10GB default quota
        },
        "sync": {
            "interval_seconds": 300,  # 5 minutes
            "max_file_size_mb": 1024,
            "excluded_patterns": [".DS_Store", "Thumbs.db", "*.tmp"]
        },
        "security": {
            "encryption_algorithm": "AES-256-GCM",
            "key_length": 256,
            "max_password_attempts": 5
        },
        "network": {
            "port": 8080,
            "enable_remote_access": False,
            "allowed_ips": ["127.0.0.1"]
        },
        "devices": {
            "max_devices": 10,
            "inactive_timeout_days": 30
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Custom path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine config path
        if config_path:
            self.config_file = Path(config_path)
        else:
            # Default to user's home directory or current working directory
            home_dir = Path.home()
            self.config_file = home_dir / ".prometheum" / "config.json"
        
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from the config file.
        
        If the file doesn't exist, use default configuration.
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_file}")
            else:
                self.config = self.DEFAULT_CONFIG.copy()
                self.logger.info("Using default configuration")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config = self.DEFAULT_CONFIG.copy()
    
    def save(self) -> bool:
        """
        Save current configuration to the config file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., "sync.interval_seconds")
            default: Default value to return if the key doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        parts = key_path.split('.')
        current = self.config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the config value
            value: Value to set
        """
        parts = key_path.split('.')
        current = self.config
        
        # Navigate to the deepest dict
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
    
    def initialize(self) -> bool:
        """
        Initialize the configuration for first-time use.
        
        Returns:
            bool: True if successful
        """
        if not self.config_file.exists():
            self.config = self.DEFAULT_CONFIG.copy()
            return self.save()
        return True
    
    def reset_to_defaults(self) -> None:
        """Reset all configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()

