"""
System-wide configuration for Prometheum cloud storage system.

This module defines the core configuration settings for the Prometheum
personal cloud storage system including storage locations, network settings,
security parameters, and system behavior.
"""

import os
import json
import socket
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Base directories
HOME_DIR = os.path.expanduser("~")
DEFAULT_CONFIG_DIR = os.path.join(HOME_DIR, ".prometheum")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_CONFIG_DIR, "data")
DEFAULT_LOG_DIR = os.path.join(DEFAULT_CONFIG_DIR, "logs")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_CONFIG_DIR, "cache")
DEFAULT_BACKUP_DIR = os.path.join(DEFAULT_CONFIG_DIR, "backups")

# Application settings
APP_NAME = "Prometheum"
APP_VERSION = "1.0.0"
DEBUG_MODE = False

# Network settings
API_HOST = "0.0.0.0"  # Listen on all interfaces
API_PORT = 8080
API_URL_PREFIX = "/api"
ENABLE_SSL = True
SSL_CERT_PATH = os.path.join(DEFAULT_CONFIG_DIR, "ssl", "cert.pem")
SSL_KEY_PATH = os.path.join(DEFAULT_CONFIG_DIR, "ssl", "key.pem")

# Authentication settings
JWT_SECRET_KEY = None  # Will be auto-generated on first run
JWT_EXPIRATION_MINUTES = 60
PASSWORD_HASH_ALGORITHM = "argon2"
MIN_PASSWORD_LENGTH = 12
REQUIRE_COMPLEX_PASSWORDS = True

# Storage settings
DEFAULT_QUOTA_MB = 10 * 1024  # 10 GB per user
MAX_UPLOAD_SIZE_MB = 1024  # 1 GB max upload size
STORAGE_ENCRYPTION_ENABLED = True
STORAGE_ENCRYPTION_ALGORITHM = "AES-256-GCM"
FILE_VERSIONING_ENABLED = True
MAX_FILE_VERSIONS = 10

# Network interception settings
ENABLE_NETWORK_INTERCEPTION = True
DNS_SPOOFING_ENABLED = False  # DNS spoofing is disabled by default for security reasons
ARP_POISONING_ENABLED = False # ARP poisoning is disabled by default for security reasons
DEFAULT_INTERCEPTION_MODE = "PROXY_REDIRECT"  # Safe default

# Cloud service settings
SUPPORTED_CLOUD_SERVICES = ["APPLE_ICLOUD", "MICROSOFT_ONEDRIVE", "GOOGLE_DRIVE", "DROPBOX"]
DEFAULT_CLOUD_SERVICE = "APPLE_ICLOUD"

# Synchronization settings
SYNC_INTERVAL_SECONDS = 300  # 5 minutes
MAX_SYNC_RETRY_COUNT = 3
CONFLICT_RESOLUTION_STRATEGY = "NEWEST_WINS"  # Options: NEWEST_WINS, LOCAL_WINS, REMOTE_WINS, ALWAYS_ASK

# System performance settings
MAX_CONCURRENT_UPLOADS = 5
MAX_CONCURRENT_DOWNLOADS = 5
MAX_CONCURRENT_SYNC_TASKS = 3
BACKGROUND_TASK_THREADS = 4
USE_PROCESS_POOL = True
PROCESS_POOL_SIZE = max(2, os.cpu_count() - 1) if os.cpu_count() else 2

# Security settings
SCAN_FILES_FOR_MALWARE = True
ENFORCE_DEVICE_VERIFICATION = True
ALLOW_EXTERNAL_SHARING = True
REQUIRE_2FA_FOR_ADMIN = True

class SystemConfig:
    """System configuration manager that handles loading, saving, and accessing config."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the system configuration.
        
        Args:
            config_path: Optional path to the configuration file
        """
        self.config_path = config_path or os.path.join(DEFAULT_CONFIG_DIR, "system_config.json")
        self.config = self._load_defaults()
        self._load_from_file()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        hostname = socket.gethostname()
        
        return {
            "app": {
                "name": APP_NAME,
                "version": APP_VERSION,
                "debug_mode": DEBUG_MODE,
                "hostname": hostname,
                "platform": platform.system(),
                "python_version": platform.python_version()
            },
            "paths": {
                "config_dir": DEFAULT_CONFIG_DIR,
                "data_dir": DEFAULT_DATA_DIR,
                "log_dir": DEFAULT_LOG_DIR,
                "cache_dir": DEFAULT_CACHE_DIR,
                "backup_dir": DEFAULT_BACKUP_DIR
            },
            "network": {
                "api_host": API_HOST,
                "api_port": API_PORT,
                "api_url_prefix": API_URL_PREFIX,
                "enable_ssl": ENABLE_SSL,
                "ssl_cert_path": SSL_CERT_PATH,
                "ssl_key_path": SSL_KEY_PATH
            },
            "auth": {
                "jwt_secret_key": JWT_SECRET_KEY,
                "jwt_expiration_minutes": JWT_EXPIRATION_MINUTES,
                "password_hash_algorithm": PASSWORD_HASH_ALGORITHM,
                "min_password_length": MIN_PASSWORD_LENGTH,
                "require_complex_passwords": REQUIRE_COMPLEX_PASSWORDS,
                "require_2fa_for_admin": REQUIRE_2FA_FOR_ADMIN
            },
            "storage": {
                "default_quota_mb": DEFAULT_QUOTA_MB,
                "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
                "encryption_enabled": STORAGE_ENCRYPTION_ENABLED,
                "encryption_algorithm": STORAGE_ENCRYPTION_ALGORITHM,
                "file_versioning_enabled": FILE_VERSIONING_ENABLED,
                "max_file_versions": MAX_FILE_VERSIONS
            },
            "interception": {
                "enabled": ENABLE_NETWORK_INTERCEPTION,
                "dns_spoofing_enabled": DNS_SPOOFING_ENABLED, 
                "arp_poisoning_enabled": ARP_POISONING_ENABLED,
                "default_mode": DEFAULT_INTERCEPTION_MODE,
                "supported_cloud_services": SUPPORTED_CLOUD_SERVICES,
                "default_cloud_service": DEFAULT_CLOUD_SERVICE
            },
            "sync": {
                "interval_seconds": SYNC_INTERVAL_SECONDS,
                "max_retry_count": MAX_SYNC_RETRY_COUNT,
                "conflict_resolution_strategy": CONFLICT_RESOLUTION_STRATEGY
            },
            "performance": {
                "max_concurrent_uploads": MAX_CONCURRENT_UPLOADS,
                "max_concurrent_downloads": MAX_CONCURRENT_DOWNLOADS,
                "max_concurrent_sync_tasks": MAX_CONCURRENT_SYNC_TASKS,
                "background_task_threads": BACKGROUND_TASK_THREADS,
                "use_process_pool": USE_PROCESS_POOL,
                "process_pool_size": PROCESS_POOL_SIZE
            },
            "security": {
                "scan_files_for_malware": SCAN_FILES_FOR_MALWARE,
                "enforce_device_verification": ENFORCE_DEVICE_VERIFICATION,
                "allow_external_sharing": ALLOW_EXTERNAL_SHARING
            }
        }
    
    def _load_from_file(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults, preserving file values
                self._merge_configs(self.config, file_config)
        except Exception as e:
            print(f"Warning: Could not load configuration from {self.config_path}: {str(e)}")
    
    def _merge_configs(self, default_config: Dict[str, Any], file_config: Dict[str, Any]) -> None:
        """
        Merge file configuration with defaults recursively.
        
        Args:
            default_config: Default configuration dictionary
            file_config: Configuration from file
        """
        for key, value in file_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_configs(default_config[key], value)
            else:
                default_config[key] = value
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            bool: True if successful
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            The configuration value or default
        """
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def initialize_directories(self) -> None:
        """Create necessary directories for the system."""
        for path_key in ["config_dir", "data_dir", "log_dir", "cache_dir", "backup_dir"]:
            path = self.get("paths", path_key)
            if path:
                os.makedirs(path, exist_ok=True)
        
        # Create SSL directory if SSL is enabled
        if self.get("network", "enable_ssl"):
            ssl_dir = os.path.dirname(self.get("network", "ssl_cert_path"))
            os.makedirs(ssl_dir, exist_ok=True)

# Global instance (for convenience)
system_config = SystemConfig()

