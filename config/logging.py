"""
Logging configuration for Prometheum cloud storage system.

This module configures the logging system for the Prometheum application,
providing different log handlers and formatters for various components.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Import system configuration
try:
    from config.system import system_config
except ImportError:
    # If system_config is not available, use default values
    class DummyConfig:
        def get(self, section, key, default=None):
            return default
    system_config = DummyConfig()

# Default log locations
DEFAULT_LOG_DIR = system_config.get("paths", "log_dir", os.path.expanduser("~/.prometheum/logs"))
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "prometheum.log")
DEFAULT_ERROR_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "error.log")
DEFAULT_ACCESS_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "access.log")
DEFAULT_SYNC_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "sync.log")
DEFAULT_SECURITY_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "security.log")

# Log rotation settings
LOG_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 10  # Keep 10 backup files

# Log levels
DEFAULT_LOG_LEVEL = logging.INFO
DEBUG_LOG_LEVEL = logging.DEBUG

# Log formatters
STANDARD_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
ACCESS_FORMAT = "%(asctime)s - %(message)s"

def setup_logging(
    app_name: str = "prometheum",
    log_dir: Optional[str] = None,
    log_level: Optional[int] = None,
    console_output: bool = True,
    file_output: bool = True,
    debug_mode: Optional[bool] = None
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        app_name: Name of the application
        log_dir: Directory for log files
        log_level: Logging level
        console_output: Whether to output to console
        file_output: Whether to output to file
        debug_mode: Whether to use debug mode
        
    Returns:
        logging.Logger: The configured root logger
    """
    # Use system config if parameters not provided
    log_dir = log_dir or DEFAULT_LOG_DIR
    debug_mode = debug_mode if debug_mode is not None else system_config.get("app", "debug_mode", False)
    log_level = log_level or (DEBUG_LOG_LEVEL if debug_mode else DEFAULT_LOG_LEVEL)
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    standard_formatter = logging.Formatter(STANDARD_FORMAT)
    detailed_formatter = logging.Formatter(DETAILED_FORMAT)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(standard_formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
    
    # Add file handlers if requested
    if file_output:
        # Main log file
        main_log_file = os.path.join(log_dir, f"{app_name}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=LOG_MAX_SIZE_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        
        # Error log file (errors only)
        error_log_file = os.path.join(log_dir, "error.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=LOG_MAX_SIZE_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        error_handler.setFormatter(detailed_formatter)
        error_

