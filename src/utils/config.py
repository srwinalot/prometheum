
"""
Configuration management utilities.
"""

import json
import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def load_config(path: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file with fallback to defaults.
    
    Args:
        path: Path to config file
        default: Default config to use if file missing
        
    Returns:
        Configuration dictionary
    """
    if default is None:
        default = {}
        
    if not os.path.exists(path):
        logger.info(f"Config file not found: {path}, using defaults")
        return default.copy()
        
    try:
        with open(path, 'r') as f:
            config = json.load(f)
            
        # Merge with defaults for missing keys
        merged = default.copy()
        merged.update(config)
        return merged
    
    except Exception as e:
        logger.error(f"Error loading config {path}: {e}")
        return default.copy()

def save_config(path: str, config: Dict[str, Any]) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        path: Path to save config
        config: Configuration to save
        
    Returns:
        True if saved successfully
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    
    except Exception as e:
        logger.error(f"Error saving config {path}: {e}")
        return False

