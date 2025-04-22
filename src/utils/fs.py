
"""
Filesystem utilities.
"""

import os
import logging
from typing import List, Dict, Any

from .command import run_command

logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> bool:
    """
    Ensure directory exists, creating if needed.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False

def path_exists(path: str) -> bool:
    """Check if path exists."""
    return os.path.exists(path)

def scan_devices() -> List[str]:
    """
    Scan for block devices in the system.
    
    Returns:
        List of device names (e.g., ['sda', 'sdb'])
    """
    try:
        result = run_command("lsblk -d -n -o NAME")
        devices = []
        
        for line in result.stdout.splitlines():
            device = line.strip()
            if device and not device.startswith("loop"):
                devices.append(device)
                
        return devices
    
    except Exception as e:
        logger.error(f"Error scanning devices: {e}")
        return []

def get_device_info(device: str) -> Dict[str, Any]:
    """
    Get information about a block device.
    
    Args:
        device: Device name (e.g., 'sda')
        
    Returns:
        Dictionary with device information
    """
    if not device.startswith("/dev/"):
        device = f"/dev/{device}"
        
    try:
        size_cmd = run_command(f"lsblk -b -d -n -o SIZE {device}")
        size_bytes = int(size_cmd.stdout.strip())
        
        fs_cmd = run_command(f"lsblk -n -o FSTYPE {device}", check=False)
        fs_type = fs_cmd.stdout.strip() if fs_cmd.success else None
        
        return {
            "name": os.path.basename(device),
            "path": device,
            "size_bytes": size_bytes,
            "size_human": format_size(size_bytes),
            "fs_type": fs_type
        }
    
    except Exception as e:
        logger.error(f"Error getting device info for {device}: {e}")
        return {"name": os.path.basename(device), "path": device, "error": str(e)}

def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size_bytes < 1024 or unit == 'PB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

