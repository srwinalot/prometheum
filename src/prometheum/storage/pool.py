"""Storage pool management module for testing."""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class StoragePoolManager:
    """Test implementation of storage pool manager."""
    
    def __init__(self):
        """Initialize with test data."""
        # Test data structure
        self._test_pools = {
            "tank": {
                "devices": ["sda", "sdb"],
                "state": "ONLINE",
                "size": 1099511627776,  # 1 TB
                "used": 549755813888,   # 512 GB
                "free": 549755813888    # 512 GB
            },
            "backup": {
                "devices": ["sdc", "sdd"],
                "state": "ONLINE",
                "size": 2199023255552,  # 2 TB
                "used": 1099511627776,  # 1 TB
                "free": 1099511627776   # 1 TB
            }
        }
        
        self._device_errors = {
            "sdd": {
                "read_errors": 2,
                "write_errors": 0,
                "checksum_errors": 1
            }
        }

    def get_all_devices(self) -> List[str]:
        """Get all devices across all pools."""
        devices = set()
        for pool_info in self._test_pools.values():
            devices.update(pool_info["devices"])
        return sorted(list(devices))

    def get_pools(self) -> List[str]:
        """Get all pool names."""
        return sorted(self._test_pools.keys())

    def get_pool_status(self, pool: str) -> Dict[str, Any]:
        """Get status of a pool."""
        if pool not in self._test_pools:
            return {"error": f"Pool {pool} not found"}
        
        pool_info = self._test_pools[pool]
        return {
            "state": pool_info["state"],
            "capacity": {
                "total": pool_info["size"],
                "used": pool_info["used"],
                "free": pool_info["free"],
                "percentage": int((pool_info["used"] / pool_info["size"]) * 100)
            },
            "devices": {
                device: "ONLINE" for device in pool_info["devices"]
            },
            "errors": {
                device: self._device_errors[device]
                for device in pool_info["devices"]
                if device in self._device_errors
            }
        }

    def get_pool_devices(self, pool: str) -> List[str]:
        """Get devices in a pool."""
        return sorted(self._test_pools.get(pool, {}).get("devices", []))

    def check_device_errors(self, device: str) -> Dict[str, Any]:
        """Check errors for a device."""
        if device in self._device_errors:
            return {
                "state": "ONLINE",
                "errors": self._device_errors[device]
            }
        
        # Check if device exists in any pool
        for pool_info in self._test_pools.values():
            if device in pool_info["devices"]:
                return {
                    "state": "ONLINE",
                    "errors": {
                        "read_errors": 0,
                        "write_errors": 0,
                        "checksum_errors": 0
                    }
                }
        
        return {"error": f"Device {device} not found"}

    def list_pools(self) -> List:
        """List all pools (compatibility method)."""
        class Pool:
            def __init__(self, name, devices):
                self.name = name
                self.devices = devices
        
        return [
            Pool(name, info["devices"])
            for name, info in self._test_pools.items()
        ]
