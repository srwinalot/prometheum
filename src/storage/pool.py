"""
Storage pool management for Prometheum.

This module provides classes for creating and managing storage pools
using different filesystems like ZFS and BTRFS.
"""

import json
import logging
import os
import subprocess
from enum import Enum
from typing import Dict, List, Optional, Union

from .utils import run_command, CommandError

logger = logging.getLogger(__name__)


class FilesystemType(Enum):
    """Supported filesystem types for storage pools."""
    
    ZFS = "zfs"
    BTRFS = "btrfs"
    LVM = "lvm"  # LVM with ext4
    
    @staticmethod
    def from_string(fs_type: str) -> "FilesystemType":
        """Convert string to FilesystemType enum."""
        try:
            return FilesystemType(fs_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported filesystem type: {fs_type}")


class StoragePool:
    """Represents a storage pool in the system."""
    
    def __init__(
        self,
        name: str,
        fs_type: Union[FilesystemType, str],
        devices: List[str],
        mountpoint: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        uuid: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        """Initialize a storage pool.
        
        Args:
            name: The name of the storage pool
            fs_type: The filesystem type
            devices: List of devices in the pool
            mountpoint: Where the pool is mounted
            options: Additional options for the pool
            uuid: Unique identifier (generated if not provided)
            created_at: Creation timestamp (generated if not provided)
        """
        self.name = name
        self.fs_type = fs_type if isinstance(fs_type, FilesystemType) else FilesystemType.from_string(fs_type)
        self.devices = devices
        self.mountpoint = mountpoint or f"/mnt/{name}"
        self.options = options or {}
        
        # Generate UUID if not provided
        if uuid is None:
            import uuid as uuid_lib
            self.uuid = str(uuid_lib.uuid4())
        else:
            self.uuid = uuid
            
        # Set creation timestamp if not provided
        if created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        else:
            self.created_at = created_at
            
        self.status = "unknown"
    
    def to_dict(self) -> Dict:
        """Convert pool to dictionary representation."""
        return {
            "name": self.name,
            "fs_type": self.fs_type.value,
            "devices": self.devices,
            "mountpoint": self.mountpoint,
            "options": self.options,
            "uuid": self.uuid,
            "created_at": self.created_at,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoragePool":
        """Create pool from dictionary representation."""
        return cls(
            name=data["name"],
            fs_type=FilesystemType.from_string(data["fs_type"]),
            devices=data["devices"],
            mountpoint=data.get("mountpoint"),
            options=data.get("options", {}),
            uuid=data.get("uuid"),
            created_at=data.get("created_at")
        )
    
    def __str__(self) -> str:
        """String representation of the pool."""
        return f"StoragePool({self.name}, {self.fs_type.value}, {len(self.devices)} devices)"


class StoragePoolManager:
    """Manager for storage pools."""
    
    def __init__(self, config_path: str = "/var/lib/prometheum/storage/pools.json"):
        """Initialize the storage pool manager.
        
        Args:
            config_path: Path to the pools configuration file
        """
        self.config_path = config_path
        self.pools: Dict[str, StoragePool] = {}
        self._load_pools()
    
    def _load_pools(self) -> None:
        """Load pools from configuration file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    pools_data = json.load(f)
                
                for pool_data in pools_data.get("pools", []):
                    pool = StoragePool.from_dict(pool_data)
                    self.pools[pool.name] = pool
                
                logger.info(f"Loaded {len(self.pools)} storage pools from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading pools: {e}")
        else:
            logger.info(f"Pools configuration file not found at {self.config_path}")
    
    def _save_pools(self) -> None:
        """Save pools to configuration file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        pools_data = {
            "pools": [pool.to_dict() for pool in self.pools.values()]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(pools_data, f, indent=2)
        
        logger.info(f"Saved {len(self.pools)} storage pools to {self.config_path}")
    
    def get_pool(self, name: str) -> Optional[StoragePool]:
        """Get a storage pool by name."""
        return self.pools.get(name)
    
    def list_pools(self) -> List[StoragePool]:
        """List all storage pools."""
        return list(self.pools.values())
    
    def create_pool(
        self,
        name: str,
        fs_type: Union[FilesystemType, str],
        devices: List[str],
        mountpoint: Optional[str] = None,
        options: Optional[Dict[str, str]] = None
    ) -> StoragePool:
        """Create a new storage pool.
        
        Args:
            name: The name of the storage pool
            fs_type: The filesystem type
            devices: List of devices in the pool
            mountpoint: Where to mount the pool
            options: Additional options for the pool
            
        Returns:
            The created StoragePool
            
        Raises:
            ValueError: If the pool name already exists
            CommandError: If there's an error creating the pool
        """
        if name in self.pools:
            raise ValueError(f"Storage pool '{name}' already exists")
        
        # Convert fs_type to enum if needed
        if isinstance(fs_type, str):
            fs_type = FilesystemType.from_string(fs_type)
        
        # Create the pool based on filesystem type
        if fs_type == FilesystemType.ZFS:
            self._create_zfs_pool(name, devices, options or {})
        elif fs_type == FilesystemType.BTRFS:
            self._create_btrfs_pool(name, devices, mountpoint, options or {})
        elif fs_type == FilesystemType.LVM:
            self._create_lvm_pool(name, devices, mountpoint, options or {})
        else:
            raise ValueError(f"Unsupported filesystem type: {fs_type}")
        
        # Create and register the pool
        pool = StoragePool(name, fs_type, devices, mountpoint, options)
        self.pools[name] = pool
        self._save_pools()
        
        logger.info(f"Created storage pool: {pool}")
        return pool
    
    def delete_pool(self, name: str) -> bool:
        """Delete a storage pool.
        
        Args:
            name: The name of the pool to delete
            
        Returns:
            True if the pool was deleted, False otherwise
            
        Raises:
            CommandError: If there's an error deleting the pool
        """
        pool = self.get_pool(name)
        if not pool:
            logger.warning(f"Cannot delete non-existent pool: {name}")
            return False
        
        # Delete the pool based on filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            self._delete_zfs_pool(name)
        elif pool.fs_type == FilesystemType.BTRFS:
            self._delete_btrfs_pool(name, pool.mountpoint)
        elif pool.fs_type == FilesystemType.LVM:
            self._delete_lvm_pool(name, pool.mountpoint)
        
        # Remove from registry
        del self.pools[name]
        self._save_pools()
        
        logger.info(f"Deleted storage pool: {name}")
        return True
    
    def expand_pool(self, name: str, new_devices: List[str]) -> bool:
        """Expand a storage pool with additional devices.
        
        Args:
            name: The name of the pool to expand
            new_devices: List of new devices to add
            
        Returns:
            True if the pool was expanded, False otherwise
            
        Raises:
            CommandError: If there's an error expanding the pool
        """
        pool = self.get_pool(name)
        if not pool:
            logger.warning(f"Cannot expand non-existent pool: {name}")
            return False
        
        # Expand the pool based on filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            self._expand_zfs_pool(name, new_devices)
        elif pool.fs_type == FilesystemType.BTRFS:
            self._expand_btrfs_pool(name, pool.mountpoint, new_devices)
        elif pool.fs_type == FilesystemType.LVM:
            self._expand_lvm_pool(name, new_devices)
        
        # Update devices list
        pool.devices.extend(new_devices)
        self._save_pools()
        
        logger.info(f"Expanded storage pool '{name}' with devices: {new_devices}")
        return True
    
    def update_pool_status(self, name: str) -> Dict:
        """Update and return the status of a storage pool.
        
        Args:
            name: The name of the pool to update
            
        Returns:
            A dictionary with pool status information
            
        Raises:
            ValueError: If the pool doesn't exist
        """
        pool = self.get_pool(name)
        if not pool:
            raise ValueError(f"Storage pool '{name}' doesn't exist")
        
        # Get status based on filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            status = self._get_zfs_pool_status(name)
        elif pool.fs_type == FilesystemType.BTRFS:
            status = self._get_btrfs_pool_status(name, pool.mountpoint)
        elif pool.fs_type == FilesystemType.LVM:
            status = self._get_lvm_pool_status(name)
        else:
            status = {"state": "unknown"}
        
        # Update pool status
        pool.status = status.get("state", "unknown")
        self._save_pools()
        
        return status
    
    # ZFS-specific methods
    def _create_zfs_pool(self, name: str, devices: List[str], options: Dict[str, str]) -> None:
        """Create a ZFS pool."""
        opts = []
        for key, value in options.items():
            opts.append(f"-o {key}={value}")
        
        devices_str = " ".join(devices)
        opts_str = " ".join(opts)
        
        cmd = f"zpool create {opts_str} {name} {devices_str}"
        run_command(cmd)
    
    def _delete_zfs_pool(self, name: str) -> None:
        """Delete a ZFS pool."""
        cmd = f"zpool destroy -f {name}"
        run_command(cmd)
    
    def _expand_zfs_pool(self, name: str, devices: List[str]) -> None:
        """Expand a ZFS pool."""
        devices_str = " ".join(devices)
        cmd = f"zpool add {name} {devices_str}"
        run_command(cmd)
    
    def _get_zfs_pool_status(self, name: str) -> Dict:
        """Get status of a ZFS pool."""
        cmd = f"zpool status {name} -v"
        result = run_command(cmd)
        
        # Parse ZFS pool status - this is a simplified example
        status = {
            "state": "ONLINE",
            "scan": "",
            "devices": []
        }
        
        for line in result.stdout.splitlines():
            line = line.strip()
            if "state:" in line:
                status["state"] = line.split("state:")[1].strip()
            elif "scan:" in line:
                status["scan"] = line.split("scan:")[1].strip()
            elif "ONLINE" in line or "DEGRADED" in line or "FAULTED" in line:
                parts = line.split()
                if len(parts) >= 2 and "/" in parts[0]:
                    status["devices"].append({
                        "name": parts[0],
                        "state": parts[1]
                    })
        
        return status
    
    # BTRFS-specific methods
    def _create_btrfs_pool(self, name: str, devices: List[str], mountpoint: Optional[str], options: Dict[str, str]) -> None:
        """Create a BTRFS pool."""
        mountpoint = mountpoint or f"/mnt/{name}"
        os.makedirs(mountpoint, exist_ok=True)
        
        # Create BTRFS filesystem
        devices_str = " ".join(devices)
        cmd = f"mkfs.btrfs -f -L {name} {devices_str}"
        run_command(cmd)
        
        # Add to fstab for mounting
        fstab_entry = f"LABEL={name} {mountpoint} btrfs defaults 0 0"
        with open("/etc/fstab", "a") as f:
            f.write(f"\n{fstab_entry}\n")
        
        # Mount
        cmd = f"mount {mountpoint}"
        run_command(cmd)
    
    def _delete_btrfs_pool(self, name: str, mountpoint: str) -> None:
        """Delete a BTRFS pool."""
        # Unmount
        cmd = f"umount {mountpoint}"
        run_command(cmd)
        
        # Remove from fstab
        with open("/etc/fstab", "r") as f:
            fstab_lines = f.readlines()
        
        with open("/etc/fstab", "w") as f:
            for line in fstab_lines:
                if f"LABEL={name}" not in line:
                    f.write(line)
    
    def _expand_btrfs_pool(self, name: str, mountpoint: str, new_devices: List[str]) -> None:
        """Expand a BTRFS pool."""
        devices_str = " ".join(new_devices)
        cmd = f"btrfs device add {devices_str} {mountpoint}"
        run_command(cmd)
    
    def _

