"""
Volume management for Prometheum.

This module provides classes for creating and managing volumes within storage pools,
including snapshot management, quota management, and mount point handling.
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from .pool import FilesystemType, StoragePool, StoragePoolManager
from .utils import run_command, CommandError

logger = logging.getLogger(__name__)


class VolumeType(Enum):
    """Types of volumes in the system."""
    
    FILESYSTEM = "filesystem"  # Regular filesystem
    BLOCK = "block"            # Block device (e.g., for VMs)
    SHARE = "share"            # Network share (SMB, NFS)
    
    @staticmethod
    def from_string(vol_type: str) -> "VolumeType":
        """Convert string to VolumeType enum."""
        try:
            return VolumeType(vol_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported volume type: {vol_type}")


class ShareProtocol(Enum):
    """Network share protocols."""
    
    SMB = "smb"      # Windows/macOS compatible
    NFS = "nfs"      # Unix/Linux compatible
    AFP = "afp"      # Apple Filing Protocol (legacy)
    WEBDAV = "webdav"  # HTTP-based protocol
    
    @staticmethod
    def from_string(protocol: str) -> "ShareProtocol":
        """Convert string to ShareProtocol enum."""
        try:
            return ShareProtocol(protocol.lower())
        except ValueError:
            raise ValueError(f"Unsupported share protocol: {protocol}")


class Volume:
    """Represents a volume in the storage system."""
    
    def __init__(
        self,
        name: str,
        pool_name: str,
        path: str,
        type: VolumeType = VolumeType.FILESYSTEM,
        size: Optional[str] = None,
        mountpoint: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        share_config: Optional[Dict] = None,
        uuid: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        """Initialize a volume.
        
        Args:
            name: The name of the volume
            pool_name: The name of the parent storage pool
            path: Path within the pool (e.g. zfs dataset name or btrfs subvolume path)
            type: Type of volume (filesystem, block, share)
            size: Size limit (e.g. "10G", "1T")
            mountpoint: Where the volume is mounted
            options: Additional options for the volume
            share_config: Configuration for network shares
            uuid: Unique identifier (generated if not provided)
            created_at: Creation timestamp (generated if not provided)
        """
        self.name = name
        self.pool_name = pool_name
        self.path = path
        self.type = type if isinstance(type, VolumeType) else VolumeType.from_string(type)
        self.size = size
        self.mountpoint = mountpoint or f"/mnt/{pool_name}/{name}"
        self.options = options or {}
        self.share_config = share_config or {}
        
        # Generate UUID if not provided
        if uuid is None:
            import uuid as uuid_lib
            self.uuid = str(uuid_lib.uuid4())
        else:
            self.uuid = uuid
            
        # Set creation timestamp if not provided
        if created_at is None:
            self.created_at = datetime.now().isoformat()
        else:
            self.created_at = created_at
            
        self.status = "unknown"
        self.snapshots = []  # List of snapshot names
    
    def to_dict(self) -> Dict:
        """Convert volume to dictionary representation."""
        return {
            "name": self.name,
            "pool_name": self.pool_name,
            "path": self.path,
            "type": self.type.value,
            "size": self.size,
            "mountpoint": self.mountpoint,
            "options": self.options,
            "share_config": self.share_config,
            "uuid": self.uuid,
            "created_at": self.created_at,
            "status": self.status,
            "snapshots": self.snapshots
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Volume":
        """Create volume from dictionary representation."""
        return cls(
            name=data["name"],
            pool_name=data["pool_name"],
            path=data["path"],
            type=VolumeType.from_string(data["type"]),
            size=data.get("size"),
            mountpoint=data.get("mountpoint"),
            options=data.get("options", {}),
            share_config=data.get("share_config", {}),
            uuid=data.get("uuid"),
            created_at=data.get("created_at")
        )
    
    def __str__(self) -> str:
        """String representation of the volume."""
        return f"Volume({self.name}, pool={self.pool_name}, type={self.type.value})"


class VolumeManager:
    """Manager for volumes in the storage system."""
    
    def __init__(
        self, 
        pool_manager: StoragePoolManager,
        config_path: str = "/var/lib/prometheum/storage/volumes.json"
    ):
        """Initialize the volume manager.
        
        Args:
            pool_manager: The storage pool manager
            config_path: Path to the volumes configuration file
        """
        self.pool_manager = pool_manager
        self.config_path = config_path
        self.volumes: Dict[str, Volume] = {}  # name -> Volume
        self._load_volumes()
    
    def _load_volumes(self) -> None:
        """Load volumes from configuration file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    volumes_data = json.load(f)
                
                for volume_data in volumes_data.get("volumes", []):
                    volume = Volume.from_dict(volume_data)
                    self.volumes[volume.name] = volume
                
                logger.info(f"Loaded {len(self.volumes)} volumes from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading volumes: {e}")
        else:
            logger.info(f"Volumes configuration file not found at {self.config_path}")
    
    def _save_volumes(self) -> None:
        """Save volumes to configuration file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        volumes_data = {
            "volumes": [volume.to_dict() for volume in self.volumes.values()]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(volumes_data, f, indent=2)
        
        logger.info(f"Saved {len(self.volumes)} volumes to {self.config_path}")
    
    def get_volume(self, name: str) -> Optional[Volume]:
        """Get a volume by name."""
        return self.volumes.get(name)
    
    def list_volumes(self, pool_name: Optional[str] = None) -> List[Volume]:
        """List all volumes, optionally filtered by pool."""
        if pool_name:
            return [v for v in self.volumes.values() if v.pool_name == pool_name]
        return list(self.volumes.values())
    
    def create_volume(
        self,
        name: str,
        pool_name: str,
        size: Optional[str] = None,
        mountpoint: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        type: Union[VolumeType, str] = VolumeType.FILESYSTEM,
        share_config: Optional[Dict] = None
    ) -> Volume:
        """Create a new volume in a storage pool.
        
        Args:
            name: The name of the volume
            pool_name: The name of the parent storage pool
            size: Size limit (e.g. "10G", "1T")
            mountpoint: Where to mount the volume
            options: Additional options for the volume
            type: Type of volume (filesystem, block, share)
            share_config: Configuration for network shares
            
        Returns:
            The created Volume
            
        Raises:
            ValueError: If the volume name already exists or pool doesn't exist
            CommandError: If there's an error creating the volume
        """
        # Check if volume already exists
        if name in self.volumes:
            raise ValueError(f"Volume '{name}' already exists")
        
        # Check if pool exists
        pool = self.pool_manager.get_pool(pool_name)
        if not pool:
            raise ValueError(f"Storage pool '{pool_name}' doesn't exist")
        
        # Convert type to enum if needed
        if isinstance(type, str):
            type = VolumeType.from_string(type)
        
        # Create path based on pool/volume name
        path = f"{pool_name}/{name}"
        
        # Prepare mountpoint
        mountpoint = mountpoint or f"/mnt/{path}"
        
        # Create the volume based on pool's filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            self._create_zfs_volume(pool_name, name, size, mountpoint, options or {})
        elif pool.fs_type == FilesystemType.BTRFS:
            self._create_btrfs_volume(pool_name, name, pool.mountpoint, mountpoint, options or {})
        elif pool.fs_type == FilesystemType.LVM:
            self._create_lvm_volume(pool_name, name, size, mountpoint, options or {})
        else:
            raise ValueError(f"Unsupported filesystem type: {pool.fs_type}")
        
        # Configure as share if needed
        if type == VolumeType.SHARE and share_config:
            self._configure_share(name, mountpoint, share_config)
        
        # Create and register the volume
        volume = Volume(
            name=name,
            pool_name=pool_name,
            path=path,
            type=type,
            size=size,
            mountpoint=mountpoint,
            options=options,
            share_config=share_config
        )
        
        self.volumes[name] = volume
        self._save_volumes()
        
        logger.info(f"Created volume: {volume}")
        return volume
    
    def delete_volume(self, name: str, destroy_data: bool = False) -> bool:
        """Delete a volume.
        
        Args:
            name: The name of the volume to delete
            destroy_data: Whether to destroy data or just unmount
            
        Returns:
            True if the volume was deleted, False otherwise
            
        Raises:
            CommandError: If there's an error deleting the volume
        """
        volume = self.get_volume(name)
        if not volume:
            logger.warning(f"Cannot delete non-existent volume: {name}")
            return False
        
        # Get the pool
        pool = self.pool_manager.get_pool(volume.pool_name)
        if not pool:
            logger.warning(f"Pool '{volume.pool_name}' for volume '{name}' doesn't exist")
            return False
        
        # Remove share configuration if it's a share
        if volume.type == VolumeType.SHARE and volume.share_config:
            self._remove_share(name, volume.share_config)
        
        # Delete based on filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            self._delete_zfs_volume(volume.pool_name, volume.name, destroy_data)
        elif pool.fs_type == FilesystemType.BTRFS:
            self._delete_btrfs_volume(volume.mountpoint, destroy_data)
        elif pool.fs_type == FilesystemType.LVM:
            self._delete_lvm_volume(volume.pool_name, volume.name, destroy_data)
        
        # Remove from registry
        del self.volumes[name]
        self._save_volumes()
        
        logger.info(f"Deleted volume: {name}")
        return True
    
    def create_snapshot(self, volume_name: str, snapshot_name: Optional[str] = None) -> Dict:
        """Create a snapshot of a volume.
        
        Args:
            volume_name: The name of the volume to snapshot
            snapshot_name: Optional name for the snapshot (default: auto-generated)
            
        Returns:
            Dict with snapshot details
            
        Raises:
            ValueError: If the volume doesn't exist
            CommandError: If there's an error creating the snapshot
        """
        volume = self.get_volume(volume_name)
        if not volume:
            raise ValueError(f"Volume '{volume_name}' doesn't exist")
        
        # Get the pool
        pool = self.pool_manager.get_pool(volume.pool_name)
        if not pool:
            raise ValueError(f"Pool '{volume.pool_name}' doesn't exist")
        
        # Generate snapshot name if not provided
        if snapshot_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            snapshot_name = f"{volume_name}@{timestamp}"
        
        # Create snapshot based on filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            self._create_zfs_snapshot(volume.pool_name, volume.name, snapshot_name)
        elif pool.fs_type == FilesystemType.BTRFS:
            self._create_btrfs_snapshot(volume.mountpoint, snapshot_name)
        elif pool.fs_type == FilesystemType.LVM:
            self._create_lvm_snapshot(volume.pool_name, volume.name, snapshot_name)
        
        # Add to volume's snapshots list
        volume.snapshots.append(snapshot_name)
        self._save_volumes()
        
        logger.info(f"Created snapshot '{snapshot_name}' of volume '{volume_name}'")
        return {
            "name": snapshot_name,
            "volume": volume_name,
            "created_at": datetime.now().isoformat()
        }
    
    def delete_snapshot(self, volume_name: str, snapshot_name: str) -> bool:
        """Delete a snapshot.
        
        Args:
            volume_name: The name of the volume
            snapshot_name: The name of the snapshot to delete
            
        Returns:
            True if the snapshot was deleted, False otherwise
            
        Raises:
            ValueError: If the volume doesn't exist
            CommandError: If there's an error deleting the snapshot
        """
        volume = self.get_volume(volume_name)
        if not volume:
            raise ValueError(f"Volume '{volume_name}' doesn't exist")
        
        # Check if snapshot exists
        if snapshot_name not in volume.snapshots:
            logger.warning(f"Snapshot '{snapshot_name}' not found for volume '{volume_name}'")
            return False
        
        # Get the pool
        pool = self.pool_manager.get_pool(volume.pool_name)
        if not pool:
            raise ValueError(f"Pool '{volume.pool_name}' doesn't exist")
            
        # Delete the snapshot based on filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            self._delete_zfs_snapshot(volume.pool_name, volume.name, snapshot_name)
        elif pool.fs_type == FilesystemType.BTRFS:
            self._delete_btrfs_snapshot(volume.mountpoint, snapshot_name)
        elif pool.fs_type == FilesystemType.LVM:
            self._delete_lvm_snapshot(volume.pool_name, volume.name, snapshot_name)
            
        # Remove from volume's snapshots list
        if snapshot_name in volume.snapshots:
            volume.snapshots.remove(snapshot_name)
            self._save_volumes()
            
        logger.info(f"Deleted snapshot '{snapshot_name}' of volume '{volume_name}'")
        return True
    
    def set_quota(self, volume_name: str, quota_size: str) -> bool:
        """Set a quota on a volume.
        
        Args:
            volume_name: The name of the volume
            quota_size: The quota size (e.g. "10G", "1T")
            
        Returns:
            True if the quota was set, False otherwise
            
        Raises:
            ValueError: If the volume doesn't exist
            CommandError: If there's an error setting the quota
        """
        volume = self.get_volume(volume_name)
        if not volume:
            raise ValueError(f"Volume '{volume_name}' doesn't exist")
            
        # Get the pool
        pool = self.pool_manager.get_pool(volume.pool_name)
        if not pool:
            raise ValueError(f"Pool '{volume.pool_name}' doesn't exist")
            
        # Set quota based on filesystem type
        if pool.fs_type == FilesystemType.ZFS:
            self._set_zfs_quota(volume.pool_name, volume.name, quota_size)
        elif pool.fs_type == FilesystemType.BTRFS:
            self._set_btrfs_quota(volume.mountpoint, quota_size)
        elif pool.fs_type == FilesystemType.LVM:
            self._set_lvm_quota(volume.pool_name, volume.name, quota_size)
            
        # Update volume size
        volume.size = quota_size
        self._save_volumes()
        
        logger.info(f"Set quota of {quota_size} on volume '{volume_name}'")
        return True
        
    def configure_share(
        self,
        volume_name: str,
        protocol: Union[ShareProtocol, str],
        share_options: Optional[Dict[str, str]] = None
    ) -> bool:
        """Configure a volume as a network share.
        
        Args:
            volume_name: The name of the volume
            protocol: The sharing protocol (SMB, NFS, etc.)
            share_options: Protocol-specific sharing options
            
        Returns:
            True if the share was configured, False otherwise
            
        Raises:
            ValueError: If the volume doesn't exist
            CommandError: If there's an error configuring the share
        """
        volume = self.get_volume(volume_name)
        if not volume:
            raise ValueError(f"Volume '{volume_name}' doesn't exist")
            
        # Convert protocol to enum if needed
        if isinstance(protocol, str):
            protocol = ShareProtocol.from_string(protocol)
            
        # Prepare share config
        share_config = {
            "protocol": protocol.value,
            "options": share_options or {}
        }
        
        # Configure the share
        self._configure_share(volume_name, volume.mountpoint, share_config)
        
        # Update volume type and share config
        volume.type = VolumeType.SHARE
        volume.share_config = share_config
        self._save_volumes()
        
        logger.info(f"Configured volume '{volume_name}' as {protocol.value} share")
        return True
    
    # ZFS-specific methods
    def _create_zfs_volume(self, pool_name: str, volume_name: str, size: Optional[str], mountpoint: str, options: Dict[str, str]) -> None:
        """Create a ZFS dataset/volume."""
        # Build options string
        opts = [f"-o mountpoint={mountpoint}"]
        
        if size:
            opts.append(f"-o quota={size}")
            
        for key, value in options.items():
            opts.append(f"-o {key}={value}")
            
        opts_str = " ".join(opts)
        
        # Create the dataset
        cmd = f"zfs create {opts_str} {pool_name}/{volume_name}"
        run_command(cmd)
        
    def _delete_zfs_volume(self, pool_name: str, volume_name: str, destroy_data: bool) -> None:
        """Delete a ZFS dataset/volume."""
        if destroy_data:
            cmd = f"zfs destroy -r {pool_name}/{volume_name}"
            run_command(cmd)
        else:
            # Just unmount, don't destroy data
            cmd = f"zfs unmount {pool_name}/{volume_name}"
            run_command(cmd)
            
    def _create_zfs_snapshot(self, pool_name: str, volume_name: str, snapshot_name: str) -> None:
        """Create a ZFS snapshot."""
        cmd = f"zfs snapshot {pool_name}/{volume_name}@{snapshot_name}"
        run_command(cmd)
        
    def _delete_zfs_snapshot(self, pool_name: str, volume_name: str, snapshot_name: str) -> None:
        """Delete a ZFS snapshot."""
        cmd = f"zfs destroy {pool_name}/{volume_name}@{snapshot_name}"
        run_command(cmd)
        
    def _set_zfs_quota(self, pool_name: str, volume_name: str, quota_size: str) -> None:
        """Set a quota on a ZFS dataset/volume."""
        cmd = f"zfs set quota={quota_size} {pool_name}/{volume_name}"
        run_command(cmd)
        
    # BTRFS-specific methods
    def _create_btrfs_volume(self, pool_name: str, volume_name: str, pool_mountpoint: str, mountpoint: str, options: Dict[str, str]) -> None:
        """Create a BTRFS subvolume."""
        # Create the subvolume
        cmd = f"btrfs subvolume create {pool_mountpoint}/{volume_name}"
        run_command(cmd)
        
        # Create mountpoint
        os.makedirs(mountpoint, exist_ok=True)
        
        # Add to fstab for mounting
        fstab_entry = f"{pool_mountpoint}/{volume_name} {mountpoint} btrfs subvol={volume_name},defaults 0 0"
        with open("/etc/fstab", "a") as f:
            f.write(f"\n{fstab_entry}\n")
            
        # Mount
        cmd = f"mount {mountpoint}"
        run_command(cmd)
        
    def _delete_btrfs_volume(self, mountpoint: str, destroy_data: bool) -> None:
        """Delete a BTRFS subvolume."""
        # Unmount
        cmd = f"umount {mountpoint}"
        run_command(cmd)
        
        # Remove from fstab
        with open("/etc/fstab", "r") as f:
            fstab_lines = f.readlines()
            
        with open("/etc/fstab", "w") as f:
            for line in fstab_lines:
                if mountpoint not in line:
                    f.write(line)
                    
        if destroy_data:
            # Delete the subvolume
            cmd = f"btrfs subvolume delete {mountpoint}"
            run_command(cmd)
            
    def _create_btrfs_snapshot(self, mountpoint: str, snapshot_name: str) -> None:
        """Create a BTRFS snapshot."""
        # Extract parent directory
        parent_dir = os.path.dirname(mountpoint)
        
        # Create snapshot directory
        snapshot_dir = f"{parent_dir}/snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Create snapshot
        cmd = f"btrfs subvolume snapshot -r {mountpoint} {snapshot_dir}/{snapshot_name}"
        run_command(cmd)
        
    def _delete_btrfs_snapshot(self, mountpoint: str, snapshot_name: str) -> None:
        """Delete a BTRFS snapshot."""
        # Extract parent directory
        parent_dir = os.path.dirname(mountpoint)
        
        # Create snapshot path
        snapshot_path = f"{parent_dir}/snapshots/{snapshot_name}"
        
        # Delete snapshot
        cmd = f"btrfs subvolume delete {snapshot_path}"
        run_command(cmd)
        
    def _set_btrfs_quota(self, mountpoint: str, quota_size: str) -> None:
        """Set a quota on a BTRFS subvolume."""
        # Enable quotas on the filesystem
        cmd = f"btrfs quota enable {mountpoint}"
        run_command(cmd)
        
        # Set quota limit
        cmd = f"btrfs qgroup limit {quota_size} 0/5 {mountpoint}"
        run_command(cmd)
        
    # LVM-specific methods
    def _create_lvm_volume(self, pool_name: str, volume_name: str, size: Optional[str], mountpoint: str, options: Dict[str, str]) -> None:
        """Create an LVM logical volume."""
        # Create the logical volume
        size_arg = f"--size {size}" if size else "--extents 100%FREE"
        cmd = f"lvcreate {size_arg} --name {volume_name} {pool_name}"
        run_command(cmd)
        
        # Create filesystem
        fs_type = options.get("fs_type", "ext4")
        cmd = f"mkfs.{fs_type} /dev/{pool_name}/{volume_name}"
        run_command(cmd)
        
        # Create mountpoint
        os.makedirs(mountpoint, exist_ok=True)
        
        # Add to fstab for mounting
        fstab_entry = f"/dev/{pool_name}/{volume_name} {mountpoint} {fs_type} defaults 0 0"
        with open("/etc/fstab", "a") as f:
            f.write(f"\n{fstab_entry}\n")
            
        # Mount
        cmd = f"mount {mountpoint}"
        run_command(cmd)
        
    def _delete_lvm_volume(self, pool_name: str, volume_name: str, destroy_data: bool) -> None:
        """Delete an LVM logical volume."""
        # Unmount
        cmd = f"umount /dev/{pool_name}/{volume_name}"
        run_command(cmd)
        
        # Remove from fstab
        with open("/etc/fstab", "r") as f:
            fstab_lines = f.readlines()
            
        with open("/etc/fstab", "w") as f:
            for line in fstab_lines:
                if f"/dev/{pool_name}/{volume_name}" not in line:
                    f.write(line)
                    
        if destroy_data:
            # Delete the logical volume
            cmd = f"lvremove -f {pool_name}/{volume_name}"
            run_command(cmd)
            
    def _create_lvm_snapshot(self, pool_name: str, volume_name: str, snapshot_name: str) -> None:
        """Create an LVM snapshot."""
        cmd = f"lvcreate --snapshot --name {snapshot_name} {pool_name}/{volume_name}"
        run_command(cmd)
        
    def _delete_lvm_snapshot(self, pool_name: str, volume_name: str, snapshot_name: str) -> None:
        """Delete an LVM snapshot."""
        cmd = f"lvremove -f {pool_name}/{snapshot_name}"
        run_command(cmd)
        
    def _configure_share(self, volume_name: str, mountpoint: str, share_config: Dict) -> None:
        """Configure a network share."""
        protocol = share_config.get("protocol", "smb")
        options = share_config.get("options", {})
        
        if protocol == ShareProtocol.SMB.value:
            self._configure_smb_share(volume_name, mountpoint, options)
        elif protocol == ShareProtocol.NFS.value:
            self._configure_nfs_share(volume_name, mountpoint, options)
        elif protocol == ShareProtocol.AFP.value:
            self._configure_afp_share(volume_name, mountpoint, options)
        elif protocol == ShareProtocol.WEBDAV.value:
            self._configure_webdav_share(volume_name, mountpoint, options)
        else:
            raise ValueError(f"Unsupported share protocol: {protocol}")
    
    def _remove_share(self, volume_name: str, share_config: Dict) -> None:
        """Remove a network share."""
        protocol = share_config.get("protocol", "smb")
        
        if protocol == ShareProtocol.SMB.value:
            self._remove_smb_share(volume_name)
        elif protocol == ShareProtocol.NFS.value:
            self._remove_nfs_share(volume_name)
        elif protocol == ShareProtocol.AFP.value:
            self._remove_afp_share(volume_name)
        elif protocol == ShareProtocol.WEBDAV.value:
            self._remove_webdav_share(volume_name)
        else:
            raise ValueError(f"Unsupported share protocol: {protocol}")
    
    def _configure_smb_share(self, volume_name: str, mountpoint: str, options: Dict[str, str]) -> None:
        """Configure an SMB share."""
        # Create smb config if it doesn't exist
        smb_conf = "/etc/samba/smb.conf"
        os.makedirs(os.path.dirname(smb_conf), exist_ok=True)
        
        # Default options
        default_options = {
            "comment": f"Prometheum share - {volume_name}",
            "path": mountpoint,
            "browseable": "yes",
            "read only": "no",
            "guest ok": "no",
            "create mask": "0644",
            "directory mask": "0755"
        }
        
        # Merge with user-provided options
        share_options = {**default_options, **options}
        
        # Generate config
        config_lines = [f"[{volume_name}]"]
        for key, value in share_options.items():
            config_lines.append(f"    {key} = {value}")
        
        # Add to config file
        with open(smb_conf, "r") as f:
            smb_lines = f.readlines()
        
        # Check if share already exists
        share_exists = False
        share_header = f"[{volume_name}]"
        for i, line in enumerate(smb_lines):
            if line.strip() == share_header:
                share_exists = True
                break
        
        if not share_exists:
            with open(smb_conf, "a") as f:
                f.write("\n" + "\n".join(config_lines) + "\n")
        
        # Restart Samba service
        cmd = "systemctl restart smbd nmbd"
        run_command(cmd)
    
    def _remove_smb_share(self, volume_name: str) -> None:
        """Remove an SMB share."""
        smb_conf = "/etc/samba/smb.conf"
        
        if not os.path.exists(smb_conf):
            return
        
        # Read config
        with open(smb_conf, "r") as f:
            smb_lines = f.readlines()
        
        # Find share section
        new_lines = []
        skip_section = False
        share_header = f"[{volume_name}]"
        
        for line in smb_lines:
            if line.strip() == share_header:
                skip_section = True
                continue
            
            if skip_section and line.startswith("["):
                skip_section = False
            
            if not skip_section:
                new_lines.append(line)
        
        # Write updated config
        with open(smb_conf, "w") as f:
            f.writelines(new_lines)
        
        # Restart Samba service
        cmd = "systemctl restart smbd nmbd"
        run_command(cmd)
    
    def _configure_nfs_share(self, volume_name: str, mountpoint: str, options: Dict[str, str]) -> None:
        """Configure an NFS share."""
        # Default options
        default_options = {
            "rw": None,
            "sync": None,
            "no_subtree_check": None,
            "root_squash": None
        }
        
        # Merge with user-provided options
        share_options = {**default_options, **options}
        
        # Generate exports line
        exports_line = f"{mountpoint} "
        
        # Add client specs
        clients = options.get("clients", "*")
        exports_line += clients
        
        # Add options
        option_parts = []
        for key, value in share_options.items():
            if key != "clients":
                if value is None:
                    option_parts.append(key)
                else:
                    option_parts.append(f"{key}={value}")
        
        if option_parts:
            exports_line += "(" + ",".join(option_parts) + ")"
        
        # Add to exports file
        exports_file = "/etc/exports"
        
        with open(exports_file, "r") as f:
            exports_lines = f.readlines()
        
        # Check if mountpoint already exists
        mountpoint_exists = False
        for i, line in enumerate(exports_lines):
            if line.startswith(mountpoint):
                exports_lines[i] = exports_line + "\n"
                mountpoint_exists = True
                break
        
        if not mountpoint_exists:
            exports_lines.append(exports_line + "\n")
        
        with open(exports_file, "w") as f:
            f.writelines(exports_lines)
        
        # Restart NFS service
        cmd = "exportfs -ra"
        run_command(cmd)
    
    def _remove_nfs_share(self, volume_name: str) -> None:
        """Remove an NFS share."""
        # Get volume
        volume = self.get_volume(volume_name)
        if not volume:
            return
        
        # Remove from exports file
        exports_file = "/etc/exports"
        
        if not os.path.exists(exports_file):
            return
        
        with open(exports_file, "r") as f:
            exports_lines = f.readlines()
        
        new_lines = []
        for line in exports_lines:
            if not line.startswith(volume.mountpoint):
                new_lines.append(line)
        
        with open(exports_file, "w") as f:
            f.writelines(new_lines)
        
        # Restart NFS service
        cmd = "exportfs -ra"
        run_command(cmd)
    
    def _configure_afp_share(self, volume_name: str, mountpoint: str, options: Dict[str, str]) -> None:
        """Configure an AFP share."""
        # Note: AFP is deprecated, implementation is simplified
        logger.warning("AFP sharing is deprecated, consider using SMB instead")
        
        # Default options
        default_options = {
            "path": mountpoint,
            "valid users": "@users",
            "time machine": "no"
        }
        
        # Merge with user-provided options
        share_options = {**default_options, **options}
        
        # Generate config
        config_lines = [f"[{volume_name}]"]
        for key, value in share_options.items():
            config_lines.append(f"{key} = {value}")
        
        # Write config to netatalk configuration
        afp_conf = "/etc/netatalk/afp.conf"
        os.makedirs(os.path.dirname(afp_conf), exist_ok=True)
        
        with open(afp_conf, "a") as f:
            f.write("\n" + "\n".join(config_lines) + "\n")
        
        # Restart netatalk service
        cmd = "systemctl restart netatalk"
        run_command(cmd)
    
    def _remove_afp_share(self, volume_name: str) -> None:
        """Remove an AFP share."""
        afp_conf = "/etc/netatalk/afp.conf"
        
        if not os.path.exists(afp_conf):
            return
        
        # Read config
        with open(afp_conf, "r") as f:
            afp_lines = f.readlines()
        
        # Find share section
        new_lines = []
        skip_section = False
        share_header = f"[{volume_name}]"
        
        for line in afp_lines:
            if line.strip() == share_header:
                skip_section = True
                continue
            
            if skip_section and line.startswith("["):
                skip_section = False
            
            if not skip_section:
                new_lines.append(line)
        
        # Write updated config
        with open(afp_conf, "w") as f:
            f.writelines(new_lines)
        
        # Restart netatalk service
        cmd = "systemctl restart netatalk"
        run_command(cmd)
    
    def _configure_webdav_share(self, volume_name: str, mountpoint: str, options: Dict[str, str]) -> None:
        """Configure a WebDAV share."""
        # Default options
        default_options = {
            "alias": f"/{volume_name}",
            "auth_type": "digest"
        }
        
        # Merge with user-provided options
        share_options = {**default_options, **options}
        
        # Generate Apache config
        config_lines = [
            f"Alias {share_options.get('alias')} {mountpoint}",
            f"<Location {share_options.get('alias')}>",
            "    DAV On",
            f"    AuthType {share_options.get('auth_type')}",
            "    AuthName \"Prometheum WebDAV\"",
            "    AuthUserFile /etc/apache2/webdav.passwd",
            "    Require valid-user",
            f"    <Directory {mountpoint}>",
            "        Options Indexes",
            "        AllowOverride None",
            "        Require valid-user",
            "    </Directory>",
            "</Location>"
        ]
        
        # Write config to apache configuration
        webdav_conf = "/etc/apache2/conf-available/webdav.conf"
        os.makedirs(os.path.dirname(webdav_conf), exist_ok=True)
        
        with open(webdav_conf, "a") as f:
            f.write("\n" + "\n".join(config_lines) + "\n")
        
        # Enable webdav module and configuration
        cmd = "a2enmod dav dav_fs && a2enconf webdav && systemctl restart apache2"
        run_command(cmd)
    
    def _remove_webdav_share(self, volume_name: str) -> None:
        """Remove a WebDAV share."""
        webdav_conf = "/etc/apache2/conf-available/webdav.conf"
        
        if not os.path.exists(webdav_conf):
            return
        
        # Read config
        with open(webdav_conf, "r") as f:
            webdav_lines = f.readlines()
        
        # Find share section by looking for the alias
        new_lines = []
        skip_section = False
        location_start = None
        
        for i, line in enumerate(webdav_lines):
            if f"/{volume_name}" in line and "Alias" in line:
                skip_section = True
                continue
            
            if skip_section and "</Location>" in line:
                skip_section = False
                continue
            
            if not skip_section:
                new_lines.append(line)
        
        # Write updated config
        with open(webdav_conf, "w") as f:
            f.writelines(new_lines)
        
        # Restart Apache service
        
    # Share configuration methods
    def _configure_share(self, volume_name: str, mountpoint: str, share_config: Dict) -> None:
        """Configure a network share."""
        protocol = share_config.get("
