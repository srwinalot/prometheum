from typing import Dict, List, Optional
import os
import subprocess
from dataclasses import dataclass
from enum import Enum

class StorageDeviceType(Enum):
    DISK = "disk"
    PARTITION = "partition"
    LVM = "lvm"
    RAID = "raid"

@dataclass
class StorageDevice:
    name: str
    device_type: StorageDeviceType
    size: int  # Size in bytes
    model: str
    mount_point: Optional[str] = None
    filesystem: Optional[str] = None
    used_space: Optional[int] = None
    raid_level: Optional[int] = None
    raid_devices: Optional[List[str]] = None

class StorageManager:
    def __init__(self):
        self.devices: Dict[str, StorageDevice] = {}
        self.refresh_devices()

    def refresh_devices(self) -> None:
        """Scan and update the list of storage devices."""
        # Clear existing devices
        self.devices.clear()
        
        # Scan physical disks
        self._scan_physical_disks()
        
        # Scan RAID arrays
        self._scan_raid_arrays()
        
        # Scan LVM volumes
        self._scan_lvm_volumes()

    def _scan_physical_disks(self) -> None:
        """Scan for physical disks using lsblk."""
        try:
            result = subprocess.run(
                ["lsblk", "-bPo", "NAME,TYPE,SIZE,MODEL,MOUNTPOINT,FSTYPE"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.splitlines():
                if not line:
                    continue
                    
                info = dict(item.split("=", 1) for item in line.strip().split(" "))
                
                # Remove quotes from values
                info = {k: v.strip('"') for k, v in info.items()}
                
                if info["TYPE"] in ["disk", "part"]:
                    device = StorageDevice(
                        name=info["NAME"],
                        device_type=StorageDeviceType.DISK if info["TYPE"] == "disk" 
                                  else StorageDeviceType.PARTITION,
                        size=int(info["SIZE"]),
                        model=info.get("MODEL", ""),
                        mount_point=info.get("MOUNTPOINT"),
                        filesystem=info.get("FSTYPE")
                    )
                    self.devices[device.name] = device

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to scan physical disks: {e}")

    def _scan_raid_arrays(self) -> None:
        """Scan for RAID arrays using mdadm."""
        try:
            result = subprocess.run(
                ["mdadm", "--detail", "--scan"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("ARRAY"):
                        parts = line.split()
                        device_name = parts[1]
                        
                        # Get detailed info
                        detail = subprocess.run(
                            ["mdadm", "--detail", device_name],
                            capture_output=True,
                            text=True
                        )
                        
                        if detail.returncode == 0:
                            info = self._parse_mdadm_detail(detail.stdout)
                            
                            device = StorageDevice(
                                name=device_name,
                                device_type=StorageDeviceType.RAID,
                                size=info["size"],
                                model=f"RAID-{info['level']}",
                                raid_level=info["level"],
                                raid_devices=info["devices"]
                            )
                            self.devices[device_name] = device

        except subprocess.CalledProcessError:
            # RAID tools might not be installed
            pass

    def _scan_lvm_volumes(self) -> None:
        """Scan for LVM volumes."""
        try:
            result = subprocess.run(
                ["lvs", "--noheadings", "--units", "b", "-o", "lv_path,lv_size"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if not line.strip():
                        continue
                        
                    path, size = line.strip().split()
                    size = int(size.rstrip("B"))
                    
                    device = StorageDevice(
                        name=path,
                        device_type=StorageDeviceType.LVM,
                        size=size,
                        model="LVM Volume"
                    )
                    self.devices[path] = device

        except subprocess.CalledProcessError:
            # LVM tools might not be installed
            pass

    def _parse_mdadm_detail(self, detail_output: str) -> Dict:
        """Parse mdadm --detail output."""
        info = {
            "level": 0,
            "size": 0,
            "devices": []
        }
        
        for line in detail_output.splitlines():
            line = line.strip()
            if "Raid Level" in line:
                info["level"] = int(line.split(":")[-1].strip().replace("raid", ""))
            elif "Array Size" in line:
                size_str = line.split(":")[-1].strip().split()[0]
                info["size"] = int(size_str) * 1024  # Convert KB to bytes
            elif "active sync" in line:
                device = line.split()[-1]
                info["devices"].append(device)
                
        return info

    def mount_device(self, device_name: str, mount_point: str) -> None:
        """Mount a storage device."""
        device = self.devices.get(device_name)
        if not device:
            raise ValueError(f"Device not found: {device_name}")
            
        if not device.filesystem:
            raise ValueError(f"No filesystem on device: {device_name}")
            
        if not os.path.exists(mount_point):
            os.makedirs(mount_point)
            
        try:
            subprocess.run(
                ["mount", device_name, mount_point],
                check=True
            )
            device.mount_point = mount_point
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to mount {device_name}: {e}")

    def unmount_device(self, device_name: str) -> None:
        """Unmount a storage device."""
        device = self.devices.get(device_name)
        if not device:
            raise ValueError(f"Device not found: {device_name}")
            
        if not device.mount_point:
            raise ValueError(f"Device not mounted: {device_name}")
            
        try:
            subprocess.run(
                ["umount", device_name],
                check=True
            )
            device.mount_point = None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to unmount {device_name}: {e}")

    def format_device(self, device_name: str, filesystem: str = "ext4") -> None:
        """Format a storage device."""
        device = self.devices.get(device_name)
        if not device:
            raise ValueError(f"Device not found: {device_name}")
            
        if device.mount_point:
            raise ValueError(f"Device is mounted: {device_name}")
            
        try:
            subprocess.run(
                ["mkfs", "-t", filesystem, device_name],
                check=True
            )
            device.filesystem = filesystem
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to format {device_name}: {e}")

    def create_raid(self, devices: List[str], level: int, name: str) -> None:
        """Create a new RAID array."""
        # Verify devices exist
        for device_name in devices:
            if device_name not in self.devices:
                raise ValueError(f"Device not found: {device_name}")
                
        # Create RAID array
        try:
            cmd = ["mdadm", "--create", name, f"--level={level}",
                  f"--raid-devices={len(devices)}"] + devices
            subprocess.run(cmd, check=True)
            
            # Refresh devices to include new RAID array
            self.refresh_devices()
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create RAID array: {e}")

    def create_lvm(self, devices: List[str], volume_name: str, size: str) -> None:
        """Create an LVM volume."""
        # Create volume group
        vg_name = f"vg_{volume_name}"
        try:
            subprocess.run(
                ["vgcreate", vg_name] + devices,
                check=True
            )
            
            # Create logical volume
            subprocess.run(
                ["lvcreate", "-L", size, "-n", volume_name, vg_name],
                check=True
            )
            
            # Refresh devices to include new LVM volume
            self.refresh_devices()
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create LVM volume: {e}")

