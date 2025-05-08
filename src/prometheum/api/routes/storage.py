from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from prometheum.storage.manager import StorageManager, StorageDevice, StorageDeviceType

router = APIRouter()
storage_manager = StorageManager()

class DeviceInfo(BaseModel):
    name: str
    device_type: str
    size: int
    model: str
    mount_point: Optional[str] = None
    filesystem: Optional[str] = None
    raid_level: Optional[int] = None
    raid_devices: Optional[List[str]] = None

@router.get("/devices")
async def list_devices():
    """List all storage devices."""
    storage_manager.refresh_devices()
    return {
        "devices": [
            DeviceInfo(
                name=device.name,
                device_type=device.device_type.value,
                size=device.size,
                model=device.model,
                mount_point=device.mount_point,
                filesystem=device.filesystem,
                raid_level=device.raid_level,
                raid_devices=device.raid_devices
            )
            for device in storage_manager.devices.values()
        ]
    }

@router.post("/mount/{device_name}")
async def mount_device(device_name: str, mount_point: str):
    """Mount a storage device."""
    try:
        storage_manager.mount_device(device_name, mount_point)
        return {"message": f"Device {device_name} mounted at {mount_point}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/unmount/{device_name}")
async def unmount_device(device_name: str):
    """Unmount a storage device."""
    try:
        storage_manager.unmount_device(device_name)
        return {"message": f"Device {device_name} unmounted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/format/{device_name}")
async def format_device(device_name: str, filesystem: str = "ext4"):
    """Format a storage device."""
    try:
        storage_manager.format_device(device_name, filesystem)
        return {"message": f"Device {device_name} formatted with {filesystem}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/raid")
async def create_raid(devices: List[str], level: int, name: str):
    """Create a RAID array."""
    try:
        storage_manager.create_raid(devices, level, name)
        return {"message": f"RAID array {name} created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/lvm")
async def create_lvm(devices: List[str], volume_name: str, size: str):
    """Create an LVM volume."""
    try:
        storage_manager.create_lvm(devices, volume_name, size)
        return {"message": f"LVM volume {volume_name} created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

