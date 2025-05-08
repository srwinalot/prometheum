from typing import Dict, List
from prometheum.storage.manager import StorageManager, StorageDevice, StorageDeviceType

class MockStorageManager(StorageManager):
    def __init__(self, mock_devices: List[Dict] = None):
        self.devices = {}
        if mock_devices:
            for device_data in mock_devices:
                device = StorageDevice(
                    name=device_data["name"],
                    device_type=StorageDeviceType(device_data["device_type"]),
                    size=device_data["size"],
                    model=device_data["model"],
                    filesystem=device_data.get("filesystem"),
                    mount_point=device_data.get("mount_point"),
                    raid_level=device_data.get("raid_level"),
                    raid_devices=device_data.get("raid_devices")
                )
                self.devices[device.name] = device

    def refresh_devices(self) -> None:
        """Override to prevent actual scanning in tests"""
        pass

    def _scan_physical_disks(self) -> None:
        """Mock scan - does nothing as devices are pre-populated"""
        pass

    def _scan_raid_arrays(self) -> None:
        """Mock scan - does nothing as devices are pre-populated"""
        pass

    def _scan_lvm_volumes(self) -> None:
        """Mock scan - does nothing as devices are pre-populated"""
        pass

