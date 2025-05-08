import pytest
import os
from prometheum.storage.manager import StorageManager

@pytest.mark.skipif(not os.getenv("RUN_SYSTEM_TESTS"), 
                    reason="System tests not enabled")
class TestStorageSystem:
    """System tests for storage management.
    
    These tests require actual hardware and root privileges.
    Run with RUN_SYSTEM_TESTS=1 pytest tests/system/
    """
    
    @pytest.fixture
    def storage_manager(self):
        return StorageManager()
    
    def test_device_detection(self, storage_manager):
        """Test physical device detection."""
        storage_manager.refresh_devices()
        assert len(storage_manager.devices) > 0
        
        for device in storage_manager.devices.values():
            assert device.name.startswith("/dev/")
            assert device.size > 0
    
    def test_mount_operations(self, storage_manager):
        """Test mounting operations on real devices."""
        # WARNING: This test requires a formatted device
        # and will mount/unmount it
        test_device = None
        test_mount = "/mnt/test"
        
        # Find a suitable device
        for device in storage_manager.devices.values():
            if device.filesystem and not device.mount_point:
                test_device = device
                break
                
        if not test_device:
            pytest.skip("No suitable device found for mount testing")
            
        try:
            # Create mount point
            os.makedirs(test_mount, exist_ok=True)
            
            # Test mount
            storage_manager.mount_device(test_device.name, test_mount)
            assert os.path.ismount(test_mount)
            
            # Test unmount
            storage_manager.unmount_device(test_device.name)
            assert not os.path.ismount(test_mount)
            
        finally:
            # Cleanup
            if os.path.exists(test_mount):
                if os.path.ismount(test_mount):
                    os.system(f"umount {test_mount}")
                os.rmdir(test_mount)

