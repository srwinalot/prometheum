import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from prometheum.api.main import app
from tests.mocks.storage import MockStorageManager

@pytest.fixture
def mock_storage_manager(mock_devices):
    with patch('prometheum.api.routes.storage.storage_manager', 
              MockStorageManager(mock_devices)):
        yield

@pytest.fixture
def api_client():
    return TestClient(app)

def test_list_devices(api_client, mock_storage_manager, mock_devices):
    """Test listing storage devices."""
    response = api_client.get("/api/storage/devices")
    assert response.status_code == 200
    
    data = response.json()
    assert "devices" in data
    assert len(data["devices"]) == len(mock_devices)
    
    # Verify device information
    for device in data["devices"]:
        assert any(device["name"] == mock_device["name"] for mock_device in mock_devices)
        found = next((m for m in mock_devices if m["name"] == device["name"]), None)
        assert device["device_type"] == found["device_type"]
        assert device["size"] == found["size"]
        assert device["model"] == found["model"]

def test_mount_device(api_client, mock_storage_manager, mock_devices):
    """Test mounting a storage device."""
    device_name = mock_devices[0]["name"]
    mount_point = "/mnt/test"
    
    # Format device first (this modifies the mock device to have a filesystem)
    response = api_client.post(f"/api/storage/format/{device_name}", params={"filesystem": "ext4"})
    assert response.status_code == 200
    
    # Mount device
    response = api_client.post(
        f"/api/storage/mount/{device_name}",
        params={"mount_point": mount_point}
    )
    assert response.status_code == 200
    assert "mounted" in response.json()["message"]

def test_create_raid(api_client, mock_storage_manager, mock_devices):
    """Test creating a RAID array."""
    devices = [device["name"] for device in mock_devices[:2]]
    
    response = api_client.post(
        "/api/storage/raid",
        json={
            "devices": devices,
            "level": 1,
            "name": "/dev/md0"
        }
    )
    assert response.status_code == 200
    assert "RAID array" in response.json()["message"]

def test_create_lvm(api_client, mock_storage_manager, mock_devices):
    """Test creating an LVM volume."""
    devices = [device["name"] for device in mock_devices[:2]]
    
    response = api_client.post(
        "/api/storage/lvm",
        json={
            "devices": devices,
            "volume_name": "test_vol",
            "size": "500M"
        }
    )
    assert response.status_code == 200
    assert "LVM volume" in response.json()["message"]

def test_error_handling(api_client, mock_storage_manager):
    """Test error handling for invalid operations."""
    # Try to mount non-existent device
    response = api_client.post(
        "/api/storage/mount/nonexistent",
        params={"mount_point": "/mnt/test"}
    )
    assert response.status_code == 400
    assert "Device not found" in response.json()["detail"]
    
    # Try to create RAID with non-existent devices
    response = api_client.post(
        "/api/storage/raid",
        json={
            "devices": ["/dev/nonexistent1", "/dev/nonexistent2"],
            "level": 1,
            "name": "/dev/md0"
        }
    )
    assert response.status_code == 400
    assert "Device not found" in response.json()["detail"]

