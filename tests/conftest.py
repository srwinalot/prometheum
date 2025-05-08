import pytest
from fastapi.testclient import TestClient
from prometheum.api.main import app

@pytest.fixture
def api_client():
    return TestClient(app)

@pytest.fixture
def mock_devices():
    return [
        {
            "name": "/dev/sda",
            "device_type": "disk",
            "size": 1000000000,
            "model": "Test Disk 1",
            "filesystem": None
        },
        {
            "name": "/dev/sdb",
            "device_type": "disk",
            "size": 1000000000,
            "model": "Test Disk 2",
            "filesystem": None
        }
    ]

