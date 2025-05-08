import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
from datetime import datetime

from prometheum.api.main import app
from prometheum.backup.manager import BackupManager, BackupJob, BackupType

@pytest.fixture
def mock_backup_manager():
    mock_manager = MagicMock(spec=BackupManager)
    mock_manager.jobs = {
        "test_backup": BackupJob(
            name="test_backup",
            source="/test/source",
            destination="/test/destination",
            backup_type=BackupType.FULL,
            schedule="0 0 * * *",
            retention_days=7,
            exclude_patterns=["*.tmp"],
            last_backup=datetime.now(),
            enabled=True
        ),
        "incremental_backup": BackupJob(
            name="incremental_backup",
            source="/data/source",
            destination="/backup/destination",
            backup_type=BackupType.INCREMENTAL,
            schedule="0 * * * *",
            retention_days=30,
            exclude_patterns=None,
            enabled=True
        )
    }
    return mock_manager

@pytest.fixture
def api_client():
    return TestClient(app)

def test_list_jobs(api_client, mock_backup_manager):
    """Test listing backup jobs."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        response = api_client.get("/api/backup/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 2
        
        # Verify job information
        job = next(j for j in data["jobs"] if j["name"] == "test_backup")
        assert job["source"] == "/test/source"
        assert job["destination"] == "/test/destination"
        assert job["backup_type"] == "full"
        assert job["schedule"] == "0 0 * * *"
        assert job["retention_days"] == 7
        assert job["exclude_patterns"] == ["*.tmp"]
        assert job["enabled"] is True

def test_create_job(api_client, mock_backup_manager):
    """Test creating a new backup job."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        new_job = {
            "name": "new_backup",
            "source": "/new/source",
            "destination": "/new/destination",
            "backup_type": "differential",
            "schedule": "0 12 * * *",
            "retention_days": 14,
            "exclude_patterns": ["*.log", "tmp/*"],
            "enabled": True
        }
        
        response = api_client.post("/api/backup/jobs", json=new_job)
        assert response.status_code == 200
        assert "Backup job created" in response.json()["message"]
        
        # Verify create_job was called with correct parameters
        mock_backup_manager.create_job.assert_called_once()
        called_job = mock_backup_manager.create_job.call_args[0][0]
        assert called_job.name == new_job["name"]
        assert called_job.source == new_job["source"]
        assert called_job.backup_type == BackupType.DIFFERENTIAL

def test_update_job(api_client, mock_backup_manager):
    """Test updating an existing backup job."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        updated_job = {
            "name": "test_backup",
            "source": "/updated/source",
            "destination": "/updated/destination",
            "backup_type": "incremental",
            "schedule": "0 6 * * *",
            "retention_days": 10,
            "exclude_patterns": ["*.bak"],
            "enabled": True
        }
        
        response = api_client.put("/api/backup/jobs/test_backup", json=updated_job)
        assert response.status_code == 200
        assert "Backup job updated" in response.json()["message"]
        
        # Verify update_job was called with correct parameters
        mock_backup_manager.update_job.assert_called_once()
        called_job = mock_backup_manager.update_job.call_args[0][0]
        assert called_job.source == updated_job["source"]
        assert called_job.backup_type == BackupType.INCREMENTAL

def test_delete_job(api_client, mock_backup_manager):
    """Test deleting a backup job."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        response = api_client.delete("/api/backup/jobs/test_backup")
        assert response.status_code == 200
        assert "Backup job deleted" in response.json()["message"]
        
        # Verify delete_job was called
        mock_backup_manager.delete_job.assert_called_once_with("test_backup")

def test_execute_job(api_client, mock_backup_manager):
    """Test executing a backup job."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        response = api_client.post("/api/backup/jobs/test_backup/execute")
        assert response.status_code == 200
        assert "Backup job executed" in response.json()["message"]
        
        # Verify execute_backup was called
        mock_backup_manager.execute_backup.assert_called_once_with("test_backup")

def test_error_handling(api_client, mock_backup_manager):
    """Test error handling for various scenarios."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        # Test creating duplicate job
        mock_backup_manager.create_job.side_effect = ValueError("Backup job already exists")
        response = api_client.post("/api/backup/jobs", json={
            "name": "test_backup",
            "source": "/test/source",
            "destination": "/test/destination",
            "backup_type": "full",
            "schedule": "0 0 * * *",
            "retention_days": 7
        })
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
        
        # Test updating non-existent job
        mock_backup_manager.update_job.side_effect = ValueError("Backup job not found")
        response = api_client.put("/api/backup/jobs/nonexistent", json={
            "name": "nonexistent",
            "source": "/test/source",
            "destination": "/test/destination",
            "backup_type": "full",
            "schedule": "0 0 * * *",
            "retention_days": 7
        })
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]
        
        # Test executing disabled job
        mock_backup_manager.execute_backup.side_effect = ValueError("Backup job is disabled")
        response = api_client.post("/api/backup/jobs/disabled_job/execute")
        assert response.status_code == 400
        assert "disabled" in response.json()["detail"]

def test_invalid_backup_type(api_client, mock_backup_manager):
    """Test handling of invalid backup type."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        response = api_client.post("/api/backup/jobs", json={
            "name": "invalid_backup",
            "source": "/test/source",
            "destination": "/test/destination",
            "backup_type": "invalid",
            "schedule": "0 0 * * *",
            "retention_days": 7
        })
        assert response.status_code == 400
        assert "backup_type" in response.json()["detail"].lower()

def test_invalid_schedule(api_client, mock_backup_manager):
    """Test handling of invalid schedule format."""
    with patch('prometheum.api.routes.backup.backup_manager', mock_backup_manager):
        response = api_client.post("/api/backup/jobs", json={
            "name": "invalid_schedule",
            "source": "/test/source",
            "destination": "/test/destination",
            "backup_type": "full",
            "schedule": "invalid",
            "retention_days": 7
        })
        assert response.status_code == 400

