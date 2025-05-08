import pytest
import os
import shutil
import tempfile
from datetime import datetime, timedelta
import json
from typing import Generator, Tuple

from prometheum.backup.manager import BackupManager, BackupJob, BackupType

@pytest.fixture
def test_directories() -> Generator[Tuple[str, str], None, None]:
    """Create temporary source and destination directories for testing."""
    with tempfile.TemporaryDirectory() as source_dir, \
         tempfile.TemporaryDirectory() as dest_dir:
        
        # Create some test files in source directory
        test_files = [
            "file1.txt",
            "file2.dat",
            "subdir/file3.txt",
            "subdir/file4.log"
        ]
        
        for file_path in test_files:
            full_path = os.path.join(source_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(f"Test content for {file_path}")
        
        yield source_dir, dest_dir

@pytest.fixture
def backup_manager(test_directories) -> BackupManager:
    """Create a backup manager with a temporary config file."""
    source_dir, dest_dir = test_directories
    config_file = tempfile.mktemp(suffix='.json')
    
    manager = BackupManager(config_path=config_file)
    
    # Add a test job
    job = BackupJob(
        name="test_backup",
        source=source_dir,
        destination=dest_dir,
        backup_type=BackupType.FULL,
        schedule="0 0 * * *",
        retention_days=7,
        exclude_patterns=["*.log"],
        enabled=True
    )
    manager.create_job(job)
    
    yield manager
    
    # Cleanup
    if os.path.exists(config_file):
        os.unlink(config_file)

@pytest.mark.skipif(not os.getenv("RUN_SYSTEM_TESTS"), 
                    reason="System tests not enabled")
class TestBackupOperations:
    """System tests for backup operations."""
    
    def test_full_backup(self, backup_manager, test_directories):
        """Test performing a full backup."""
        source_dir, dest_dir = test_directories
        
        # Execute backup
        backup_manager.execute_backup("test_backup")
        
        # Verify backup was created
        backup_dirs = os.listdir(dest_dir)
        assert len(backup_dirs) == 1
        
        backup_path = os.path.join(dest_dir, backup_dirs[0])
        
        # Verify files were copied
        assert os.path.exists(os.path.join(backup_path, "file1.txt"))
        assert os.path.exists(os.path.join(backup_path, "file2.dat"))
        assert os.path.exists(os.path.join(backup_path, "subdir/file3.txt"))
        
        # Verify excluded files were not copied
        assert not os.path.exists(os.path.join(backup_path, "subdir/file4.log"))
        
        # Verify file contents
        with open(os.path.join(backup_path, "file1.txt")) as f:
            assert f.read() == "Test content for file1.txt"
    
    def test_incremental_backup(self, backup_manager, test_directories):
        """Test performing incremental backups."""
        source_dir, dest_dir = test_directories
        
        # Update job to incremental
        job = backup_manager.jobs["test_backup"]
        job.backup_type = BackupType.INCREMENTAL
        backup_manager.update_job(job)
        
        # Perform initial backup
        backup_manager.execute_backup("test_backup")
        
        # Modify some files and add new ones
        with open(os.path.join(source_dir, "file1.txt"), 'a') as f:
            f.write("\nModified content")
        
        with open(os.path.join(source_dir, "newfile.txt"), 'w') as f:
            f.write("New file content")
        
        # Perform incremental backup
        backup_manager.execute_backup("test_backup")
        
        # Verify backups
        backup_dirs = sorted(os.listdir(dest_dir))
        assert len(backup_dirs) == 2
        
        # Verify modified file in latest backup
        latest_backup = os.path.join(dest_dir, backup_dirs[-1])
        with open(os.path.join(latest_backup, "file1.txt")) as f:
            content = f.read()
            assert "Modified content" in content
        
        # Verify new file in latest backup
        assert os.path.exists(os.path.join(latest_backup, "newfile.txt"))
    
    def test_differential_backup(self, backup_manager, test_directories):
        """Test performing differential backups."""
        source_dir, dest_dir = test_directories
        
        # Update job to differential
        job = backup_manager.jobs["test_backup"]
        job.backup_type = BackupType.DIFFERENTIAL
        backup_manager.update_job(job)
        
        # Perform initial backup
        backup_manager.execute_backup("test_backup")
        
        # Make changes over time and perform backups
        for i in range(3):
            # Add new files
            with open(os.path.join(source_dir, f"diff_file_{i}.txt"), 'w') as f:
                f.write(f"Differential content {i}")
            
            # Perform differential backup
            backup_manager.execute_backup("test_backup")
        
        # Verify backups
        backup_dirs = sorted(os.listdir(dest_dir))
        assert len(backup_dirs) == 4  # Initial + 3 differential
        
        # Verify each differential backup contains all changes
        latest_backup = os.path.join(dest_dir, backup_dirs[-1])
        for i in range(3):
            assert os.path.exists(os.path.join(latest_backup, f"diff_file_{i}.txt"))
    
    def test_backup_retention(self, backup_manager, test_directories):
        """Test backup retention policy."""
        source_dir, dest_dir = test_directories
        
        # Set short retention period
        job = backup_manager.jobs["test_backup"]
        job.retention_days = 1
        backup_manager.update_job(job)
        
        # Create multiple backups with different timestamps
        for days_ago in [3, 2, 1, 0]:
            # Perform backup
            backup_manager.execute_backup("test_backup")
            
            # Modify timestamp of backup directory
            backup_dirs = os.listdir(dest_dir)
            latest_backup = os.path.join(dest_dir, backup_dirs[-1])
            timestamp = datetime.now() - timedelta(days=days_ago)
            os.utime(latest_backup, (timestamp.timestamp(), timestamp.timestamp()))
        
        # Clean up old backups
        backup_manager._cleanup_old_backups(job)
        
        # Verify only recent backups remain
        remaining_backups = os.listdir(dest_dir)
        assert len(remaining_backups) == 2  # Today and yesterday
    
    def test_backup_with_errors(self, backup_manager, test_directories):
        """Test handling of backup errors."""
        source_dir, dest_dir = test_directories
        
        # Make source directory unreadable
        os.chmod(source_dir, 0o000)
        
        try:
            # Attempt backup
            with pytest.raises(RuntimeError) as exc_info:
                backup_manager.execute_backup("test_backup")
            assert "Permission denied" in str(exc_info.value)
            
            # Verify no partial backup was created
            assert len(os.listdir(dest_dir)) == 0
            
        finally:
            # Restore permissions
            os.chmod(source_dir, 0o755)

def test_backup_config_persistence(backup_manager, test_directories):
    """Test that backup configuration persists correctly."""
    source_dir, dest_dir = test_directories
    
    # Add another job
    job = BackupJob(
        name="second_backup",
        source=source_dir,
        destination=dest_dir,
        backup_type=BackupType.INCREMENTAL,
        schedule="0 12 * * *",
        retention_days=14,
        enabled=True
    )
    backup_manager.create_job(job)
    
    # Create new manager instance with same config file
    new_manager = BackupManager(config_path=backup_manager.config_path)
    
    # Verify jobs were loaded
    assert len(new_manager.jobs) == 2
    assert "test_backup" in new_manager.jobs
    assert "second_backup" in new_manager.jobs
    
    # Verify job details were preserved
    loaded_job = new_manager.jobs["second_backup"]
    assert loaded_job.backup_type == BackupType.INCREMENTAL
    assert loaded_job.schedule == "0 12 * * *"
    assert loaded_job.retention_days == 14

