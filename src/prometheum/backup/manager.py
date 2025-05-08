from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os
import subprocess
from typing import Dict, List, Optional
import json

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

@dataclass
class BackupJob:
    name: str
    source: str  # Source path or device
    destination: str  # Backup destination
    backup_type: BackupType
    schedule: str  # Cron expression
    retention_days: int
    exclude_patterns: List[str] = None
    last_backup: Optional[datetime] = None
    enabled: bool = True

class BackupManager:
    def __init__(self, config_path: str = "/etc/prometheum/backup.json"):
        self.config_path = config_path
        self.jobs: Dict[str, BackupJob] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load backup configuration from file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                for job_data in config.get('jobs', []):
                    job = BackupJob(
                        name=job_data['name'],
                        source=job_data['source'],
                        destination=job_data['destination'],
                        backup_type=BackupType(job_data['backup_type']),
                        schedule=job_data['schedule'],
                        retention_days=job_data['retention_days'],
                        exclude_patterns=job_data.get('exclude_patterns', []),
                        enabled=job_data.get('enabled', True)
                    )
                    if 'last_backup' in job_data:
                        job.last_backup = datetime.fromisoformat(job_data['last_backup'])
                    self.jobs[job.name] = job

    def save_config(self) -> None:
        """Save backup configuration to file."""
        config = {
            'jobs': [
                {
                    'name': job.name,
                    'source': job.source,
                    'destination': job.destination,
                    'backup_type': job.backup_type.value,
                    'schedule': job.schedule,
                    'retention_days': job.retention_days,
                    'exclude_patterns': job.exclude_patterns,
                    'last_backup': job.last_backup.isoformat() if job.last_backup else None,
                    'enabled': job.enabled
                }
                for job in self.jobs.values()
            ]
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def create_job(self, job: BackupJob) -> None:
        """Create a new backup job."""
        if job.name in self.jobs:
            raise ValueError(f"Backup job already exists: {job.name}")
        
        self.jobs[job.name] = job
        self.save_config()

    def update_job(self, job: BackupJob) -> None:
        """Update an existing backup job."""
        if job.name not in self.jobs:
            raise ValueError(f"Backup job not found: {job.name}")
        
        self.jobs[job.name] = job
        self.save_config()

    def delete_job(self, job_name: str) -> None:
        """Delete a backup job."""
        if job_name not in self.jobs:
            raise ValueError(f"Backup job not found: {job_name}")
        
        del self.jobs[job_name]
        self.save_config()

    def execute_backup(self, job_name: str) -> None:
        """Execute a backup job."""
        job = self.jobs.get(job_name)
        if not job:
            raise ValueError(f"Backup job not found: {job_name}")
            
        if not job.enabled:
            raise ValueError(f"Backup job is disabled: {job_name}")

        # Create destination directory if it doesn't exist
        os.makedirs(job.destination, exist_ok=True)

        # Determine backup path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(job.destination, f"{job.name}_{timestamp}")

        try:
            # Execute backup based on type
            if job.backup_type == BackupType.FULL:
                self._execute_full_backup(job, backup_path)
            elif job.backup_type == BackupType.INCREMENTAL:
                self._execute_incremental_backup(job, backup_path)
            elif job.backup_type == BackupType.DIFFERENTIAL:
                self._execute_differential_backup(job, backup_path)

            # Update last backup time
            job.last_backup = datetime.now()
            self.save_config()

            # Clean up old backups
            self._cleanup_old_backups(job)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Backup failed: {str(e)}")

    def _execute_full_backup(self, job: BackupJob, backup_path: str) -> None:
        """Execute a full backup."""
        exclude_args = []
        if job.exclude_patterns:
            for pattern in job.exclude_patterns:
                exclude_args.extend(['--exclude', pattern])

        cmd = ['rsync', '-av', '--delete'] + exclude_args + [job.source + '/', backup_path]
        subprocess.run(cmd, check=True)

    def _execute_incremental_backup(self, job: BackupJob, backup_path: str) -> None:
        """Execute an incremental backup."""
        # Find latest backup to use as reference
        latest = self._find_latest_backup(job)
        
        if latest:
            # Use hard links to previous backup for unchanged files
            cmd = [
                'rsync', '-av', '--delete', '--link-dest=' + latest,
                job.source + '/', backup_path
            ]
        else:
            # No previous backup, do full backup
            cmd = ['rsync', '-av', '--delete', job.source + '/', backup_path]

        if job.exclude_patterns:
            for pattern in job.exclude_patterns:
                cmd.extend(['--exclude', pattern])

        subprocess.run(cmd, check=True)

    def _execute_differential_backup(self, job: BackupJob, backup_path: str) -> None:
        """Execute a differential backup."""
        # Find initial full backup
        initial = self._find_initial_backup(job)
        
        if initial:
            # Use initial backup as reference
            cmd = [
                'rsync', '-av', '--delete', '--compare-dest=' + initial,
                job.source + '/', backup_path
            ]
        else:
            # No initial backup, do full backup
            cmd = ['rsync', '-av', '--delete', job.source + '/', backup_path]

        if job.exclude_patterns:
            for pattern in job.exclude_patterns:
                cmd.extend(['--exclude', pattern])

        subprocess.run(cmd, check=True)

    def _find_latest_backup(self, job: BackupJob) -> Optional[str]:
        """Find the latest backup for a job."""
        backups = []
        for item in os.listdir(job.destination):
            if item.startswith(job.name + '_'):
                path = os.path.join(job.destination, item)
                if os.path.isdir(path):
                    backups.append(path)
        
        return max(backups, default=None)

    def _find_initial_backup(self, job: BackupJob) -> Optional[str]:
        """Find the initial full backup for a job."""
        backups = []
        for item in os.listdir(job.destination):
            if item.startswith(job.name + '_'):
                path = os.path.join(job.destination, item)
                if os.path.isdir(path):
                    backups.append(path)
        
        return min(backups, default=None)

    def _cleanup_old_backups(self, job: BackupJob) -> None:
        """Clean up backups older than retention period."""
        if job.retention_days <= 0:
            return
            
        cutoff = datetime.now().timestamp() - (job.retention_days * 86400)
        
        for item in os.listdir(job.destination):
            if item.startswith(job.name + '_'):
                path = os.path.join(job.destination, item)
                if os.path.isdir(path):
                    stat = os.stat(path)
                    if stat.st_mtime < cutoff:
                        subprocess.run(['rm', '-rf', path], check=True)

