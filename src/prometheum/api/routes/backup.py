from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from prometheum.backup.manager import BackupManager, BackupJob, BackupType

router = APIRouter()
backup_manager = BackupManager()

class BackupJobRequest(BaseModel):
    name: str
    source: str
    destination: str
    backup_type: str
    schedule: str
    retention_days: int
    exclude_patterns: Optional[List[str]] = None
    enabled: bool = True

class BackupJobResponse(BackupJobRequest):
    last_backup: Optional[datetime] = None

@router.get("/jobs")
async def list_jobs():
    """List all backup jobs."""
    return {
        "jobs": [
            BackupJobResponse(
                name=job.name,
                source=job.source,
                destination=job.destination,
                backup_type=job.backup_type.value,
                schedule=job.schedule,
                retention_days=job.retention_days,
                exclude_patterns=job.exclude_patterns,
                enabled=job.enabled,
                last_backup=job.last_backup
            )
            for job in backup_manager.jobs.values()
        ]
    }

@router.post("/jobs")
async def create_job(job: BackupJobRequest):
    """Create a new backup job."""
    try:
        backup_job = BackupJob(
            name=job.name,
            source=job.source,
            destination=job.destination,
            backup_type=BackupType(job.backup_type),
            schedule=job.schedule,
            retention_days=job.retention_days,
            exclude_patterns=job.exclude_patterns,
            enabled=job.enabled
        )
        backup_manager.create_job(backup_job)
        return {"message": f"Backup job created: {job.name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/jobs/{job_name}")
async def update_job(job_name: str, job: BackupJobRequest):
    """Update an existing backup job."""
    try:
        backup_job = BackupJob(
            name=job_name,
            source=job.source,
            destination=job.destination,
            backup_type=BackupType(job.backup_type),
            schedule=job.schedule,
            retention_days=job.retention_days,
            exclude_patterns=job.exclude_patterns,
            enabled=job.enabled
        )
        backup_manager.update_job(backup_job)
        return {"message": f"Backup job updated: {job_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/jobs/{job_name}")
async def delete_job(job_name: str):
    """Delete a backup job."""
    try:
        backup_manager.delete_job(job_name)
        return {"message": f"Backup job deleted: {job_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/jobs/{job_name}/execute")
async def execute_job(job_name: str):
    """Execute a backup job."""
    try:
        backup_manager.execute_backup(job_name)
        return {"message": f"Backup job executed: {job_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

