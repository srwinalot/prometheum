
"""
Storage management routes.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..dependencies import get_pool_manager, get_volume_manager, get_current_user

router = APIRouter()

# Models
class PoolCreate(BaseModel):
    """Pool creation request."""
    name: str
    fs_type: str
    devices: List[str]
    mountpoint: Optional[str] = None
    options: Optional[Dict[str, str]] = None

class VolumeCreate(BaseModel):
    """Volume creation request."""
    name: str
    pool_name: str
    size: Optional[str] = None
    mountpoint: Optional[str] = None
    type: str = "filesystem"
    options: Optional[Dict[str, str]] = None

class ShareConfig(BaseModel):
    """Share configuration."""
    protocol: str
    options: Optional[Dict[str, str]] = None

# Pool endpoints
@router.get("/pools")
async def list_pools(user = Depends(get_current_user)):
    """List all storage pools."""
    pool_manager = get_pool_manager()
    return [pool.to_dict() for pool in pool_manager.list_pools()]

@router.post("/pools", status_code=status.HTTP_201_CREATED)
async def create_pool(pool_data: PoolCreate, user = Depends(get_current_user)):
    """Create a new storage pool."""
    pool_manager = get_pool_manager()
    
    try:
        pool = pool_manager.create_pool(
            name=pool_data.name,
            fs_type=pool_data.fs_type,
            devices=pool_data.devices,
            mountpoint=pool_data.mountpoint,
            options=pool_data.options
        )
        return pool.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/pools/{name}")
async def get_pool(name: str, user = Depends(get_current_user)):
    """Get a storage pool by name."""
    pool_manager = get_pool_manager()
    pool = pool_manager.get_pool(name)
    
    if not pool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pool '{name}' not found"
        )
    
    return pool.to_dict()

@router.delete("/pools/{name}")
async def delete_pool(name: str, user = Depends(get_current_user)):
    """Delete a storage pool."""
    pool_manager = get_pool_manager()
    
    try:
        if not pool_manager.delete_pool(name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pool '{name}' not found"
            )
        return {"success": True, "message": f"Pool '{name}' deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pool: {str(e)}"
        )

# Volume endpoints
@router.get("/volumes")
async def list_volumes(
    pool_name: Optional[str] = None,
    user = Depends(get_current_user)
):
    """List all volumes, optionally filtered by pool."""
    volume_manager = get_volume_manager()
    return [vol.to_dict() for vol in volume_manager.list_volumes(pool_name)]

@router.post("/volumes", status_code=status.HTTP_201_CREATED)
async def create_volume(volume_data: VolumeCreate, user = Depends(get_current_user)):
    """Create a new volume."""
    volume_manager = get_volume_manager()
    
    try:
        volume = volume_manager.create_volume(
            name=volume_data.name,
            pool_name=volume_data.pool_name,
            size=volume_data.size,
            mountpoint=volume_data.mountpoint,
            type=volume_data.type,
            options=volume_data.options
        )
        return volume.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/volumes/{name}")
async def get_volume(name: str, user = Depends(get_current_user)):
    """Get a volume by name."""
    volume_manager = get_volume_manager()
    volume = volume_manager.get_volume(name)
    
    if not volume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Volume '{name}' not found"
        )
    
    return volume.to_dict()

@router.delete("/volumes/{name}")
async def delete_volume(
    name: str, 
    destroy_data: bool = False,
    user = Depends(get_current_user)
):
    """Delete a volume."""
    volume_manager = get_volume_manager()
    
    try:
        if not volume_manager.delete_volume(name, destroy_data):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Volume '{name}' not found"
            )
        return {"success": True, "message": f"Volume '{name}' deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete volume: {str(e)}"
        )

# Snapshot management
@router.post("/volumes/{name}/snapshots")
async def create_snapshot(
    name: str, 
    snapshot_name: Optional[str] = None,
    user = Depends(get_current_user)
):
    """Create a snapshot of a volume."""
    volume_manager = get_volume_manager()
    
    try:
        result = volume_manager.create_snapshot(name, snapshot_name)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create snapshot: {str(e)}"
        )

@router.delete("/volumes/{name}/snapshots/{snapshot_name}")
async def delete_snapshot(
    name: str,
    snapshot_name: str,
    user = Depends(get_current_user)
):
    """Delete a snapshot."""
    volume_manager = get_volume_manager()
    
    try:
        if not volume_manager.delete_snapshot(name, snapshot_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot '{snapshot_name}' not found"
            )
        return {"success": True, "message": f"Snapshot '{snapshot_name}' deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete snapshot: {str(e)}"
        )

# Share configuration
@router.post("/volumes/{name}/share")
async def configure_share(
    name: str,
    share_config: ShareConfig,
    user = Depends(get_current_user)
):
    """Configure a volume as a network share."""
    volume_manager = get_volume_manager()
    
    try:
        if volume_manager.configure_share(
            name,
            share_config.protocol,
            share_config.options
        ):
            return {"success": True, "message": f"Volume '{name}' shared as {share_config.protocol}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to configure share"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure share: {str(e)}"
        )

@router.post("/volumes/{name}/quota")
async def set_quota(
    name: str,
    quota_size: str,
    user = Depends(get_current_user)
):
    """Set a quota on a volume."""
    volume_manager = get_volume_manager()
    
    try:
        if volume_manager.set_quota(name, quota_size):
            return {"success": True, "message": f"Quota set to {quota_size} for volume '{name}'"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to set quota"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set quota: {str(e)}"
        )
