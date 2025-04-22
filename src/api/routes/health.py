
"""
Health monitoring routes.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_health_monitor, get_current_user

router = APIRouter()

@router.get("/status")
async def get_health_status(user = Depends(get_current_user)):
    """Get system health status overview."""
    health_monitor = get_health_monitor()
    
    # Make sure health monitoring is running
    if not health_monitor.monitoring_active:
        health_monitor.start_monitoring()
    
    # Compile health overview
    try:
        return {
            "health_status": "healthy",  # Default to healthy, will be overridden if issues found
            "device_summary": {
                device: data.get("health_status", "unknown")
                for device, data in health_monitor.get_all_health_data().items()
            },
            "active_alerts_count": len(health_monitor.get_alerts()),
            "timestamp": health_monitor.last_update_time if hasattr(health_monitor, "last_update_time") else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving health status: {str(e)}"
        )

@router.get("/devices")
async def list_device_health(user = Depends(get_current_user)):
    """List health data for all devices."""
    health_monitor = get_health_monitor()
    
    try:
        return health_monitor.get_all_health_data()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving device health data: {str(e)}"
        )

@router.get("/devices/{device}")
async def get_device_health(device: str, user = Depends(get_current_user)):
    """Get health data for a specific device."""
    health_monitor = get_health_monitor()
    
    try:
        health_data = health_monitor.get_device_health(device)
        
        if "error" in health_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=health_data["error"]
            )
        
        return health_data
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving health data for {device}: {str(e)}"
        )

@router.post("/check")
async def run_health_check(
    device: Optional[str] = None,
    user = Depends(get_current_user)
):
    """Manually run a health check."""
    health_monitor = get_health_monitor()
    
    try:
        result = health_monitor.manual_check(device)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running health check: {str(e)}"
        )

# Alert management
@router.get("/alerts")
async def get_alerts(
    include_resolved: bool = False,
    user = Depends(get_current_user)
):
    """Get system health alerts."""
    health_monitor = get_health_monitor()
    
    try:
        return health_monitor.get_alerts(include_resolved)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving alerts: {str(e)}"
        )

@router.get("/alerts/{device}")
async def get_device_alerts(device: str, user = Depends(get_current_user)):
    """Get alerts for a specific device."""
    health_monitor = get_health_monitor()
    
    try:
        return health_monitor.get_alerts_for_device(device)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving alerts for {device}: {str(e)}"
        )

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user = Depends(get_current_user)):
    """Acknowledge an alert."""
    health_monitor = get_health_monitor()
    
    try:
        if health_monitor.acknowledge_alert(alert_id):
            return {"success": True, "message": f"Alert {alert_id} acknowledged"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found"
            )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error acknowledging alert: {str(e)}"
        )

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, user = Depends(get_current_user)):
    """Resolve an alert."""
    health_monitor = get_health_monitor()
    
    try:
        if health_monitor.resolve_alert(alert_id):
            return {"success": True, "message": f"Alert {alert_id} resolved"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found"
            )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resolving alert: {str(e)}"
        )

