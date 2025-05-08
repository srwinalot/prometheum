"""
Health monitoring API routes.

This module provides endpoints for monitoring the health of the Prometheum system,
including storage devices, pools, and services.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Path

from prometheum.api.dependencies import get_health_monitor
from prometheum.storage.health import HealthMonitor
from prometheum.ml.predictor import HealthPredictor
from prometheum.ml.features import HealthFeatureExtractor
from prometheum.ml.trainer import HealthModelTrainer
from prometheum.ml.config import MLConfig

router = APIRouter()

@router.get("/status")
async def get_health_status():
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

# ML Configuration Management
@router.get("/predictions/config")
async def get_ml_config():
    """Get ML configuration."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'ml_config'):
            health_monitor.ml_config = MLConfig()
        
        return health_monitor.ml_config.config
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting ML configuration: {str(e)}"
        )

@router.post("/predictions/config")
async def update_ml_config(
    updates: Dict[str, Any] = Body(..., description="Configuration updates")
):
    """Update ML configuration."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'ml_config'):
            health_monitor.ml_config = MLConfig()
        
        health_monitor.ml_config.update_config(updates)
        
        # Update predictor thresholds if they exist in updates
        if hasattr(health_monitor, 'predictor') and health_monitor.predictor:
            prediction_config = health_monitor.ml_config.get_prediction_config()
            if "thresholds" in prediction_config:
                health_monitor.predictor.prediction_thresholds = prediction_config["thresholds"]
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "config": health_monitor.ml_config.config
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating ML configuration: {str(e)}"
        )

@router.get("/predictions/config/{section}")
async def get_ml_config_section(
    section: str = Path(..., description="Configuration section name")
):
    """Get a specific ML configuration section."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'ml_config'):
            health_monitor.ml_config = MLConfig()
        
        if section == "feature_extraction":
            return health_monitor.ml_config.get_feature_config()
        elif section == "prediction":
            return health_monitor.ml_config.get_prediction_config()
        elif section == "training":
            return health_monitor.ml_config.get_training_config()
        elif section == "monitoring":
            return health_monitor.ml_config.get_monitoring_config()
        elif section == "paths":
            return health_monitor.ml_config.config.get("paths", {})
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown configuration section: {section}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting ML configuration section: {str(e)}"
        )

@router.post("/predictions/config/{section}")
async def update_ml_config_section(
    section: str = Path(..., description="Configuration section name"),
    updates: Dict[str, Any] = Body(..., description="Section updates")
):
    """Update a specific ML configuration section."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'ml_config'):
            health_monitor.ml_config = MLConfig()
        
        # Create nested update
        config_update = {section: updates}
        health_monitor.ml_config.update_config(config_update)
        
        # Update predictor if prediction thresholds changed
        if section == "prediction" and hasattr(health_monitor, 'predictor'):
            if "thresholds" in updates:
                health_monitor.predictor.prediction_thresholds = updates["thresholds"]
        
        return {
            "status": "success",
            "message": f"Configuration section '{section}' updated",
            "section": health_monitor.ml_config.config.get(section, {})
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating ML configuration section: {str(e)}"
        )

@router.get("/predictions/config/value/{path:path}")
async def get_ml_config_value(
    path: str = Path(..., description="Configuration value path (dot-separated)")
):
    """Get a specific ML configuration value."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'ml_config'):
            health_monitor.ml_config = MLConfig()
        
        keys = path.split('.')
        value = health_monitor.ml_config.get_value(*keys)
        
        if value is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration value not found: {path}"
            )
        
        return {
            "path": path,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting ML configuration value: {str(e)}"
        )

@router.post("/predictions/config/value/{path:path}")
async def set_ml_config_value(
    path: str = Path(..., description="Configuration value path (dot-separated)"),
    value: Any = Body(..., description="New value")
):
    """Set a specific ML configuration value."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'ml_config'):
            health_monitor.ml_config = MLConfig()
        
        keys = path.split('.')
        health_monitor.ml_config.set_value(value, *keys)
        
        # Update predictor if prediction threshold changed
        if (len(keys) > 2 and keys[0] == "prediction" and 
            keys[1] == "thresholds" and hasattr(health_monitor, 'predictor')):
            health_monitor.predictor.prediction_thresholds[keys[2]] = value
        
        return {
            "status": "success",
            "message": f"Configuration value updated: {path}",
            "path": path,
            "value": value
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting ML configuration value: {str(e)}"
        )
# Prediction endpoints
@router.get("/predictions/{device}")
async def get_device_prediction(
    device: str,
    window_hours: int = Query(default=24, ge=1, le=168)
):
    """Get health prediction for a device."""
    health_monitor = get_health_monitor()
    
    try:
        # Initialize ML components if needed
        if not hasattr(health_monitor, 'predictor'):
            feature_extractor = HealthFeatureExtractor(health_monitor.history_db)
            health_monitor.predictor = HealthPredictor(feature_extractor)
        
        # Get prediction
        prediction = await health_monitor.predictor.predict_device_health(
            device,
            window_hours=window_hours
        )
        
        if "error" in prediction:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=prediction["error"]
            )
        
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting prediction for {device}: {str(e)}"
        )

@router.post("/predictions/train")
async def train_prediction_model(
    devices: Optional[List[str]] = None,
    window_hours: int = Query(default=24, ge=1, le=168),
    history_days: int = Query(default=30, ge=1, le=365),
    test_size: float = Query(default=0.2, ge=0.1, le=0.5)
):
    """Train the health prediction model."""
    health_monitor = get_health_monitor()
    
    try:
        # Get all devices if none specified
        if not devices:
            devices = health_monitor.pool_manager.get_all_devices()
        
        # Initialize ML components if needed
        if not hasattr(health_monitor, 'predictor'):
            feature_extractor = HealthFeatureExtractor(health_monitor.history_db)
            health_monitor.predictor = HealthPredictor(feature_extractor)
        
        trainer = HealthModelTrainer(health_monitor.predictor)
        
        # Train model
        metrics = await trainer.train_model(
            devices=devices,
            window_hours=window_hours,
            history_days=history_days,
            test_size=test_size
        )
        
        return {
            "status": "success",
            "message": "Model training completed",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training model: {str(e)}"
        )

@router.get("/predictions/features/{device}")
async def get_device_features(
    device: str,
    window_hours: int = Query(default=24, ge=1, le=168)
):
    """Get ML features for a device."""
    health_monitor = get_health_monitor()
    
    try:
        # Initialize ML components if needed
        if not hasattr(health_monitor, 'predictor'):
            feature_extractor = HealthFeatureExtractor(health_monitor.history_db)
            health_monitor.predictor = HealthPredictor(feature_extractor)
        
        # Get features
        features = await health_monitor.predictor.feature_extractor.get_device_features(
            device,
            window_hours=window_hours
        )
        
        return features
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting features for {device}: {str(e)}"
        )

@router.post("/predictions/model/save")
async def save_prediction_model(
    path: str = Query(..., description="Path to save the model file")
):
    """Save the trained prediction model."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'predictor') or not health_monitor.predictor.model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No trained model available"
            )
        
        health_monitor.predictor.save_model(path)
        return {
            "status": "success",
            "message": f"Model saved to {path}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving model: {str(e)}"
        )

@router.post("/predictions/model/load")
async def load_prediction_model(
    path: str = Query(..., description="Path to the model file")
):
    """Load a trained prediction model."""
    health_monitor = get_health_monitor()
    
    try:
        if not hasattr(health_monitor, 'predictor'):
            feature_extractor = HealthFeatureExtractor(health_monitor.history_db)
            health_monitor.predictor = HealthPredictor(feature_extractor)
        
        health_monitor.predictor.load_model(path)
        return {
            "status": "success",
            "message": f"Model loaded from {path}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )
@router.get("/devices")
async def list_device_health():
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
async def get_device_health(device: str):
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
async def run_health_check(device: Optional[str] = None):
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
async def get_alerts(include_resolved: bool = False):
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
async def get_device_alerts(device: str):
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
async def acknowledge_alert(alert_id: str):
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
async def resolve_alert(alert_id: str):
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

