"""
Storage health monitoring system.

This module provides a comprehensive health monitoring system for storage devices,
integrating SMART data collection, performance metrics, and ML-based prediction.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json

from ..storage.smart import get_smart_data, is_device_supported
from ..storage.metrics import MetricsCollector, get_device_metrics
from ..ml.predictor import HealthPredictor
from ..ml.features import HealthFeatureExtractor
from ..ml.trainer import HealthModelTrainer
from ..ml.config import MLConfig
from ..storage.history import HealthHistoryDB

logger = logging.getLogger(__name__)

class HealthAlert:
    """Storage health alert."""
    
    # Alert severity levels
    SEVERITY_INFO = "info"
    SEVERITY_WARNING = "warning"
    SEVERITY_CRITICAL = "critical"
    
    def __init__(
        self, 
        device: str, 
        message: str, 
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize alert."""
        self.id = str(uuid.uuid4())
        self.device = device
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        self.acknowledged = False
        self.resolved = False
        self.resolution_time = None
    
    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
    
    def resolve(self) -> None:
        """Resolve the alert."""
        self.resolved = True
        self.resolution_time = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "device": self.device,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time
        }


class HealthMonitor:
    """Storage health monitoring system.
    
    Monitors storage devices health by:
    1. Collecting and analyzing SMART data
    2. Tracking performance metrics
    3. Using ML to predict potential failures
    4. Generating alerts for health issues
    5. Maintaining historical health data
    """
    
    def __init__(
        self,
        history_db_path: str = "health_history.db",
        devices: Optional[List[str]] = None,
        ml_config_path: Optional[str] = None
    ):
        """Initialize health monitor.
        
        Args:
            history_db_path: Path to the SQLite database for health history
            devices: List of devices to monitor (optional)
            ml_config_path: Path to ML configuration file (optional)
        """
        # Initialize components
        self.history_db = HealthHistoryDB(history_db_path)
        self.metrics_collector = MetricsCollector()
        self.devices = devices or []
        self.disk_health: Dict[str, Dict[str, Any]] = {}
        self.alerts: Dict[str, HealthAlert] = {}
        self.monitoring_active = False
        self.monitoring_task = None
        self.last_update_time = None
        
        # Initialize ML components
        self.ml_config = MLConfig(config_path=ml_config_path)
        self.feature_extractor = HealthFeatureExtractor(self.history_db)
        self.predictor = HealthPredictor(self.feature_extractor)
        
        # Load model if available
        model_path = self.ml_config.get_value("paths", "model")
        if model_path:
            try:
                self.predictor.load_model(model_path)
                logger.info(f"Loaded prediction model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load prediction model: {e}")
        
        # Configure monitoring interval (in seconds)
        self.check_interval = self.ml_config.get_value("monitoring", "check_interval") or 300

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Check each device
                for device in self.devices:
                    await self._check_device(device)
                
                # Update last check time
                self.last_update_time = datetime.now().isoformat()
                
                # Run periodic ML tasks if configured
                await self._run_periodic_ml_tasks()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying after error

    async def _run_periodic_ml_tasks(self) -> None:
        """Run periodic ML tasks like retraining."""
        try:
            # Check if model retraining is due
            if not self.ml_config.get_value("training", "auto_retrain", False):
                return
            
            # Get last training time
            last_train = self.ml_config.get_value("training", "last_train_time")
            retrain_days = self.ml_config.get_value("training", "retrain_interval_days", 7)
            
            if last_train:
                last_train_time = datetime.fromisoformat(last_train)
                days_since = (datetime.now() - last_train_time).days
                
                if days_since >= retrain_days:
                    logger.info("Periodic model retraining triggered")
                    
                    # Retrain model
                    trainer = HealthModelTrainer(self.predictor)
                    window_hours = self.ml_config.get_value("training", "window_hours", 24)
                    history_days = self.ml_config.get_value("training", "history_days", 30)
                    
                    metrics = await trainer.train_model(
                        devices=self.devices,
                        window_hours=window_hours,
                        history_days=history_days,
                        test_size=0.2
                    )
                    
                    # Update training time
                    self.ml_config.set_value(datetime.now().isoformat(), "training", "last_train_time")
                    
                    # Save model if path is configured
                    model_path = self.ml_config.get_value("paths", "model")
                    if model_path:
                        self.predictor.save_model(model_path)
                        logger.info(f"Saved retrained model to {model_path}")
        except Exception as e:
            logger.error(f"Error running periodic ML tasks: {e}")

    async def _check_device(self, device: str) -> Dict[str, Any]:
        """Run health check for a device."""
        try:
            # Get SMART data
            smart_data = await self._get_smart_data(device)
            smart_health = smart_data.get("overall_health", "unknown")
            
            # Get performance metrics
            metrics = await self._get_performance_metrics(device)
            
            # Get prediction if ML model available
            prediction = None
            if self.predictor.model:
                try:
                    prediction = await self.predictor.predict_device_health(device)
                except Exception as e:
                    logger.error(f"Error getting prediction for {device}: {e}")
            
            # Determine overall health status
            health_status = "healthy"
            if smart_health == "failed" or (prediction and prediction.get("status") == "failure"):
                health_status = "failed"
            elif smart_health == "warning" or (prediction and prediction.get("status") == "warning"):
                health_status = "warning"
            elif smart_health == "unknown":
                health_status = "unknown"
            
            # Compile health data
            health_data = {
                "device": device,
                "timestamp": datetime.now().isoformat(),
                "health_status": health_status,
                "smart_data": smart_data,
                "metrics": metrics,
                "prediction": prediction
            }
            
            # Update cache
            self.disk_health[device] = health_data
            
            # Store in history database
            await self.history_db.store_smart_data(health_data)
            
            # Generate alerts based on health status
            self._generate_alerts(device, health_data)
            
            return health_data
        except Exception as e:
            logger.error(f"Error checking device {device}: {e}")
            error_data = {
                "device": device,
                "timestamp": datetime.now().isoformat(),
                "health_status": "error",
                "error": str(e)
            }
            self.disk_health[device] = error_data
            return error_data

    async def _get_smart_data(self, device: str) -> Dict[str, Any]:
        """Get SMART data for a device."""
        try:
            # Check if device supports SMART
            supported = await is_device_supported(device)
            if not supported:
                return {
                    "overall_health": "unknown",
                    "error": "Device does not support SMART"
                }
            
            # Get SMART data
            smart_data = await get_smart_data(device)
            return smart_data.to_dict()
        except Exception as e:
            logger.error(f"Error getting SMART data for {device}: {e}")
            return {
                "overall_health": "error",
                "error": str(e)
            }

    async def _get_performance_metrics(self, device: str) -> Dict[str, Any]:
        """Get performance metrics for a device."""
        try:
            # Get current metrics
            current = await get_device_metrics(device, self.metrics_collector)
            
            # Get average metrics over last hour
            averages = self.metrics_collector.devices.get(device)
            if averages:
                average_metrics = averages.get_average_metrics(interval=3600)
            else:
                average_metrics = {"status": "no_data"}
            
            return {
                "current": current,
                "hourly_average": average_metrics
            }
        except Exception as e:
            logger.error(f"Error getting metrics for {device}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _generate_alerts(self, device: str, health_data: Dict[str, Any]) -> None:
        """Generate alerts based on health data."""
        health_status = health_data.get("health_status")
        smart_data = health_data.get("smart_data", {})
        prediction = health_data.get("prediction", {})
        
        # Check for failure status
        if health_status == "failed":
            self.add_alert(
                device,
                "Device health check failed",
                HealthAlert.SEVERITY_CRITICAL,
                details={"smart_health": smart_data.get("overall_health")}
            )
            return
        
        # Check for warning status
        if health_status == "warning":
            self.add_alert(
                device,
                "Device health check warning",
                HealthAlert.SEVERITY_WARNING,
                details={"smart_health": smart_data.get("overall_health")}
            )
        
        # Check for high temperature
        temperature = smart_data.get("temperature")
        if temperature and temperature >= 65:
            self.add_alert(
                device,
                f"High temperature detected: {temperature}°C",
                HealthAlert.SEVERITY_CRITICAL
            )
        elif temperature and temperature >= 55:
            self.add_alert(
                device,
                f"Elevated temperature detected: {temperature}°C",
                HealthAlert.SEVERITY_WARNING
            )
        
        # Check for prediction warnings
        if prediction and prediction.get("status") == "warning":
            factors = prediction.get("contributing_factors", [])
            self.add_alert(
                device,
                "Potential failure predicted",
                HealthAlert.SEVERITY_WARNING,
                details={
                    "confidence": prediction.get("confidence"),
                    "factors": factors
                }
            )
        
        # Check for prediction failure
        if prediction and prediction.get("status") == "failure":
            factors = prediction.get("contributing_factors", [])
            self.add_alert(
                device,
                "Imminent failure predicted",
                HealthAlert.SEVERITY_CRITICAL,
                details={
                    "confidence": prediction.get("confidence"),
                    "factors": factors
                }
            )

    def add_alert(
        self, 
        device: str, 
        message: str, 
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None
    ) -> HealthAlert:
        """Add a health alert."""
        alert = HealthAlert(device, message, severity, details)
        self.alerts[alert.id] = alert
        
        # Store in database
        asyncio.create_task(self.history_db.store_alert(alert.to_dict()))
        
        logger.warning(f"Health alert: {device} - {message} ({severity})")
        return alert

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledge()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolve()
            return True
        return False

    def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [
            alert.to_dict()
            for alert in self.alerts.values()
            if include_resolved or not alert.resolved
        ]

    def get_device_health(self, device: str) -> Dict[str, Any]:
        """Retrieve health data for a specific device.
        
        Args:
            device: The device identifier to retrieve health data for
            
        Returns:
            Dictionary containing health data for the specified device
            
        Raises:
            ValueError: If the device is not being monitored
        """
        if device not in self.devices:
            raise ValueError(f"Device {device} is not being monitored")
            
        # If we have cached data, return it
        if device in self.disk_health:
            return self.disk_health[device]
            
        # Otherwise, inform that data is not available
        return {
            "device": device,
            "timestamp": datetime.now().isoformat(),
            "health_status": "unknown",
            "message": "Health data not yet collected"
        }
    
    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all monitored devices.
        
        Returns:
            Dictionary mapping device identifiers to their health data
        """
        result = {}
        for device in self.devices:
            try:
                result[device] = self.get_device_health(device)
            except Exception as e:
                logger.error(f"Error getting health for {device}: {e}")
                result[device] = {
                    "device": device,
                    "timestamp": datetime.now().isoformat(),
                    "health_status": "error",
                    "error": str(e)
                }
        return result
    
    def add_device(self, device: str) -> bool:
        """Add a device to monitoring.
        
        Args:
            device: The device identifier to add
            
        Returns:
            True if device was added successfully, False if already monitored
            
        Raises:
            ValueError: If the device does not exist or is not a valid storage device
        """
        # Check if device is already monitored
        if device in self.devices:
            logger.info(f"Device {device} is already being monitored")
            return False
            
        # Verify device exists and is valid
        try:
            # Run a check to verify device exists and supports health monitoring
            asyncio.get_event_loop().run_until_complete(is_device_supported(device))
            
            # Add device to monitored list
            self.devices.append(device)
            logger.info(f"Added device {device} to health monitoring")
            
            # Initiate an immediate health check
            if self.monitoring_active:
                asyncio.create_task(self._check_device(device))
                
            return True
        except Exception as e:
            logger.error(f"Could not add device {device}: {e}")
            raise ValueError(f"Device {device} cannot be monitored: {e}")
    
    def remove_device(self, device: str) -> bool:
        """Remove a device from monitoring.
        
        Args:
            device: The device identifier to remove
            
        Returns:
            True if device was removed, False if not found
        """
        if device not in self.devices:
            logger.info(f"Device {device} is not being monitored")
            return False
            
        # Remove from monitoring list
        self.devices.remove(device)
        
        # Remove cached health data
        if device in self.disk_health:
            del self.disk_health[device]
            
        logger.info(f"Removed device {device} from health monitoring")
        return True
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get current ML configuration.
        
        Returns:
            Dictionary containing ML configuration settings
        """
        return self.ml_config.get_all()
    
    def update_ml_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update ML configuration.
        
        Args:
            config_updates: Dictionary with configuration updates
            
        Returns:
            True if configuration was updated successfully
            
        Raises:
            ValueError: If configuration updates are invalid
        """
        try:
            # Apply each update
            for path, value in config_updates.items():
                # Split path into sections
                parts = path.split('.')
                
                if len(parts) == 1:
                    # Top-level setting
                    self.ml_config.set_value(value, parts[0])
                elif len(parts) == 2:
                    # Nested setting
                    self.ml_config.set_value(value, parts[0], parts[1])
                else:
                    # Deeper nesting not supported
                    raise ValueError(f"Invalid configuration path: {path}")
            
            # Save configuration
            self.ml_config.save()
            
            # Check if we need to reload model
            if "paths.model" in config_updates or "monitoring.use_ml" in config_updates:
                model_path = self.ml_config.get_value("paths", "model")
                use_ml = self.ml_config.get_value("monitoring", "use_ml", True)
                
                if use_ml and model_path:
                    try:
                        self.predictor.load_model(model_path)
                        logger.info(f"Reloaded prediction model from {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to reload prediction model: {e}")
            
            logger.info("Updated ML configuration")
            return True
        except Exception as e:
            logger.error(f"Error updating ML configuration: {e}")
            raise ValueError(f"Failed to update configuration: {e}")
    
    async def get_health_history(
        self, 
        device: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve historical health data.
        
        Args:
            device: Device identifier to retrieve history for
            start_time: Optional start time for filtering data
            end_time: Optional end time for filtering data
            limit: Maximum number of records to return
            
        Returns:
            List of health data entries in chronological order
        """
        try:
            # Default timeframes if not specified
            if not end_time:
                end_time = datetime.now()
                
            if not start_time:
                start_time = end_time - timedelta(days=7)
                
            # Query history database
            history = await self.history_db.get_smart_history(
                device, 
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving health history for {device}: {e}")
            return []
    
    async def get_prediction(self, device: str) -> Dict[str, Any]:
        """Get latest prediction for a device.
        
        Args:
            device: Device identifier to get prediction for
            
        Returns:
            Dictionary with prediction results or error information
            
        Raises:
            ValueError: If the device is not being monitored
        """
        if device not in self.devices:
            raise ValueError(f"Device {device} is not being monitored")
            
        try:
            # Check if ML is enabled
            if not self.predictor.model:
                return {
                    "status": "unavailable",
                    "message": "ML prediction is not enabled or model not loaded"
                }
                
            # Get prediction
            prediction = await self.predictor.predict_device_health(device)
            
            # If prediction exists in device health data, it may have more details
            if device in self.disk_health and "prediction" in self.disk_health[device]:
                cached_prediction = self.disk_health[device]["prediction"]
                if cached_prediction:
                    # Use cached prediction if more recent
                    if (cached_prediction.get("timestamp") and 
                        prediction.get("timestamp") and
                        cached_prediction["timestamp"] > prediction["timestamp"]):
                        return cached_prediction
            
            return prediction
        except Exception as e:
            logger.error(f"Error getting prediction for {device}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

