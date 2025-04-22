"""
Disk health monitoring for Prometheum.

This module provides classes for monitoring disk health, collecting performance
metrics, detecting and reporting errors, and generating alerts for potential issues.
"""

import json
import logging
import os
import re
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Callable

from .pool import StoragePoolManager, StoragePool
from .volume import VolumeManager, Volume
from .utils import run_command, CommandError

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for devices and pools."""
    
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceMetric(Enum):
    """Types of performance metrics collected."""
    
    IOPS_READ = "iops_read"
    IOPS_WRITE = "iops_write"
    THROUGHPUT_READ = "throughput_read"
    THROUGHPUT_WRITE = "throughput_write"
    LATENCY_READ = "latency_read"
    LATENCY_WRITE = "latency_write"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    TEMPERATURE = "temperature"


class SmartAttribute:
    """Represents a SMART attribute with its values and thresholds."""
    
    def __init__(
        self,
        id: int,
        name: str,
        value: int,
        worst: int,
        threshold: int,
        raw_value: str,
        status: str
    ):
        """Initialize a SMART attribute.
        
        Args:
            id: Attribute ID
            name: Attribute name
            value: Normalized value (0-100)
            worst: Worst value observed
            threshold: Failure threshold
            raw_value: Raw attribute value
            status: Status (e.g., "OK", "FAILING")
        """
        self.id = id
        self.name = name
        self.value = value
        self.worst = worst
        self.threshold = threshold
        self.raw_value = raw_value
        self.status = status
    
    def is_failing(self) -> bool:
        """Check if the attribute is failing."""
        return self.value <= self.threshold or "FAIL" in self.status.upper()
    
    def is_warning(self) -> bool:
        """Check if the attribute is in warning state."""
        # Warning if the value is within 10% of the threshold
        return not self.is_failing() and self.value <= (self.threshold * 1.1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "worst": self.worst,
            "threshold": self.threshold,
            "raw_value": self.raw_value,
            "status": self.status,
            "is_failing": self.is_failing(),
            "is_warning": self.is_warning()
        }


class SmartData:
    """Contains SMART data for a disk."""
    
    def __init__(
        self,
        device: str,
        model: str,
        serial: str,
        firmware: str,
        temperature: Optional[float] = None,
        power_on_hours: Optional[int] = None,
        attributes: Optional[Dict[int, SmartAttribute]] = None
    ):
        """Initialize SMART data.
        
        Args:
            device: Device name (e.g., "sda")
            model: Device model
            serial: Serial number
            firmware: Firmware version
            temperature: Current temperature in Celsius
            power_on_hours: Hours of operation
            attributes: Dict of SMART attributes by ID
        """
        self.device = device
        self.model = model
        self.serial = serial
        self.firmware = firmware
        self.temperature = temperature
        self.power_on_hours = power_on_hours
        self.attributes = attributes or {}
        self.timestamp = datetime.now().isoformat()
    
    def get_health_status(self) -> HealthStatus:
        """Determine overall health status based on SMART attributes."""
        if not self.attributes:
            return HealthStatus.UNKNOWN
        
        # Check for failing attributes
        for attr in self.attributes.values():
            if attr.is_failing():
                return HealthStatus.CRITICAL
            
        # Check for warning attributes
        for attr in self.attributes.values():
            if attr.is_warning():
                return HealthStatus.WARNING
                
        return HealthStatus.GOOD
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "device": self.device,
            "model": self.model,
            "serial": self.serial,
            "firmware": self.firmware,
            "temperature": self.temperature,
            "power_on_hours": self.power_on_hours,
            "health_status": self.get_health_status().value,
            "attributes": {str(k): v.to_dict() for k, v in self.attributes.items()},
            "timestamp": self.timestamp
        }


class DiskHealthData:
    """Combines health data for a disk including SMART and performance metrics."""
    
    def __init__(
        self,
        device: str,
        smart_data: Optional[SmartData] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        errors: Optional[List[Dict]] = None
    ):
        """Initialize disk health data.
        
        Args:
            device: Device name (e.g., "sda")
            smart_data: SMART data for the device
            performance_metrics: Performance metric values
            errors: List of errors detected
        """
        self.device = device
        self.smart_data = smart_data
        self.performance_metrics = performance_metrics or {}
        self.errors = errors or []
        self.last_updated = datetime.now().isoformat()
    
    def get_health_status(self) -> HealthStatus:
        """Determine overall health status."""
        # If smart data indicates critical or warning, return that
        if self.smart_data:
            smart_status = self.smart_data.get_health_status()
            if smart_status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                return smart_status
                
        # If there are critical or error level errors, return critical
        if any(error.get("level") in ["critical", "error"] for error in self.errors):
            return HealthStatus.CRITICAL
            
        # If there are warning level errors, return warning
        if any(error.get("level") == "warning" for error in self.errors):
            return HealthStatus.WARNING
            
        # If we have smart data and it's good, return good
        if self.smart_data and smart_status == HealthStatus.GOOD:
            return HealthStatus.GOOD
            
        # Otherwise return unknown
        return HealthStatus.UNKNOWN
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "device": self.device,
            "smart_data": self.smart_data.to_dict() if self.smart_data else None,
            "performance_metrics": self.performance_metrics,
            "errors": self.errors,
            "health_status": self.get_health_status().value,
            "last_updated": self.last_updated
        }


class Alert:
    """Represents a health alert."""
    
    def __init__(
        self,
        level: AlertLevel,
        device: str,
        message: str,
        details: Optional[Dict] = None,
        timestamp: Optional[str] = None
    ):
        """Initialize an alert.
        
        Args:
            level: Alert severity level
            device: Device name the alert is for
            message: Alert message
            details: Additional details
            timestamp: Alert timestamp (default: current time)
        """
        self.id = f"{int(time.time())}_{device}"
        self.level = level if isinstance(level, AlertLevel) else AlertLevel(level)
        self.device = device
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.now().isoformat()
        self.acknowledged = False
        self.resolved = False
        self.resolved_timestamp = None
    
    def acknowledge(self) -> None:
        """Mark the alert as acknowledged."""
        self.acknowledged = True
    
    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved = True
        self.resolved_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "level": self.level.value,
            "device": self.device,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp
        }


class AlertManager:
    """Manages system alerts."""
    
    def __init__(
        self, 
        config_path: str = "/var/lib/prometheum/storage/alerts.json",
        alert_handlers: Optional[List[Callable[[Alert], None]]] = None
    ):
        """Initialize the alert manager.
        
        Args:
            config_path: Path to alerts configuration
            alert_handlers: List of callback functions to handle new alerts
        """
        self.config_path = config_path
        self.alert_handlers = alert_handlers or []
        self.alerts: Dict[str, Alert] = {}
        self._load_alerts()
    
    def _load_alerts(self) -> None:
        """Load alerts from configuration file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    alerts_data = json.load(f)
                
                for alert_data in alerts_data.get("alerts", []):
                    alert = Alert(
                        level=AlertLevel(alert_data["level"]),
                        device=alert_data["device"],
                        message=alert_data["message"],
                        details=alert_data.get("details", {}),
                        timestamp=alert_data["timestamp"]
                    )
                    alert.id = alert_data["id"]
                    alert.acknowledged = alert_data.get("acknowledged", False)
                    alert.resolved = alert_data.get("resolved", False)
                    alert.resolved_timestamp = alert_data.get("resolved_timestamp")
                    
                    self.alerts[alert.id] = alert
                
                logger.info(f"Loaded {len(self.alerts)} alerts from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading alerts: {e}")
        else:
            logger.info(f"Alerts configuration file not found at {self.config_path}")
    
    def _save_alerts(self) -> None:
        """Save alerts to configuration file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Only keep last 1000 alerts to prevent file growth
        recent_alerts = sorted(
            self.alerts.values(),
            key=lambda a: a.timestamp,
            reverse=True
        )[:1000]
        
        alerts_data = {
            "alerts": [alert.to_dict() for alert in recent_alerts]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(alerts_data, f, indent=2)
    
    def add_alert(
        self,
        level: Union[AlertLevel, str],
        device: str,
        message: str,
        details: Optional[Dict] = None
    ) -> Alert:
        """Add a new alert.
        
        Args:
            level: Alert severity level
            device: Device name the alert is for
            message: Alert message
            details: Additional details
            
        Returns:
            The created Alert
        """
        # Convert level to enum if needed
        if isinstance(level, str):
            level = AlertLevel(level)
            
        # Create alert
        alert = Alert(level, device, message, details)
        self.alerts[alert.id] = alert
        
        # Save alerts
        self._save_alerts()
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.info(f"Added new alert [{level.value}] for {device}: {message}")
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if the alert was acknowledged, False otherwise
        """
        if alert_id not in self.alerts:
            return False
            
        self.alerts[alert_id].acknowledge()
        self._save_alerts()
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if the alert was resolved, False otherwise
        """
        if alert_id not in self.alerts:
            return False
            
        self.alerts[alert_id].resolve()
        self._save_alerts()
        return True
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active (unresolved) alerts.
        
        Args:
            level: Optional filter by alert level
            
        Returns:
            List of active alerts
        """
        active_alerts = [
            alert for alert in self.alerts.values() 
            if not alert.resolved
        ]
        
        if level:
            active_alerts = [
                alert for alert in active_alerts
                if alert.level == level
            ]
            
        return active_alerts
    
    def get_alerts_for_device(self, device: str) -> List[Alert]:
        """Get all alerts for a specific device.
        
        Args:
            device: Device name
            
        Returns:
            List of alerts for the device
        """
        return [
            alert for alert in self.alerts.values()
            if alert.device == device
        ]


class HealthMonitor:
    """Monitors disk and storage pool health."""
    
    def __init__(
        self,
        pool_manager: StoragePoolManager,
        volume_manager: VolumeManager,
        config_path: str = "/var/lib/prometheum/storage/health_config.json",
        data_path: str = "/var/lib/prometheum/storage/health_data",
        alert_manager: Optional[AlertManager] = None
    ):
        """Initialize the health monitor.
        
        Args:
            pool_manager: Storage pool manager
            volume_manager: Volume manager
            config_path: Path to health monitoring configuration
            data_path: Path to store health data
            alert_manager: Alert manager (creates one if None)
        """
        self.pool_manager = pool_manager
        self.volume_manager = volume_manager
        self.config_path = config_path
        self.data_path = data_path
        self.alert_manager = alert_manager or AlertManager()
        
        # Initialize configuration
        self.config = self._load_config()
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        
        # Storage for health data
        self.disk_health: Dict[str, DiskHealthData] = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = self.config.get("monitoring_interval", 3600)  # Default: 1 hour
        
        # Load existing health data
        self._load_health_data()
        
        logger.info("Health Monitor initialized")
    
    def _load_config(self) -> Dict:
        """Load health monitoring configuration."""
        default_config = {
            "monitoring_enabled": True,
            "monitoring_interval": 3600,  # 1 hour
            "smart_monitoring": {
                "enabled": True,
                "attributes_to_monitor": [
                    5,    # Reallocated Sectors Count
                    187,  # Reported Uncorrectable Errors
                    197,  # Current Pending Sector Count
                    198   # Offline Uncorrectable Sector Count
                ],
                "temperature_warning": 45,  # °C
                "temperature_critical": 55  # °C
            },
            "performance_monitoring": {
                "enabled": True,
                "metrics": [
                    "iops_read",
                    "iops_write", 
                    "throughput_read",
                    "throughput_write",
                    "latency_read",
                    "latency_write"
                ]
            },
            "alert_thresholds": {
                "disk_usage": 90,  # %
                "iops": 1000,      # IOPS
                "latency": 100     # ms
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                # Update with default values for any missing keys
                for section, values in default_config.items():
                    if isinstance(values, dict) and section in config:
                        for key, value in values.items():
                            if key not in config[section]:
                                config[section][key] = value
                    elif section not in config:
                        config[section] = values
                
                return config
            except Exception as e:
                logger.error(f"Error loading health config: {e}, using defaults")
                return default_config
        else:
            # Create default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default health config at {self.config_path}")
            return default_config
    
    def _save_health_data(self) -> None:
        """Save health data to disk."""
        for device, health_data in self.disk_health.items():
            data_file = os.path.join(self.data_path, f"{device}.json")
            try:
                with open(data_file, 'w') as f:
                    json.dump(health_data.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Error saving health data for {device}: {e}")
    
    def _load_health_data(self) -> None:
        """Load existing health data from disk."""
        if not os.path.exists(self.data_path):
            return
            
        for filename in os.listdir(self.data_path):
            if not filename.endswith(".json"):
                continue
                
            device = filename.replace(".json", "")
            data_file = os.path.join(self.data_path, filename)
            
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    
                # Create SmartData object if available
                smart_data = None
                if data.get("smart_data"):
                    smart_attrs = {}
                    for attr_id, attr_data in data["smart_data"].get("attributes", {}).items():
                        smart_attrs[int(attr_id)] = SmartAttribute(
                            id=int(attr_id),
                            name=attr_data["name"],
                            value=attr_data["value"],
                            worst=attr_data["worst"],
                            threshold=attr_data["threshold"],
                            raw_value=attr_data["raw_value"],
                            status=attr_data["status"]
                        )
                    
                    # Create SmartData object
                    sd = data["smart_data"]
                    smart_data = SmartData(
                        device=sd["device"],
                        model=sd["model"],
                        serial=sd["serial"],
                        firmware=sd["firmware"],
                        temperature=sd.get("temperature"),
                        power_on_hours=sd.get("power_on_hours"),
                        attributes=smart_attrs
                    )
                
                # Create DiskHealthData object
                self.disk_health[device] = DiskHealthData(
                    device=device,
                    smart_data=smart_data,
                    performance_metrics=data.get("performance_metrics", {}),
                    errors=data.get("errors", [])
                )
                
            except Exception as e:
                logger.error(f"Error loading health data for {device}: {e}")
    
    def start_monitoring(self) -> None:
        """Start the health monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already active")
            return
            
        if not self.config.get("monitoring_enabled", True):
            logger.info("Monitoring is disabled in configuration")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Health monitoring started with interval {self.monitoring_interval}s")
    
    def stop_monitoring(self) -> None:
        """Stop the health monitoring thread."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring not active")
            return
            
        self.monitoring_active = False
        self.monitoring_thread.join(timeout=10)
        if self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread did not terminate gracefully")
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring_active:
            try:
                # Collect all health data
                self.collect_all_health_data()
                
                # Save data to disk
                self._save_health_data()
                
                # Sleep until next collection
                for _ in range(self.monitoring_interval):
                    if not self.monitoring_active:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep a minute before retrying
    
    def collect_all_health_data(self) -> None:
        """Collect health data for all devices and pools."""
        # Get all devices from all pools
        all_devices = self._get_all_devices()
        
        for device in all_devices:
            try:
                # Collect SMART data if enabled
                if self.config.get("smart_monitoring", {}).get("enabled", True):
                    self.collect_smart_data(device)
                
                # Collect performance metrics if enabled
                if self.config.get("performance_monitoring", {}).get("enabled", True):
                    self.collect_performance_metrics(device)
                    
                # Check for errors in system logs
                self.check_device_errors(device)
                
            except Exception as e:
                logger.error(f"Error collecting health data for {device}: {e}")
                # Add error to device health data
                if device not in self.disk_health:
                    self.disk_health[device] = DiskHealthData(device)
                
                self.disk_health[device].errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Error collecting health data: {str(e)}",
                    "level": "error"
                })
        
        # Check pool health
        self.check_pools_health()
        
        # Generate alerts based on collected data
        self.generate_alerts()
    
    def _get_all_devices(self) -> List[str]:
        """Get a list of all physical devices in all pools."""
        all_devices = set()
        
        # Get devices from all pools
        for pool in self.pool_manager.list_pools():
            all_devices.update(pool.devices)
        
        # Add devices that might not be in pools yet
        try:
            # Run lsblk to get all block devices
            cmd = "lsblk -d -n -o NAME"
            result = run_command(cmd)
            
            # Add devices found by lsblk
            for line in result.stdout.splitlines():
                device = line.strip()
                if device and not device.startswith("loop"):
                    all_devices.add(device)
                    
        except Exception as e:
            logger.error(f"Error getting block devices: {e}")
        
        return list(all_devices)
    
    def collect_smart_data(self, device: str) -> Optional[SmartData]:
        """Collect SMART data for a device.
        
        Args:
            device: Device name (e.g., "sda")
            
        Returns:
            SmartData object if successful, None otherwise
        """
        # Ensure device path
        if not device.startswith("/dev/"):
            device_path = f"/dev/{device}"
        else:
            device_path = device
            device = device.replace("/dev/", "")
        
        try:
            # Run smartctl to get device info
            cmd = f"smartctl -i -H {device_path}"
            info_result = run_command(cmd)
            
            # Extract basic info
            model = "Unknown"
            serial = "Unknown"
            firmware = "Unknown"
            
            for line in info_result.stdout.splitlines():
                if "Device Model" in line or "Product" in line or "Model Number" in line:
                    model = line.split(":", 1)[1].strip()
                elif "Serial Number" in line:
                    serial = line.split(":", 1)[1].strip()
                elif "Firmware Version" in line:
                    firmware = line.split(":", 1)[1].strip()
            
            # Run smartctl to get SMART attributes
            cmd = f"smartctl -A -H {device_path}"
            attr_result = run_command(cmd)
            
            # Extract temperature if available
            temperature = None
            power_on_hours = None
            
            # Parse SMART attributes
            attributes = {}
            capture_attributes = False
            
            for line in attr_result.stdout.splitlines():
                if "SMART Attributes Data Structure" in line:
                    capture_attributes = True
                    continue
                    
                if capture_attributes and line.strip() and "ID#" not in line:
                    # Parse attribute line
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 10:
                        try:
                            attr_id = int(parts[0])
                            name = parts[1]
                            value = int(parts[3])
                            worst = int(parts[4])
                            threshold = int(parts[5])
                            raw_value = parts[9]
                            status = "OK" if value > threshold else "FAILING"
                            
                            attributes[attr_id] = SmartAttribute(
                                id=attr_id,
                                name=name,
                                value=value,
                                worst=worst,
                                threshold=threshold,
                                raw_value=raw_value,
                                status=status
                            
                            # Extract temperature (usually attribute 194)
                            if attr_id == 194 and temperature is None:
                                try:
                                    temperature = float(raw_value.split()[0])
                                except (ValueError, IndexError):
                                    pass
                                    
                            # Extract power-on hours (usually attribute 9)
                            if attr_id == 9 and power_on_hours is None:
                                try:
                                    power_on_hours = int(raw_value.split()[0])
                                except (ValueError, IndexError):
                                    pass
                                    
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing SMART attribute line: {line}, error: {e}")
            
            # Create SmartData object
            smart_data = SmartData(
                device=device,
                model=model,
                serial=serial,
                firmware=firmware,
                temperature=temperature,
                power_on_hours=power_on_hours,
                attributes=attributes
            )
            
            # Update disk health data
            if device not in self.disk_health:
                self.disk_health[device] = DiskHealthData(device)
                
            self.disk_health[device].smart_data = smart_data
            self.disk_health[device].last_updated = datetime.now().isoformat()
            
            # Check for critical temperature
            if temperature is not None:
                temp_critical = self.config.get("smart_monitoring", {}).get("temperature_critical", 55)
                temp_warning = self.config.get("smart_monitoring", {}).get("temperature_warning", 45)
                
                if temperature >= temp_critical:
                    self.alert_manager.add_alert(
                        level=AlertLevel.CRITICAL,
                        device=device,
                        message=f"Critical temperature: {temperature}°C",
                        details={"temperature": temperature, "threshold": temp_critical}
                    )
                elif temperature >= temp_warning:
                    self.alert_manager.add_alert(
                        level=AlertLevel.WARNING,
                        device=device,
                        message=f"High temperature: {temperature}°C",
                        details={"temperature": temperature, "threshold": temp_warning}
                    )
            
            # Check for critical SMART attributes
            attrs_to_monitor = self.config.get("smart_monitoring", {}).get("attributes_to_monitor", [5, 187, 197, 198])
            for attr_id in attrs_to_monitor:
                if attr_id in attributes and attributes[attr_id].is_failing():
                    attr = attributes[attr_id]
                    self.alert_manager.add_alert(
                        level=AlertLevel.CRITICAL,
                        device=device,
                        message=f"Critical SMART attribute: {attr.name}",
                        details={
                            "attribute_id": attr.id,
                            "attribute_name": attr.name,
                            "value": attr.value,
                            "threshold": attr.threshold,
                            "raw_value": attr.raw_value
                        }
                    )
            
            return smart_data
            
        except Exception as e:
            logger.error(f"Error collecting SMART data for {device}: {e}")
            return None
    
    def collect_performance_metrics(self, device: str) -> Dict[str, float]:
        """Collect performance metrics for a device.
        
        Args:
            device: Device name (e.g., "sda")
            
        Returns:
            Dictionary of metrics
        """
        # Ensure device path
        if not device.startswith("/dev/"):
            device_path = f"/dev/{device}"
        else:
            device_path = device
            device = device.replace("/dev/", "")
        
        metrics = {}
        
        try:
            # Use iostat to gather IO metrics
            cmd = f"iostat -x {device} 1 2"
            result = run_command(cmd)
            
            # Parse the second (latest) data entry from iostat
            lines = result.stdout.splitlines()
            device_found = False
            
            for i, line in enumerate(lines):
                # Check for device name in the line
                if device in line.split() and i < len(lines) - 1:
                    device_found = True
                    # Get metric line (next line after device name is found)
                    metric_line = lines[i + 1]
                    parts = metric_line.split()
                    
                    # Extract metrics if enough parts are present
                    if len(parts) >= 12:
                        try:
                            metrics["iops_read"] = float(parts[3])      # r/s
                            metrics["iops_write"] = float(parts[4])     # w/s
                            metrics["throughput_read"] = float(parts[5]) * 1024   # rkB/s (convert to bytes)
                            metrics["throughput_write"] = float(parts[6]) * 1024  # wkB/s (convert to bytes)
                            metrics["await"] = float(parts[9])          # await (ms)
                            metrics["util"] = float(parts[11])          # %util
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing iostat metrics: {e}")
                    break
                    
            # If device not found in iostat, use slower but more reliable methods
            if not device_found:
                logger.warning(f"Device {device} not found in iostat output, using alternative methods")
                
                # Get current time
                start_time = time.time()
                
                # Read current IO stats from /proc/diskstats
                with open("/proc/diskstats", "r") as f:
                    stats1 = None
                    for line in f:
                        if device in line:
                            stats1 = line.split()
                            break
                
                # Wait a second
                time.sleep(1)
                
                # Read stats again
                with open("/proc/diskstats", "r") as f:
                    stats2 = None
                    for line in f:
                        if device in line:
                            stats2 = line.split()
                            break
                
                # Calculate metrics
                if stats1 and stats2 and len(stats1) >= 14 and len(stats2) >= 14:
                    # Reads
                    reads_completed = int(stats2[3]) - int(stats1[3])
                    reads_sectors = int(stats2[5]) - int(stats1[5])
                    
                    # Writes
                    writes_completed = int(stats2[7]) - int(stats1[7])
                    writes_sectors = int(stats2[9]) - int(stats1[9])
                    
                    # Calculate metrics
                    metrics["iops_read"] = reads_completed
                    metrics["iops_write"] = writes_completed
                    metrics["throughput_read"] = reads_sectors * 512  # Sector size is usually 512 bytes
                    metrics["throughput_write"] = writes_sectors * 512
            
            # Update disk health data
            if device not in self.disk_health:
                self.disk_health[device] = DiskHealthData(device)
                
            self.disk_health[device].performance_metrics = metrics
            self.disk_health[device].last_updated = datetime.now().isoformat()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics for {device}: {e}")
            return {}
    
    def check_device_errors(self, device: str) -> List[Dict]:
        """Check for errors related to a device in system logs.
        
        Args:
            device: Device name (e.g., "sda")
            
        Returns:
            List of error dictionaries
        """
        errors = []
        
        try:
            # Use journalctl to search for errors
            cmd = f"journalctl -p err..crit -b --no-pager | grep {device}"
            result = run_command(cmd, check=False)  # Don't fail if no errors found
            
            if result.exit_code == 0:
                for line in result.stdout.splitlines():
                    errors.append({
                        "timestamp": datetime.now().isoformat(),
                        "source": "system_log",
                        "message": line.strip(),
                        "level": "error"
                    })
            
            # Check dmesg for errors
            cmd = f"dmesg | grep -i error | grep {device}"
            result = run_command(cmd, check=False)  # Don't fail if no errors found
            
            if result.exit_code == 0:
                for line in result.stdout.splitlines():
                    errors.append({
                        "timestamp": datetime.now().isoformat(),
                        "source": "dmesg",
                        "message": line.strip(),
                        "level": "error"
                    })
            
            # Update disk health data
            if device not in self.disk_health:
                self.disk_health[device] = DiskHealthData(device)
                
            # Add new errors to the list
            for error in errors:
                self.disk_health[device].errors.append(error)
                
                # Generate alert for each error
                self.alert_manager.add_alert(
                    level=AlertLevel.ERROR,
                    device=device,
                    message=f"Device error detected: {error['message'][:100]}...",
                    details=error
                )
            
            return errors
            
        except Exception as e:
            logger.error(f"Error checking for device errors for {device}: {e}")
            return []
    
    def check_pools_health(self) -> Dict[str, HealthStatus]:
        """Check the health status of all storage pools.
        
        Returns:
            Dictionary mapping pool names to health status
        """
        pool_health = {}
        
        for pool in self.pool_manager.list_pools():
            try:
                # Get pool status
                status = self.pool_manager.update_pool_status(pool.name)
                
                # Determine health status
                if "state" in status:
                    if status["state"].upper() in ["ONLINE", "NORMAL"]:
                        health = HealthStatus.GOOD
                    elif status["state"].upper() in ["DEGRADED"]:
                        health = HealthStatus.WARNING
                    elif status["state"].upper() in ["FAULTED", "OFFLINE", "UNAVAIL"]:
                        health = HealthStatus.CRITICAL
                    else:
                        health = HealthStatus.UNKNOWN
                else:
                    health = HealthStatus.UNKNOWN
                
                # Store pool health
                pool_health[pool.name] = health
                
                # Generate alerts for degraded/faulted pools
                if health == HealthStatus.WARNING:
                    self.alert_manager.add_alert(
                        level=AlertLevel.WARNING,
                        device=f"pool:{pool.name}",
                        message=f"Storage pool '{pool.name}' is in degraded state",
                        details=status
                    )
                elif health == HealthStatus.CRITICAL:
                    self.alert_manager.add_alert(
                        level=AlertLevel.CRITICAL,
                        device=f"pool:{pool.name}",
                        message=f"Storage pool '{pool.name}' is in critical state: {status.get('state', 'UNKNOWN')}",
                        details=status
                    )
                
            except Exception as e:
                logger.error(f"Error checking health for pool {pool.name}: {e}")
                pool_health[pool.name] = HealthStatus.UNKNOWN
        
        return pool_health
    
    def generate_alerts(self) -> None:
        """Generate alerts based on collected health data."""
        # Check each device for health issues
        for device, health_data in self.disk_health.items():
            # Skip if we've already processed alerts for this device
            if not hasattr(health_data, 'last_alert_check') or \
               health_data.last_alert_check != health_data.last_updated:
                
                # Check overall health status
                status = health_data.get_health_status()
                if status == HealthStatus.CRITICAL:
                    self.alert_manager.add_alert(
                        level=AlertLevel.CRITICAL,
                        device=device,
                        message=f"Device {device} is in CRITICAL health state",
                        details={"health_status": status.value}
                    )
                elif status == HealthStatus.WARNING:
                    self.alert_manager.add_alert(
                        level=AlertLevel.WARNING,
                        device=device,
                        message=f"Device {device} is in WARNING health state",
                        details={"health_status": status.value}
                    )
                
                # Check performance thresholds
                if health_data.performance_metrics:
                    metrics = health_data.performance_metrics
                    
                    # Check I/O performance thresholds
                    iops_threshold = self.config.get("alert_thresholds", {}).get("iops", 1000)
                    latency_threshold = self.config.get("alert_thresholds", {}).get("latency", 100)
                    
                    # Check IOPS
                    total_iops = metrics.get("iops_read", 0) + metrics.get("iops_write", 0)
                    if total_iops > iops_threshold:
                        self.alert_manager.add_alert(
                            level=AlertLevel.WARNING,
                            device=device,
                            message=f"High IOPS on device {device}: {total_iops} IOPS",
                            details={
                                "iops_total": total_iops,
                                "iops_read": metrics.get("iops_read", 0),
                                "iops_write": metrics.get("iops_write", 0),
                                "threshold": iops_threshold
                            }
                        )
                    
                    # Check latency
                    latency = metrics.get("await", 0)
                    if latency > latency_threshold:
                        self.alert_manager.add_alert(
                            level=AlertLevel.WARNING,
                            device=device,
                            message=f"High latency on device {device}: {latency} ms",
                            details={
                                "latency": latency,
                                "threshold": latency_threshold
                            }
                        )
                    
                    # Check disk utilization
                    util = metrics.get("util", 0)
                    if util > 90:  # 90% utilization
                        self.alert_manager.add_alert(
                            level=AlertLevel.WARNING,
                            device=device,
                            message=f"High disk utilization on device {device}: {util}%",
                            details={
                                "utilization": util,
                                "threshold": 90
                            }
                        )
                
                # Check disk usage thresholds
                try:
                    # Find mountpoint for this device
                    cmd = f"findmnt -S /dev/{device} -n -o TARGET"
                    result = run_command(cmd, check=False)
                    
                    if result.exit_code == 0 and result.stdout.strip():
                        mountpoint = result.stdout.strip()
                        
                        # Check disk usage
                        cmd = f"df -h {mountpoint}"
                        df_result = run_command(cmd)
                        
                        # Parse output - looking for usage percentage
                        lines = df_result.stdout.strip().split('\n')
                        if len(lines) >= 2:  # Header + data line
                            parts = lines[1].split()
                            if len(parts) >= 5:  # Filesystem, Size, Used, Avail, Use%
                                usage_percent = int(parts[4].replace('%', ''))
                                disk_usage_threshold = self.config.get("alert_thresholds", {}).get("disk_usage", 90)
                                
                                if usage_percent >= disk_usage_threshold:
                                    self.alert_manager.add_alert(
                                        level=AlertLevel.WARNING,
                                        device=device,
                                        message=f"High disk usage on {mountpoint}: {usage_percent}%",
                                        details={
                                            "mountpoint": mountpoint,
                                            "usage_percent": usage_percent,
                                            "threshold": disk_usage_threshold
                                        }
                                    )
                                    
                                # Even more critical if usage is extremely high
                                if usage_percent >= 95:
                                    self.alert_manager.add_alert(
                                        level=AlertLevel.CRITICAL,
                                        device=device,
                                        message=f"Critical disk usage on {mountpoint}: {usage_percent}%",
                                        details={
                                            "mountpoint": mountpoint,
                                            "usage_percent": usage_percent,
                                            "threshold": 95
                                        }
                                    )
                except Exception as e:
                    logger.warning(f"Error checking disk usage for {device}: {e}")
                
                # Mark that we've checked alerts for this update
                health_data.last_alert_check = health_data.last_updated
    
    def get_device_health(self, device: str) -> Dict:
        """Get health data for a specific device.
        
        Args:
            device: Device name
            
        Returns:
            Dictionary with health data
        """
        if device not in self.disk_health:
            return {"error": f"No health data available for device {device}"}
            
        return self.disk_health[device].to_dict()
    
    def get_all_health_data(self) -> Dict:
        """Get health data for all devices.
        
        Returns:
            Dictionary mapping device names to health data
        """
        return {
            device: health_data.to_dict()
            for device, health_data in self.disk_health.items()
        }
    
    def get_alerts(self, include_resolved: bool = False) -> List[Dict]:
        """Get all active alerts.
        
        Args:
            include_resolved: Whether to include resolved alerts
            
        Returns:
            List of alert dictionaries
        """
        if include_resolved:
            alerts = self.alert_manager.alerts.values()
        else:
            alerts = self.alert_manager.get_active_alerts()
            
        return [alert.to_dict() for alert in alerts]
    
    def get_alerts_for_device(self, device: str) -> List[Dict]:
        """Get alerts for a specific device.
        
        Args:
            device: Device name
            
        Returns:
            List of alert dictionaries
        """
        alerts = self.alert_manager.get_alerts_for_device(device)
        return [alert.to_dict() for alert in alerts]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        return self.alert_manager.acknowledge_alert(alert_id)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if successful, False otherwise
        """
        return self.alert_manager.resolve_alert(alert_id)
    
    def manual_check(self, device: str = None) -> Dict:
        """Manually run a health check.
        
        Args:
            device: Optional device to check (checks all if None)
            
        Returns:
            Dictionary with check results
        """
        if device:
            # Check single device
            try:
                # Collect SMART data
                smart_data = self.collect_smart_data(device)
                
                # Collect performance metrics
                metrics = self.collect_performance_metrics(device)
                
                # Check for errors
                errors = self.check_device_errors(device)
                
                # Save data
                self._save_health_data()
                
                # Generate alerts
                self.generate_alerts()
                
                return {
                    "device": device,
                    "smart_data": smart_data.to_dict() if smart_data else None,
                    "performance_metrics": metrics,
                    "errors": errors,
                    "health_status": self.disk_health[device].get_health_status().value if device in self.disk_health else "unknown"
                }
            except Exception as e:
                return {"error": f"Error checking {device}: {str(e)}"}
        else:
            # Check all devices
            try:
                self.collect_all_health_data()
                return {
                    "devices_checked": list(self.disk_health.keys()),
                    "health_summary": {
                        device: health_data.get_health_status().value
                        for device, health_data in self.disk_health.items()
                    },
                    "alerts": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
                }
            except Exception as e:
                return {"error": f"Error checking all devices: {str(e)}"}
    
    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a handler function for new alerts.
        
        The handler function will be called with each new alert as it's generated.
        
        Args:
            handler: Function that takes an Alert object
        """
        self.alert_manager.alert_handlers.append(handler)
        logger.info(f"Registered new alert handler: {handler.__name__}")
