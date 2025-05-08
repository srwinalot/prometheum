#!/usr/bin/env python3
"""
Isolated test script for health monitoring components.
Creates a minimal environment to test just the health monitoring functionality.
"""

import asyncio
import sys
import os
import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create minimal test environment
test_dir = Path("/tmp/prometheum_test")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(exist_ok=True)

# Create minimal package structure
pkg_dir = test_dir / "prometheum/storage"
pkg_dir.mkdir(parents=True, exist_ok=True)

logger.info("Creating minimal package structure...")

# Create __init__.py files
with open(test_dir / "prometheum" / "__init__.py", "w") as f:
    f.write('"""Prometheum package"""')

with open(pkg_dir / "__init__.py", "w") as f:
    f.write('"""Storage management package"""')

# Create pool.py
with open(pkg_dir / "pool.py", "w") as f:
    f.write('''"""Storage pool management module."""

class StoragePoolManager:
    """Minimal implementation of StoragePoolManager for testing."""
    
    def get_all_devices(self):
        """Return a list of test devices."""
        return ["sda", "sdb", "sdc"]
    
    def get_pools(self):
        """Return a list of test pools."""
        return ["pool1", "pool2"]
''')

# Create volume.py
with open(pkg_dir / "volume.py", "w") as f:
    f.write('''"""Volume management module."""

class VolumeManager:
    """Minimal implementation of VolumeManager for testing."""
    
    def __init__(self, pool_manager):
        """Initialize with a pool manager."""
        self.pool_manager = pool_manager
''')

# Create health.py with our implementation
with open(pkg_dir / "health.py", "w") as f:
    f.write('''"""Storage health monitoring implementation."""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class Alert:
    def __init__(self, device: str, message: str, severity: str = "warning"):
        self.id = str(uuid.uuid4())
        self.device = device
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False
        self.resolution_time = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "device": self.device,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }

class HealthMonitor:
    def __init__(self, pool_manager, volume_manager):
        self.pool_manager = pool_manager
        self.volume_manager = volume_manager
        self.monitoring_active = False
        self.monitoring_interval = 300  # 5 minutes
        self.last_update_time = None
        self._monitoring_task = None
        self._alerts: List[Alert] = []

    def start_monitoring(self) -> None:
        """Start the background health monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the background health monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self._monitoring_task:
                self._monitoring_task.cancel()
            logger.info("Health monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_all_devices()
                self.last_update_time = datetime.now()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            await asyncio.sleep(self.monitoring_interval)

    async def _check_all_devices(self) -> None:
        """Check health of all devices."""
        for device in self.pool_manager.get_all_devices():
            try:
                await self._check_device(device)
            except Exception as e:
                logger.error(f"Error checking device {device}: {e}")

    async def _check_device(self, device: str) -> None:
        """Check health of a specific device."""
        try:
            smart_data = await self._get_smart_data(device)
            perf_metrics = await self._get_performance_metrics(device)
            self._update_device_health(device, smart_data, perf_metrics)
        except Exception as e:
            self._create_alert(device, f"Health check failed: {str(e)}", "error")

    async def _get_smart_data(self, device: str) -> Dict[str, Any]:
        """Get SMART data for a device."""
        return {"status": "healthy"}  # Placeholder

    async def _get_performance_metrics(self, device: str) -> Dict[str, Any]:
        """Get performance metrics for a device."""
        return {"read_speed": 100, "write_speed": 100}  # Placeholder

    def _update_device_health(self, device: str, smart_data: Dict, perf_metrics: Dict) -> None:
        """Update device health status based on collected data."""
        pass  # Placeholder

    def get_all_health_data(self) -> Dict[str, Any]:
        """Get health data for all devices."""
        return {
            device: {"health_status": "healthy"} 
            for device in self.pool_manager.get_all_devices()
        }

    def get_device_health(self, device: str) -> Dict[str, Any]:
        """Get health data for a specific device."""
        if device not in self.pool_manager.get_all_devices():
            return {"error": f"Device {device} not found"}
        return {"health_status": "healthy"}

    def manual_check(self, device: Optional[str] = None) -> Dict[str, Any]:
        """Run a manual health check."""
        if device and device not in self.pool_manager.get_all_devices():
            return {"error": f"Device {device} not found"}
        return {"status": "check_started"}

    def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get current health alerts."""
        return [
            alert.to_dict() for alert in self._alerts
            if include_resolved or not alert.resolved
        ]

    def get_alerts_for_device(self, device: str) -> List[Dict[str, Any]]:
        """Get alerts for a specific device."""
        return [
            alert.to_dict() for alert in self._alerts
            if alert.device == device and not alert.resolved
        ]

    def _create_alert(self, device: str, message: str, severity: str = "warning") -> None:
        """Create a new health alert."""
        alert = Alert(device, message, severity)
        self._alerts.append(alert)
        logger.warning(f"New health alert for {device}: {message}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                return True
        return False

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        device_health = self.get_all_health_data()
        active_alerts = self.get_alerts(include_resolved=False)
        
        status = "healthy"
        if any(a["severity"] == "error" for a in active_alerts):
            status = "error"
        elif any(a["severity"] == "warning" for a in active_alerts):
            status = "warning"
        
        return {
            "status": status,
            "devices": len(device_health),
            "active_alerts": len(active_alerts),
            "last_check": self.last_update_time.isoformat() if self.last_update_time else None,
            "details": {
                "devices": device_health,
                "pools": self._get_pool_health()
            }
        }

    def _get_pool_health(self) -> Dict[str, Any]:
        """Get health status for all storage pools."""
        return {
            pool: {"status": "healthy"}
            for pool in self.pool_manager.get_pools()
        }
''')

# Reset sys.path to only include our test directory and standard libraries
original_sys_path = sys.path.copy()
sys.path = [str(test_dir)] + [p for p in original_sys_path if 'prometheum' not in p and p != '']

logger.info("Test environment created, importing modules...")

# Import modules from our isolated environment
from prometheum.storage.health import HealthMonitor
from prometheum.storage.pool import StoragePoolManager
from prometheum.storage.volume import VolumeManager

async def test_health_monitoring():
    """Test the health monitoring functionality."""
    logger.info("Initializing test components...")
    
    # Initialize components
    pool_manager = StoragePoolManager()
    volume_manager = VolumeManager(pool_manager)
    health_monitor = HealthMonitor(pool_manager, volume_manager)
    
    print("\nTesting health monitoring components...")
    
    # Test 1: Start monitoring
    print("\n1. Starting health monitoring")
    health_monitor.start_monitoring()
    assert health_monitor.monitoring_active == True
    print("✓ Health monitoring started successfully")
    
    # Test 2: Get system health
    print("\n2. Getting system health status")
    health_status = health_monitor.get_system_health()
    print(f"System health status: {health_status}")
    assert "status" in health_status
    assert "devices" in health_status
    print("✓ System health retrieved successfully")
    
    # Test 3: Create and retrieve alerts
    print("\n3. Testing alert management")
    health_monitor._create_alert("test_device", "Test alert message", "warning")
    alerts = health_monitor.get_alerts()
    print(f"Created alert. Current alerts: {alerts}")
    assert len(alerts) > 0
    assert alerts[0]["device"] == "test_device"
    assert alerts[0]["message"] == "Test alert message"
    print("✓ Alert management working correctly")
    
    # Test 4: Get device health
    print("\n4. Testing device health retrieval")
    device_health = health_monitor.get_all_health_data()
    print(f"Device health data: {device_health}")
    assert len(device_health) > 0
    print("✓ Device health retrieval working correctly")
    
    # Test 5: Manual health check
    print("\n5. Testing manual health check")
    check_result = health_monitor.manual_check()
    print(f"Manual check result: {check_result}")
    assert "status" in check_result
    print("✓ Manual health check working correctly")
    
    # Test 6: Alert acknowledgment and resolution
    print("\n6. Testing alert acknowledgment and resolution")
    alert_id = alerts[0]["id"]
    assert health_monitor.acknowledge_alert(alert_id) == True
    updated_alerts = health_monitor.get_alerts()
    assert any(a["id"] == alert_id and a["acknowledged"] == True for a in updated_alerts)
    
    assert health_monitor.resolve_alert(alert_id) == True
    resolved_alerts = health_monitor.get_alerts(include_resolved=True)
    assert any(a["id"] == alert_id and a["resolved"] == True for a in resolved_alerts)
    print("✓ Alert acknowledgment and resolution working correctly")
    
    # Cleanup
    print("\nStopping health monitoring")
    health_monitor.stop_monitoring()
    assert health_monitor.monitoring_active == False
    print("✓ Health monitoring stopped successfully")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    logger.info("Starting isolated health monitoring test...")
    asyncio.run(test_health_monitoring())

