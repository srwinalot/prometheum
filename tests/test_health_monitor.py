#!/usr/bin/env python3
"""
Test script for health monitoring components.
Just tests the core functionality without full API integration.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from prometheum.storage.health import HealthMonitor
from prometheum.storage.pool import StoragePoolManager
from prometheum.storage.volume import VolumeManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock classes for testing if actual implementations aren't fully functional yet
class MockStoragePoolManager:
    def get_all_devices(self):
        return ["sda", "sdb", "sdc"]
    
    def get_pools(self):
        return ["pool1", "pool2"]

class MockVolumeManager:
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager or MockStoragePoolManager()
        
async def test_health_monitoring():
    """Test the health monitoring functionality."""
    logger.info("Initializing test components...")
    
    # Use mock classes if the actual ones aren't fully implemented
    try:
        pool_manager = StoragePoolManager()
        volume_manager = VolumeManager(pool_manager)
    except Exception as e:
        logger.warning(f"Could not initialize actual components: {e}")
        logger.info("Using mock components instead")
        pool_manager = MockStoragePoolManager()
        volume_manager = MockVolumeManager(pool_manager)
    
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
    asyncio.run(test_health_monitoring())

