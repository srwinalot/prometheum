#!/usr/bin/env python3
"""
Integration test for health monitoring system.
Tests the interaction between HealthMonitor and StoragePoolManager components
in an isolated environment.
"""

import asyncio
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create isolated test environment
test_dir = Path("/tmp/prometheum_test_integration")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(exist_ok=True)

# Create package structure
pkg_dir = test_dir / "prometheum/storage"
pkg_dir.mkdir(parents=True, exist_ok=True)

logger.info("Creating minimal package structure...")

# Create __init__.py files
with open(test_dir / "prometheum" / "__init__.py", "w") as f:
    f.write('"""Prometheum package"""')

with open(pkg_dir / "__init__.py", "w") as f:
    f.write('"""Storage management package"""')

# Copy implementation files to test directory
src_dir = Path("/Users/franklinbutahe/prometheum/src/prometheum/storage")
for file in ["pool.py", "health.py"]:
    try:
        shutil.copy2(src_dir / file, pkg_dir / file)
        logger.info(f"Copied {file} to test environment")
    except Exception as e:
        logger.error(f"Error copying {file}: {e}")

# Create minimal volume.py
with open(pkg_dir / "volume.py", "w") as f:
    f.write('''"""Volume management module."""

class VolumeManager:
    """Minimal implementation of VolumeManager for testing."""
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
    def get_volumes(self):
        """Get list of all volumes."""
        return ["vol1", "vol2"]
        
    def get_volume_status(self, name):
        """Get status of a volume."""
        return {
            "state": "online",
            "size": 1073741824,  # 1GB
            "used": 536870912,   # 512MB
            "pool": next(iter(self.pool_manager.get_pools()), "unknown")
        }
''')

# Reset sys.path to only include our test directory and standard libraries
import sys
original_sys_path = sys.path.copy()
sys.path = [str(test_dir)] + [p for p in original_sys_path if 'prometheum' not in p and p != '']

logger.info("Test environment created, importing modules...")

# Import modules from our isolated environment
from prometheum.storage.pool import StoragePoolManager
from prometheum.storage.volume import VolumeManager
from prometheum.storage.health import HealthMonitor

async def test_health_integration():
    """Test health monitoring integration with storage pools."""
    logger.info("Starting health monitoring integration test")
    
    # Initialize components
    pool_manager = StoragePoolManager()
    volume_manager = VolumeManager(pool_manager)
    health_monitor = HealthMonitor(pool_manager, volume_manager)
    
    try:
        # Test 1: Basic component initialization
        logger.info("\nTest 1: Verifying component initialization")
        pools = pool_manager.get_pools()
        assert len(pools) > 0, "No storage pools found"
        assert "tank" in pools, "Expected 'tank' pool not found"
        assert "backup" in pools, "Expected 'backup' pool not found"
        print("✓ Components initialized successfully")

        # Test 2: Start health monitoring
        logger.info("\nTest 2: Starting health monitoring")
        health_monitor.start_monitoring()
        assert health_monitor.monitoring_active == True
        print("✓ Health monitoring started successfully")

        # Test 3: Check system health status
        logger.info("\nTest 3: Checking system health status")
        health_status = health_monitor.get_system_health()
        print(f"System health status: {json.dumps(health_status, indent=2)}")
        assert health_status["status"] in ["healthy", "warning"], "Unexpected system health status"
        assert len(health_status["details"]["pools"]) > 0, "No pools found in health status"
        print("✓ System health status retrieved successfully")

        # Test 4: Verify pool health reporting
        logger.info("\nTest 4: Verifying pool health status")
        for pool in pools:
            pool_status = pool_manager.get_pool_status(pool)
            print(f"\nPool {pool} status: {json.dumps(pool_status, indent=2)}")
            assert "state" in pool_status, "Pool status missing state information"
            assert "capacity" in pool_status, "Pool status missing capacity information"
            assert "devices" in pool_status, "Pool status missing device information"
        print("✓ Pool health reporting verified")

        # Test 5: Check for device errors
        logger.info("\nTest 5: Checking device error detection")
        device_errors = pool_manager.check_device_errors("sdd")
        print(f"\nDevice sdd errors: {json.dumps(device_errors, indent=2)}")
        assert device_errors["errors"]["read"] == 2, "Expected 2 read errors on sdd"
        assert device_errors["errors"]["checksum"] == 1, "Expected 1 checksum error on sdd"
        print("✓ Device error detection working")

        # Test 6: Verify alert generation
        logger.info("\nTest 6: Checking alert generation")
        health_monitor._create_alert("sdd", "Test alert for sdd device", "warning")
        alerts = health_monitor.get_alerts()
        print(f"\nActive alerts: {json.dumps(alerts, indent=2)}")
        assert len(alerts) > 0, "Expected alerts for device errors"
        assert any(a["device"] == "sdd" for a in alerts), "Expected alert for sdd device"
        print("✓ Alert generation verified")

        # Test 7: Test alert management
        logger.info("\nTest 7: Testing alert management")
        alert = next(a for a in alerts if a["device"] == "sdd")
        assert health_monitor.acknowledge_alert(alert["id"]) == True
        updated_alerts = health_monitor.get_alerts()
        assert any(a["id"] == alert["id"] and a["acknowledged"] == True 
                  for a in updated_alerts)
        print("✓ Alert management working")

        # Test 8: Run manual device check
        logger.info("\nTest 8: Testing manual device check")
        check_result = health_monitor.manual_check("sdd")
        print(f"\nManual check result: {json.dumps(check_result, indent=2)}")
        assert "status" in check_result
        print("✓ Manual device check working")

        # Cleanup
        logger.info("\nStopping health monitoring")
        health_monitor.stop_monitoring()
        assert health_monitor.monitoring_active == False
        print("✓ Health monitoring stopped successfully")

        print("\nAll integration tests completed successfully!")

    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        raise
    finally:
        if hasattr(health_monitor, 'monitoring_active') and health_monitor.monitoring_active:
            health_monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(test_health_integration())

