#!/usr/bin/env python3
"""Integration tests for health monitoring system."""

import asyncio
import logging
import json
import subprocess
import sys
from datetime import datetime
import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules from test environment
from prometheum.storage.pool import StoragePoolManager
from prometheum.storage.volume import VolumeManager
from prometheum.storage.health import HealthMonitor
from prometheum.storage.history import HealthHistoryDB

def check_required_tools():
    """Check if required monitoring tools are available."""
    tools = {
        "smartctl": "SMART monitoring",
        "iostat": "Performance monitoring"
    }
    
    missing = []
    for tool, purpose in tools.items():
        try:
            result = subprocess.run(["which", tool], capture_output=True)
            if result.returncode != 0:
                missing.append(f"{tool} (required for {purpose})")
        except Exception:
            missing.append(f"{tool} (required for {purpose})")
    
    if missing:
        logger.warning("Some monitoring tools are not available:")
        for tool in missing:
            logger.warning(f"- {tool}")
        logger.warning("Tests will run with simulated data")
        return False
    return True

async def test_health_monitoring_integrated():
    """Test the complete health monitoring system."""
    logger.info("Starting comprehensive health monitoring test")
    
    # Check for required tools
    has_tools = check_required_tools()
    if not has_tools:
        logger.info("Running tests with simulated data")
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db') as temp_db:
        # Initialize components
        pool_manager = StoragePoolManager()
        volume_manager = VolumeManager(pool_manager)
        health_monitor = HealthMonitor(
            pool_manager=pool_manager,
            volume_manager=volume_manager,
            data_path=os.path.dirname(temp_db.name),
            config_path=os.path.join(os.path.dirname(temp_db.name), "config.json")
        )
        
        # Initialize history database
        health_monitor.history_db = HealthHistoryDB(db_path=temp_db.name)
        
        try:
            # Test 1: Basic initialization
            logger.info("\nTest 1: Verifying component initialization")
            assert health_monitor is not None
            assert health_monitor.history_db is not None
            print("✓ Components initialized successfully")

            # Test 2: Start monitoring and collect initial data
            logger.info("\nTest 2: Starting health monitoring")
            health_monitor.start_monitoring()
            assert health_monitor.monitoring_active == True
            await asyncio.sleep(2)  # Wait for first check
            print("✓ Health monitoring started successfully")

            # Test 3: Check system health status
            logger.info("\nTest 3: Creating simulated health data")
            # Add a simulated device
            device = "sdd"
            health_monitor.disk_health[device] = {}
            print("✓ Added simulated device")

            # Test 4: Create and verify alert
            logger.info("\nTest 4: Testing alert system")
            test_alert_id = await health_monitor.add_alert(
                level="warning",
                device=device,
                message="Test alert for database"
            )
            alerts = health_monitor.get_alerts()
            print(f"\nActive alerts: {json.dumps(alerts, indent=2)}")
            assert len(alerts) > 0
            alert_id = alerts[0]["id"]
            print("✓ Alert system working")

            # Test 5: Alert lifecycle
            logger.info("\nTest 5: Testing alert lifecycle")
            assert health_monitor.acknowledge_alert(alert_id) == True
            updated_alerts = health_monitor.get_alerts()
            assert any(a["id"] == alert_id and a["acknowledged"] == True 
                    for a in updated_alerts)
            
            assert health_monitor.resolve_alert(alert_id) == True
            resolved_alerts = health_monitor.get_alerts(include_resolved=True)
            assert any(a["id"] == alert_id and a["resolved"] == True 
                    for a in resolved_alerts)
            print("✓ Alert lifecycle working")

            # Test 6: History storage
            logger.info("\nTest 6: Testing database operations")
            # Create another test alert for the database
            alert_data = {
                "device": "sdd",
                "message": "Direct database alert test",
                "severity": "warning",
                "id": f"test_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat()
            }
            await health_monitor.history_db.store_alert(alert_data)
            
            # Verify alert was stored
            stored_alerts = await health_monitor.history_db.get_alerts(device="sdd")
            print(f"\nStored alerts: {json.dumps(stored_alerts, indent=2)}")
            assert any(a["id"] == alert_data["id"] for a in stored_alerts)
            print("✓ Database alert storage working")
            
            # Test 7: Device health summary
            logger.info("\nTest 7: Testing health summary")
            # Add SMART data to test health summary
            smart_data = {
                "device": "sdd",
                "model": "Test Drive",
                "serial": "TEST123",
                "firmware": "1.0",
                "temperature": 35.5,
                "power_on_hours": 1000,
                "health_status": "good",
                "attributes": "{}",
                "timestamp": datetime.now().isoformat()
            }
            await health_monitor.history_db.store_smart_data(smart_data)
            
            device_health = await health_monitor.history_db.get_device_health_summary("sdd")
            print(f"\nDevice health summary: {json.dumps(device_health, indent=2)}")
            assert "device" in device_health
            assert "current_health" in device_health
            print("✓ Health summary working")

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
            if health_monitor.monitoring_active:
                health_monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(test_health_monitoring_integrated())
