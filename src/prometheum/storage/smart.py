"""
SMART data collection and analysis module.

This module provides functionality for collecting and analyzing SMART data
from storage devices using smartmontools.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class SmartAttribute:
    """Represents a SMART attribute with its values and thresholds."""
    
    # Critical SMART attributes to monitor
    CRITICAL_ATTRIBUTES = {
        5: "Reallocated_Sector_Ct",
        187: "Reported_Uncorrect",
        188: "Command_Timeout",
        197: "Current_Pending_Sector",
        198: "Offline_Uncorrectable"
    }
    
    # Warning thresholds for temperature (in Celsius)
    TEMP_WARNING = 55
    TEMP_CRITICAL = 65

    def __init__(self, id: int, name: str, value: int, worst: int, 
                 thresh: int, raw_value: int):
        self.id = id
        self.name = name
        self.value = value
        self.worst = worst
        self.thresh = thresh
        self.raw_value = raw_value
        self.status = self._evaluate_status()

    def _evaluate_status(self) -> str:
        """Evaluate the attribute status based on thresholds."""
        if self.value <= self.thresh:
            return "failed"
        elif self.value <= self.worst:
            return "warning"
        return "healthy"

    def to_dict(self) -> Dict[str, Any]:
        """Convert attribute to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "worst": self.worst,
            "thresh": self.thresh,
            "raw_value": self.raw_value,
            "status": self.status
        }

class SmartData:
    """Container for device SMART data."""
    
    def __init__(self, device: str):
        self.device = device
        self.attributes: Dict[int, SmartAttribute] = {}
        self.temperature: Optional[int] = None
        self.power_on_hours: Optional[int] = None
        self.overall_health: str = "unknown"
        self.last_update = datetime.now()
        self.collection_errors: List[str] = []

    def add_attribute(self, attr: SmartAttribute) -> None:
        """Add a SMART attribute to the collection."""
        self.attributes[attr.id] = attr

    def evaluate_health(self) -> str:
        """Evaluate overall device health based on SMART data."""
        if self.collection_errors:
            return "error"
        
        # Check critical attributes
        for attr_id in SmartAttribute.CRITICAL_ATTRIBUTES:
            if attr_id in self.attributes:
                attr = self.attributes[attr_id]
                if attr.status == "failed":
                    return "failed"
                elif attr.status == "warning":
                    return "warning"
        
        # Check temperature
        if self.temperature:
            if self.temperature >= SmartAttribute.TEMP_CRITICAL:
                return "failed"
            elif self.temperature >= SmartAttribute.TEMP_WARNING:
                return "warning"
        
        return "healthy"

    def to_dict(self) -> Dict[str, Any]:
        """Convert SMART data to dictionary."""
        return {
            "device": self.device,
            "overall_health": self.overall_health,
            "temperature": self.temperature,
            "power_on_hours": self.power_on_hours,
            "attributes": {
                attr_id: attr.to_dict() 
                for attr_id, attr in self.attributes.items()
            },
            "last_update": self.last_update.isoformat(),
            "errors": self.collection_errors
        }

async def get_smart_data(device: str) -> SmartData:
    """Collect SMART data for a device using smartctl."""
    smart_data = SmartData(device)
    
    try:
        # Get SMART data in JSON format
        cmd = ["smartctl", "-a", "-j", f"/dev/{device}"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            smart_data.collection_errors.append(f"smartctl error: {error_msg}")
            return smart_data
        
        # Parse JSON output
        data = json.loads(stdout.decode())
        
        # Get basic information
        smart_data.overall_health = ("healthy" 
            if data.get("smart_status", {}).get("passed") 
            else "failed")
        
        # Get temperature
        temp = data.get("temperature", {}).get("current")
        if temp is not None:
            smart_data.temperature = int(temp)
        
        # Get power-on hours
        smart_data.power_on_hours = data.get("power_on_time", {}).get("hours")
        
        # Process SMART attributes
        for attr in data.get("ata_smart_attributes", {}).get("table", []):
            smart_attr = SmartAttribute(
                id=attr["id"],
                name=attr["name"],
                value=attr["value"],
                worst=attr["worst"],
                thresh=attr["thresh"],
                raw_value=int(attr["raw"]["value"])
            )
            smart_data.add_attribute(smart_attr)
        
    except Exception as e:
        logger.error(f"Error collecting SMART data for {device}: {e}")
        smart_data.collection_errors.append(str(e))
    
    return smart_data

async def is_device_supported(device: str) -> bool:
    """Check if a device supports SMART data collection."""
    try:
        cmd = ["smartctl", "-i", f"/dev/{device}"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        output = stdout.decode()
        return "SMART support is:" in output and "Unavailable" not in output
    except Exception as e:
        logger.error(f"Error checking SMART support for {device}: {e}")
        return False

def parse_smart_thresholds(raw_thresh: str) -> Dict[int, int]:
    """Parse SMART threshold values from smartctl output."""
    thresholds = {}
    for line in raw_thresh.splitlines():
        match = re.match(r"^\s*(\d+)\s+\w+\s+\d+\s+\d+\s+(\d+)", line)
        if match:
            attr_id, thresh = match.groups()
            thresholds[int(attr_id)] = int(thresh)
    return thresholds

