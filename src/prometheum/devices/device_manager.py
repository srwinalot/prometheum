"""
Device management for Prometheum.

This module handles device registration, authentication, and management
for the Prometheum personal cloud system.
"""

import os
import json
import uuid
import logging
import secrets
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from dataclasses import dataclass, asdict, field


@dataclass
class Device:
    """Represents a device registered with the Prometheum cloud."""
    
    device_id: str
    name: str
    type: str  # desktop, mobile, tablet, etc.
    os_type: str  # windows, macos, ios, android, linux
    owner: str
    public_key: str
    registered_date: datetime.datetime
    last_seen: datetime.datetime
    is_trusted: bool = True
    status: str = "active"  # active, suspended, revoked
    device_token: str = field(default_factory=lambda: secrets.token_hex(32))
    permissions: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['registered_date'] = self.registered_date.isoformat()
        data['last_seen'] = self.last_seen.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Device':
        """Create a Device instance from dictionary data."""
        # Convert ISO format strings to datetime objects
        reg_date = datetime.datetime.fromisoformat(data['registered_date'])
        last_seen = datetime.datetime.fromisoformat(data['last_seen'])
        
        return cls(
            device_id=data['device_id'],
            name=data['name'],
            type=data['type'],
            os_type=data['os_type'],
            owner=data['owner'],
            public_key=data['public_key'],
            registered_date=reg_date,
            last_seen=last_seen,
            is_trusted=data.get('is_trusted', True),
            status=data.get('status', 'active'),
            device_token=data.get('device_token', secrets.token_hex(32)),
            permissions=data.get('permissions', {})
        )


class DeviceManager:
    """
    Manages devices that can connect to the Prometheum cloud.
    
    This class handles device registration, authentication, status tracking,
    and access control to ensure only authorized devices can access the system.
    """
    
    def __init__(self, cloud_manager):
        """
        Initialize the device manager.
        
        Args:
            cloud_manager: Reference to the CloudManager instance
        """
        self.cloud_manager = cloud_manager
        self.logger = logging.getLogger(__name__)
        self.devices: Dict[str, Device] = {}
        self.devices_path = self.cloud_manager.storage_path / "devices"
    
    def initialize(self) -> bool:
        """
        Initialize the device management system.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Ensure devices directory exists
            self.devices_path.mkdir(exist_ok=True)
            
            # Load existing devices
            self._load_devices()
            
            # Check for expired devices
            self._check_inactive_devices()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize device manager: {str(e)}")
            return False
    
    def _load_devices(self) -> None:
        """Load all device data from storage."""
        try:
            for device_file in self.devices_path.glob("*.json"):
                try:
                    with open(device_file, "r") as f:
                        device_data = json.load(f)
                        device = Device.from_dict(device_data)
                        self.devices[device.device_id] = device
                except Exception as e:
                    self.logger.error(f"Error loading device from {device_file}: {str(e)}")
            
            self.logger.info(f"Loaded {len(self.devices)} devices")
        except Exception as e:
            self.logger.error(f"Failed to load devices: {str(e)}")
            # Continue with empty device list
    
    def _save_device(self, device: Device) -> bool:
        """
        Save device data to persistent storage.
        
        Args:
            device: The device to save
            
        Returns:
            bool: True if successful
        """
        try:
            device_file = self.devices_path / f"{device.device_id}.json"
            with open(device_file, "w") as f:
                json.dump(device.to_dict(), f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save device {device.device_id}: {str(e)}")
            return False
    
    def register_device(self, name: str, device_type: str, os_type: str, 
                       owner: str, public_key: str) -> Tuple[bool, Optional[Device], str]:
        """
        Register a new device with the system.
        
        Args:
            name: Human-readable device name
            device_type: Type of device (desktop, mobile, etc.)
            os_type: Operating system
            owner: Owner identifier
            public_key: Public key for device authentication
            
        Returns:
            Tuple containing:
            - bool: Success flag
            - Optional[Device]: The registered device if successful
            - str: Error message if not successful
        """
        try:
            # Check if max devices limit is reached
            max_devices = self.cloud_manager.config.get("devices.max_devices", 10)
            if len(self.devices) >= max_devices:
                return False, None, f"Maximum allowed devices ({max_devices}) reached"
            
            # Create new device ID
            device_id = str(uuid.uuid4())
            
            # Create device record
            now = datetime.datetime.now()
            device = Device(
                device_id=device_id,
                name=name,
                type=device_type,
                os_type=os_type,
                owner=owner,
                public_key=public_key,
                registered_date=now,
                last_seen=now,
                permissions={
                    "read": True,
                    "write": True,
                    "share": False,
                    "admin": False
                }
            )
            
            # Store device
            self.devices[device_id] = device
            success = self._save_device(device)
            
            if success:
                self.logger.info(f"Registered new device: {name} ({device_id})")
                return True, device, ""
            else:
                return False, None, "Failed to save device information"
                
        except Exception as e:
            error_msg = f"Device registration failed: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    def authenticate_device(self, device_id: str, device_token: str) -> bool:
        """
        Authenticate a device using its token.
        
        Args:
            device_id: The device ID
            device_token: The device token
            
        Returns:
            bool: True if authentication is successful
        """
        if device_id not in self.devices:
            self.logger.warning(f"Authentication attempt for unknown device: {device_id}")
            return False
        
        device = self.devices[device_id]
        
        if device.status != "active":
            self.logger.warning(f"Authentication attempt for {device.status} device: {device_id}")
            return False
            
        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(device.device_token, device_token):
            self.logger.warning(f"Failed authentication attempt for device: {device_id}")
            return False
        
        # Update last seen timestamp
        device.last_seen = datetime.datetime.now()
        self._save_device(device)
        
        return True
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """
        Get a device by its ID.
        
        Args:
            device_id: The device ID
            
        Returns:
            Optional[Device]: The device if found, None otherwise
        """
        return self.devices.get(device_id)
    
    def update_device(self, device_id: str, **kwargs) -> Tuple[bool, str]:
        """
        Update device properties.
        
        Args:
            device_id: The device ID
            **kwargs: Device properties to update
            
        Returns:
            Tuple[bool, str]: Success flag and error message if any
        """
        if device_id not in self.devices:
            return False, f"Device {device_id} not found"
        
        device = self.devices[device_id]
        
        # Update allowed fields
        allowed_fields = ["name", "is_trusted", "status", "permissions"]
        for field in allowed_fields:
            if field in kwargs:
                setattr(device, field, kwargs[field])
        
        # Save changes
        if self._save_device(device):
            return True, ""
        else:
            return False, "Failed to save device changes"
    
    def revoke_device(self, device_id: str) -> bool:
        """
        Revoke a device's access.
        
        Args:
            device_id: The device ID
            
        Returns:
            bool: True if successful
        """
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        device.status = "revoked"
        # Generate a new token to invalidate existing one
        device.device_token = secrets.token_hex(32)
        
        return self._save_device(device)
    
    def get_connected_devices(self) -> List[Dict[str, Any]]:
        """
        Get a list of all connected devices.
        
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        devices_info = []
        
        for device in self.devices.values():
            if device.status == "active":
                # Include non-sensitive information
                info = {
                    "device_id": device.device_id,
                    "name": device.name,
                    "type": device.type,
                    "os_type": device.os_type,
                    "registered_date": device.registered_date.isoformat(),
                    "last_seen": device.last_seen.isoformat(),
                    "is_trusted": device.is_trusted,
                    "status": device.status
                }
                devices_info.append(info)
        
        return devices_info
    
    def _check_inactive_devices(self) -> None:
        """Check for and handle inactive devices."""
        now = datetime.datetime.now()
        inactive_days = self.cloud_manager.config.get("devices.inactive_timeout_days", 30)
        inactive_threshold = now - datetime.timedelta(days=inactive_days)
        
        for device in list(self.devices.values()):
            if device.status == "active" and device.last_seen < inactive_threshold:
                self.logger.info(f"Setting inactive device to suspended: {device.name} ({device.device_id})")
                device.status = "suspended"
                self._save_device(device)

