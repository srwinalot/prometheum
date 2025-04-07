"""
File storage management for Prometheum.

This module handles the storage, retrieval, organization, and 
tracking of files in the Prometheum personal cloud system with
NAS capabilities, OneDrive-like features, and iCloud synchronization.
"""

import os
import json
import time
import shutil
import hashlib
import logging
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, BinaryIO, Any, Tuple, Set

# For better MIME type detection
mimetypes.init()


class FileMetadata:
    """Metadata for a stored file."""
    
    def __init__(self, 
                file_path: str,
                size: int,
                created: datetime,
                modified: datetime,
                checksum: Optional[str] = None,
                mime_type: Optional[str] = None,
                is_encrypted: bool = False,
                owner: Optional[str] = None,
                tags: Optional[List[str]] = None,
                version: int = 1,
                previous_versions: Optional[List[str]] = None,
                shared_with: Optional[List[str]] = None,
                last_synced: Optional[Dict[str, datetime]] = None,
                is_favorite: bool = False,
                is_offline_available: bool = False):
        """
        Initialize file metadata.
        
        Args:
            file_path: Relative path of the file in the storage
            size: File size in bytes
            created: Creation timestamp
            modified: Last modification timestamp
            checksum: File checksum/hash
            mime_type: MIME type of the file
            is_encrypted: Whether the file is encrypted
            owner: User who owns the file
            tags: User-defined tags for the file
            version: Current version number
            previous_versions: List of previous version file paths
            shared_with: List of users the file is shared with
            last_synced: Dictionary mapping device IDs to their last sync time
            is_favorite: Whether the file is marked as favorite
            is_offline_available: Whether file is available offline on devices
        """
        self.file_path = file_path
        self.size = size
        self.created = created
        self.modified = modified
        self.checksum = checksum
        self.mime_type = mime_type or mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        self.is_encrypted = is_encrypted
        self.owner = owner
        self.tags = tags or []
        self.version = version
        self.previous_versions = previous_versions or []
        self.shared_with = shared_with or []
        self.last_synced = last_synced or {}
        self.is_favorite = is_favorite
        self.is_offline_available = is_offline_available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for storage."""
        return {
            'file_path': self.file_path,
            'size': self.size,
            'created': self.created.isoformat(),
            'modified': self.modified.isoformat(),
            'checksum': self.checksum,
            'mime_type': self.mime_type,
            'is_encrypted': self.is_encrypted,
            'owner': self.owner,
            'tags': self.tags,
            'version': self.version,
            'previous_versions': self.previous_versions,
            'shared_with': self.shared_with,
            'last_synced': {k: v.isoformat() for k, v in self.last_synced.items()},
            'is_favorite': self.is_favorite,
            'is_offline_available': self.is_offline_available
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        """Create metadata instance from dictionary."""
        # Convert timestamp strings to datetime objects
        created = datetime.fromisoformat(data['created'])
        modified = datetime.fromisoformat(data['modified'])
        
        # Convert last_synced timestamp strings to datetime objects
        last_synced = {}
        if 'last_synced' in data:
            for device_id, timestamp in data['last_synced'].items():
                last_synced[device_id] = datetime.fromisoformat(timestamp)
        
        return cls(
            file_path=data['file_path'],
            size=data['size'],
            created=created,
            modified=modified,
            checksum=data.get('checksum'),
            mime_type=data.get('mime_type'),
            is_encrypted=data.get('is_encrypted', False),
            owner=data.get('owner'),
            tags=data.get('tags', []),
            version=data.get('version', 1),
            previous_versions=data.get('previous_versions', []),
            shared_with=data.get('shared_with', []),
            last_synced=last_synced,
            is_favorite=data.get('is_favorite', False),
            is_offline_available=data.get('is_offline_available', False)
        )


class FileManager:
    """
    Manages file storage operations for the Prometheum cloud.
    
    This class handles storing, retrieving, organizing, and tracking
    files with NAS capabilities, OneDrive features, and iCloud-like syncing.
    """
    
    def __init__(self, cloud_manager):
        """
        Initialize the file manager.
        
        Args:
            cloud_manager: Reference to the CloudManager instance
        """
        self.cloud_manager = cloud_manager
        self.logger = logging.getLogger(__name__)
        
        # Main storage paths
        self.storage_root = self.cloud_manager.storage_path / "files"
        self.versions_root = self.cloud_manager.storage_path / "versions"
        self.trash_root = self.cloud_manager.storage_path / "trash"
        self.metadata_root = self.cloud_manager.storage_path / "metadata"
        self.shares_root = self.cloud_manager.storage_path / "shares"
        
        # Ensure directories exist
        for path in [self.storage_root, self.versions_root, self.trash_root, 
                     self.metadata_root, self.shares_root]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Cache for file metadata
        self.metadata_cache: Dict[str, FileMetadata] = {}
        
        # Track user quotas
        self.user_quotas: Dict[str, Dict[str, int]] = {}
        
        # Special folders (like iCloud and OneDrive)
        self.special_folders = {
            "documents": self.storage_root / "Documents",
            "photos": self.storage_root / "Photos",
            "videos": self.storage_root / "Videos",
            "music": self.storage_root / "Music",
            "desktop": self.storage_root / "Desktop",
            "downloads": self.storage_root / "Downloads"
        }
        
        # Create special folders
        for folder in self.special_folders.values():
            folder.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize the file manager system.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Load all file metadata
            self._load_metadata()
            
            # Load user quotas
            self._load_quotas()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize file manager: {str(e)}")
            return False
    
    def _load_metadata(self) -> None:
        """Load all file metadata from storage."""
        try:
            for metadata_file in self.metadata_root.glob("**/*.json"):
                try:
                    with open(metadata_file, "r") as f:
                        metadata_data = json.load(f)
                        metadata = FileMetadata.from_dict(metadata_data)
                        self.metadata_cache[metadata.file_path] = metadata
                except Exception as e:
                    self.logger.error(f"Error loading metadata from {metadata_file}: {str(e)}")
            
            self.logger.info(f"Loaded metadata for {len(self.metadata_cache)} files")
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {str(e)}")
    
    def _save_metadata(self, metadata: FileMetadata) -> bool:
        """
        Save file metadata to persistent storage.
        
        Args:
            metadata: The file metadata to save
            
        Returns:
            bool: True if successful
        """
        try:
            # Hash the file path to use as filename
            path_hash = hashlib.md5(metadata.file_path.encode()).hexdigest()
            metadata_file = self.metadata_root / f"{path_hash}.json"
            
            with open(metadata_file, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update cache
            self.metadata_cache[metadata.file_path] = metadata
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save metadata for {metadata.file_path}: {str(e)}")
            return False
    
    def _load_quotas(self) -> None:
        """Load user quota information."""
        quota_file = self.cloud_manager.storage_path / "quotas.json"
        
        if quota_file.exists():
            try:
                with open(quota_file, "r") as f:
                    self.user_quotas = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load quotas: {str(e)}")
                self.user_quotas = {}
    
    def _save_quotas(self) -> bool:
        """Save user quota information."""
        try:
            quota_file = self.cloud_manager.storage_path / "quotas.json"
            
            with open(quota_file, "w") as f:
                json.dump(self.user_quotas, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save quotas: {str(e)}")
            return False
    
    def store_file(self, file_path: str, content: Union[bytes, BinaryIO], 
                  owner: Optional[str] = None, 
                  device_id: Optional[str] = None,
                  create_versions: bool = True) -> bool:
        """
        Store a file in the cloud storage.
        
        Args:
            file_path: Relative path where the file should be stored
            content: File content as bytes or file-like object
            owner: Owner of the file
            device_id: ID of the device uploading the file
            create_versions: Whether to create a new version if file exists
            
        Returns:
            bool: True if successful
        """
        try:
            # Normalize the path
            rel_path = self._normalize_path(file_path)
            abs_path = self.storage_root / rel_path
            
            # Ensure parent directory exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists and handle versioning
            existing_metadata = self.get_metadata(rel_path)
            
            if existing_metadata and create_versions:
                # Create a new version
                version_num = existing_metadata.version + 1
                version_path = self._create_version(rel_path, existing_metadata)
                
                if version_path:
                    existing_metadata.previous_versions.append(version_path)
                    existing_metadata.version = version_num
            
            # Write the file content
            if isinstance(content, bytes):
                with open(abs_path, "wb") as f:
                    f.write(content)
                file_size = len(content)
            else:
                # Assume it's a file-like object
                with open(abs_path, "wb") as f:
                    content.seek(0)
                    shutil.copyfileobj(content, f)
                content.seek(0, os.SEEK_END)
                file_size = content.tell()
            
            # Calculate checksum
            checksum = self._calculate_checksum(abs_path)
            
            # Create or update metadata
            now = datetime.now()
            
            if existing_metadata:
                metadata = existing_metadata
                metadata.size = file_size
                metadata.modified = now
                metadata.checksum = checksum
                
                # Update sync info if device_id is provided
                if device_id:
                    metadata.last_synced[device_id] = now
            else:
                metadata = FileMetadata(
                    file_path=str(rel_path),
                    size=file_size,
                    created=now,
                    modified=now,
                    checksum=checksum,
                    owner=owner
                )
                
                # Set initial sync info if device_id is provided
                if device_id:
                    metadata.last_synced = {device_id: now}
            
            # Save metadata
            self._save_metadata(metadata)
            
            # Update quota
            if owner and owner in self.user_quotas:
                self.user_quotas[owner]['used'] += file_size
                self._save_quotas()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store file {file_path}: {str(e)}")
            return False
    
    def get_file(self, file_path: str) -> Optional[Path]:
        """
        Get the absolute path to a stored file.
        
        Args:
            file_path: Relative path of the file
            
        Returns:
            Optional[Path]: Absolute path to the file, or None if not found
        """
        rel_path = self._normalize_path(file_path)
        abs_path = self.storage_root / rel_path
        
        if abs_path.exists() and abs_path.is_file():
            return abs_path
        
        return None
    
    def read_file(self, file_path: str) -> Optional[bytes]:
        """
        Read file content as bytes.
        
        Args:
            file_path: Relative path of the file
            
        Returns:
            Optional[bytes]: File content, or None if not found
        """
        abs_path = self.get_file(file_path)
        
        if abs_path:
            try:
                with open(abs_path, "rb") as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"Failed to read file {file_path}: {str(e)}")
        
        return None
    
    def get_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """
        Get metadata for a file.
        
        Args:
            file_path: Relative path of the file
            
        Returns:
            Optional[FileMetadata]: File metadata, or None if not found
        """
        rel_path = self._normalize_path(file_path)
        
        # Check cache first
        if rel_path in self.metadata_cache:
            return self.metadata_cache[rel_path]
        
        # Not in cache, try to load from storage
        path_hash = hashlib.md5(rel_path.encode()).hexdigest()
        metadata_file = self.metadata_root / f"{path_hash}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata_data = json.load(f)
                

