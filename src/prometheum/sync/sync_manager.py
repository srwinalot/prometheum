"""
File synchronization management for Prometheum.

This module handles tracking, comparing, and synchronizing files
between the local system and the cloud storage.
"""

import os
import time
import uuid
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class SyncItem:
    """Represents a synchronized directory configuration."""
    
    def __init__(self, local_path: str, sync_id: str, 
                 sync_policy: str = "two-way",
                 auto_backup: bool = False):
        """
        Initialize a sync item.
        
        Args:
            local_path: Path to the local directory
            sync_id: Unique identifier for this sync relationship
            sync_policy: How synchronization should be performed
            auto_backup: Whether to automatically create backups
        """
        self.local_path = Path(local_path)
        self.sync_id = sync_id
        self.sync_policy = sync_policy
        self.auto_backup = auto_backup
        self.last_sync = None
        self.status = "initialized"
        self.error = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "sync_id": self.sync_id,
            "local_path": str(self.local_path),
            "sync_policy": self.sync_policy,
            "auto_backup": self.auto_backup,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "status": self.status,
            "error": self.error
        }


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system change events for sync directories."""
    
    def __init__(self, sync_manager):
        """
        Initialize the file change handler.
        
        Args:
            sync_manager: Reference to the SyncManager instance
        """
        self.sync_manager = sync_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle modified file events."""
        if not event.is_directory:
            self.logger.debug(f"File modified: {event.src_path}")
            self.sync_manager.queue_file_for_sync(event.src_path)
    
    def on_created(self, event):
        """Handle file creation events."""
        self.logger.debug(f"File created: {event.src_path}")
        self.sync_manager.queue_file_for_sync(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        self.logger.debug(f"File deleted: {event.src_path}")
        self.sync_manager.queue_deletion_for_sync(event.src_path)
    
    def on_moved(self, event):
        """Handle file move events."""
        self.logger.debug(f"File moved: {event.src_path} -> {event.dest_path}")
        self.sync_manager.queue_move_for_sync(event.src_path, event.dest_path)


class SyncManager:
    """
    Manages file synchronization between local devices and cloud storage.
    
    This class handles tracking which files need to be synced, performing
    synchronization operations, and resolving conflicts.
    """
    
    def __init__(self, cloud_manager):
        """
        Initialize the sync manager.
        
        Args:
            cloud_manager: Reference to the CloudManager instance
        """
        self.cloud_manager = cloud_manager
        self.logger = logging.getLogger(__name__)
        self.sync_items: Dict[str, SyncItem] = {}
        self.sync_queue: List[Dict[str, Any]] = []
        self.observer = Observer()
        self.is_running = False
        self.sync_thread = None
        self._lock = threading.Lock()
    
    def add_directory(self, local_path: str, 
                     sync_policy: str = "two-way",
                     auto_backup: bool = False) -> str:
        """
        Add a directory to be synchronized.
        
        Args:
            local_path: Path to the local directory
            sync_policy: Synchronization policy
            auto_backup: Whether to automatically back up
            
        Returns:
            str: ID of the new sync configuration
        """
        sync_id = str(uuid.uuid4())
        
        path = Path(local_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        sync_item = SyncItem(local_path, sync_id, sync_policy, auto_backup)
        self.sync_items[sync_id] = sync_item
        
        # Watch the directory for changes if the sync service is running
        if self.is_running:
            self._watch_directory(sync_item)
        
        self.logger.info(f"Added sync directory: {local_path} with ID {sync_id}")
        return sync_id
    
    def remove_directory(self, sync_id: str) -> bool:
        """
        Remove a synchronized directory.
        
        Args:

