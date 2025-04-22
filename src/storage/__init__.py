"""
Prometheum Storage Management System with AI Integration.

This package provides comprehensive storage management functionality for the Prometheum NAS Router OS,
including pool management, volume management, backup handling, disk health monitoring, and AI-powered
data analysis and cataloging.
"""

from .pool import StoragePool, StoragePoolManager
from .volume import Volume, VolumeManager
from .backup import BackupLocation, BackupManager
from .health import HealthMonitor
from .disk import DiskManager
from .ai import AIDataManager, ContentAnalyzer, QueryProcessor

__version__ = "0.1.0"

