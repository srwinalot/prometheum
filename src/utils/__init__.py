
"""
Shared utility functions for Prometheum.
"""

from .command import run_command, CommandResult, CommandError
from .config import load_config, save_config
from .fs import ensure_dir, path_exists, scan_devices

