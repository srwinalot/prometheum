"""
Utility functions for the storage management system.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""
    
    exit_code: int
    stdout: str
    stderr: str


class CommandError(Exception):
    """Exception raised when a command fails."""
    
    def __init__(self, message: str, exit_code: int, stderr: str):
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(f"{message} (exit code {exit_code}): {stderr}")


def run_command(cmd: str, check: bool = True, timeout: Optional[int] = None) -> CommandResult:
    """Run a shell command.
    
    Args:
        cmd: The command to run
        check: Whether to raise an exception if the command fails
        timeout: Timeout in seconds
        
    Returns:
        CommandResult with exit code, stdout and stderr
        
    Raises:
        CommandError: If the command fails and check is True
    """
    try:
        logger.debug(f"Running command: {cmd}")
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Get output with timeout
        stdout, stderr = process.communicate(timeout=timeout)
        exit_code = process.returncode
        
        # Check for errors
        if check and exit_code != 0:
            raise CommandError(f"Command failed: {cmd}", exit_code, stderr)
        
        return CommandResult(exit_code, stdout, stderr)
        
    except subprocess.TimeoutExpired:
        # Kill the process if it times out
        process.kill()
        stdout, stderr = process.communicate()
        
        if check:
            raise CommandError(f"Command timed out: {cmd}", -1, stderr)
        
        return CommandResult(-1, stdout, stderr)
        
    except Exception as e:
        if check:
            raise CommandError(f"Error running command: {cmd}", -1, str(e))
        
        return CommandResult(-1, "", str(e))

