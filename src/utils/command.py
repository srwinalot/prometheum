
"""
Command execution utilities.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Union, List

logger = logging.getLogger(__name__)

@dataclass
class CommandResult:
    """Result of command execution."""
    exit_code: int
    stdout: str
    stderr: str
    
    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0

class CommandError(Exception):
    """Error executing command."""
    def __init__(self, cmd: str, exit_code: int, stderr: str):
        self.cmd = cmd
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(f"Command failed (exit code {exit_code}): {cmd}\n{stderr}")

def run_command(
    cmd: Union[str, List[str]], 
    check: bool = True,
    timeout: Optional[int] = None
) -> CommandResult:
    """Run shell command with error handling."""
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    logger.debug(f"Running: {cmd_str}")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=isinstance(cmd, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        exit_code = process.returncode
        
        result = CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)
        
        if check and exit_code != 0:
            raise CommandError(cmd_str, exit_code, stderr)
        
        return result
    
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        if check:
            raise CommandError(cmd_str, -1, f"Timeout after {timeout}s")
        return CommandResult(-1, stdout, stderr)
        
    except Exception as e:
        if check:
            raise CommandError(cmd_str, -1, str(e))
        return CommandResult(-1, "", str(e))

