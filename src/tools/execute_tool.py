"""
Execute Tool - Run system commands safely.
"""

import asyncio
import subprocess
from typing import Any, Dict
import shlex

from . import BaseTool


class ExecuteTool(BaseTool):
    """Tool to execute system commands."""
    
    def _get_description(self) -> str:
        return (
            "Execute a system command safely. "
            "Use this to run shell commands, open applications, or perform system operations. "
            "Only safe commands are allowed."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute (e.g., 'ls -la', 'open https://google.com')"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": 30
                }
            },
            "required": ["command"]
        }
    
    async def execute(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a system command."""
        try:
            # Safety check - block dangerous commands
            dangerous_commands = ['rm -rf', 'format', 'mkfs', 'dd if=', 'del /f']
            for dangerous in dangerous_commands:
                if dangerous in command.lower():
                    return {
                        "success": False,
                        "error": f"Command blocked for safety: {dangerous}",
                        "output": None
                    }
            
            print(f"⚡ Executing: {command}")
            
            # Use subprocess with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                output = stdout.decode().strip() if stdout else ""
                error = stderr.decode().strip() if stderr else ""
                
                return {
                    "success": process.returncode == 0,
                    "returncode": process.returncode,
                    "output": output,
                    "error": error if error else None
                }
                
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "output": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": None
            }
