"""
File Tools - Read, write, and manage files.
"""

import os
from typing import Any, Dict
from pathlib import Path

from . import BaseTool


class ReadFileTool(BaseTool):
    """Tool to read file contents."""
    
    def _get_description(self) -> str:
        return "Read the contents of a file. Use this to access documents, code, or any text files."
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (0 for all)",
                    "default": 0
                }
            },
            "required": ["path"]
        }
    
    async def execute(self, path: str, limit: int = 0) -> Dict[str, Any]:
        """Read file contents."""
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}",
                    "content": None
                }
            
            if not file_path.is_file():
                return {
                    "success": False,
                    "error": f"Path is not a file: {path}",
                    "content": None
                }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if limit > 0:
                    lines = f.readlines()[:limit]
                    content = ''.join(lines)
                else:
                    content = f.read()
            
            return {
                "success": True,
                "path": str(file_path),
                "content": content,
                "size": len(content),
                "truncated": limit > 0 and len(content) < os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }


class WriteFileTool(BaseTool):
    """Tool to write to files."""
    
    def _get_description(self) -> str:
        return "Write or append content to a file. Use this to save notes, create documents, or write code."
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to file instead of overwriting",
                    "default": False
                }
            },
            "required": ["path", "content"]
        }
    
    async def execute(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """Write content to file."""
        try:
            file_path = Path(path).expanduser().resolve()
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(file_path),
                "bytes_written": len(content.encode('utf-8')),
                "append": append
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ListDirectoryTool(BaseTool):
    """Tool to list directory contents."""
    
    def _get_description(self) -> str:
        return "List the contents of a directory. Use this to browse files and folders."
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list",
                    "default": "."
                }
            }
        }
    
    async def execute(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        try:
            dir_path = Path(path).expanduser().resolve()
            
            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {path}",
                    "items": []
                }
            
            if not dir_path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}",
                    "items": []
                }
            
            items = []
            for item in dir_path.iterdir():
                item_info = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                }
                items.append(item_info)
            
            # Sort: directories first, then alphabetically
            items.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"].lower()))
            
            return {
                "success": True,
                "path": str(dir_path),
                "items": items,
                "count": len(items)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "items": []
            }
