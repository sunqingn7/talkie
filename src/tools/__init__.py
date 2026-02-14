"""
Base Tool class and tool interface definitions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Base class for all MCP tools."""
    
    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__.lower().replace('tool', '')
        self.description = self._get_description()
        self.input_schema = self._get_input_schema()
    
    @abstractmethod
    def _get_description(self) -> str:
        """Return tool description."""
        pass
    
    @abstractmethod
    def _get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for tool inputs."""
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        pass
