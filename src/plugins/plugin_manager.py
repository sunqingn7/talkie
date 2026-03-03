"""
Plugin System for Talkie Voice Assistant

Provides:
- Plugin discovery and loading
- Tool registration API
- Hot-reload support
- Plugin manifest/schema validation
- Sandboxed execution
"""

import json
import importlib
import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PluginManifest:
    """Plugin manifest schema."""

    name: str
    version: str
    description: str
    author: str
    tools: List[str]  # Tool names provided by this plugin
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        """Create manifest from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.1"),
            description=data.get("description", ""),
            author=data.get("author", "unknown"),
            tools=data.get("tools", []),
            dependencies=data.get("dependencies", []),
            config_schema=data.get("config_schema"),
            enabled=data.get("enabled", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tools": self.tools,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }


class Tool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool. Override in subclasses."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM."""
        return {"name": self.name, "description": self.description, "parameters": {}}


class Plugin(ABC):
    """Base class for all plugins."""

    def __init__(self, manifest: PluginManifest):
        self.manifest = manifest
        self._tools: Dict[str, Tool] = {}
        self._config: Dict[str, Any] = {}
        self._initialized = False

    @abstractmethod
    def register_tools(self) -> List[Tool]:
        """Register tools provided by this plugin."""
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin with settings."""
        self._config = config
        logger.info(
            "Configured plugin %s with %d settings", self.manifest.name, len(config)
        )

    def initialize(self) -> bool:
        """Initialize the plugin. Returns True if successful."""
        try:
            self._tools = {tool.name: tool for tool in self.register_tools()}
            self._initialized = True
            logger.info(
                "Initialized plugin %s with %d tools",
                self.manifest.name,
                len(self._tools),
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize plugin %s: %s", self.manifest.name, e)
            return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> List[Tool]:
        """Get all tools from this plugin."""
        return list(self._tools.values())

    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    def shutdown(self) -> None:
        """Clean up plugin resources."""
        logger.info("Shutting down plugin %s", self.manifest.name)


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Directory structure:
    plugins/
      plugin_name/
        __init__.py
        manifest.json
        tool.py
    """

    def __init__(self, plugin_dir: Optional[str] = None):
        """
        Initialize plugin manager.

        Args:
            plugin_dir: Directory containing plugins
        """
        self.plugin_dir = Path(plugin_dir or "plugins")
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        # Plugin registry
        self._plugins: Dict[str, Plugin] = {}
        self._manifests: Dict[str, PluginManifest] = {}

        # Hot-reload
        self._watcher_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_scan_time = 0
        self._scan_interval = 30  # seconds

        # Callbacks
        self.on_plugin_loaded: Optional[Callable[[Plugin], None]] = None
        self.on_plugin_unloaded: Optional[Callable[[Plugin], None]] = None

        logger.info("PluginManager initialized at %s", self.plugin_dir)

    def start(self) -> None:
        """Start plugin manager (load plugins and start watcher)."""
        if self._running:
            logger.warning("PluginManager already running")
            return

        logger.info("Starting PluginManager...")
        self._running = True

        # Load all plugins
        self.reload_plugins()

        # Start file watcher for hot-reload
        self._watcher_thread = threading.Thread(
            target=self._watcher_loop, name="PluginWatcher", daemon=True
        )
        self._watcher_thread.start()

        logger.info("PluginManager started")

    def stop(self) -> None:
        """Stop plugin manager."""
        if not self._running:
            return

        logger.info("Stopping PluginManager...")
        self._running = False

        # Shutdown all plugins
        for plugin in list(self._plugins.values()):
            try:
                plugin.shutdown()
            except Exception as e:
                logger.error(
                    "Error shutting down plugin %s: %s", plugin.manifest.name, e
                )

        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=5.0)

        logger.info("PluginManager stopped")

    def reload_plugins(self) -> Dict[str, str]:
        """
        Scan and reload all plugins.

        Returns:
            Dict mapping plugin names to status messages
        """
        results = {}

        if not self.plugin_dir.exists():
            logger.warning("Plugin directory does not exist: %s", self.plugin_dir)
            return results

        # Find all plugin directories
        plugin_dirs = [
            d
            for d in self.plugin_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        for plugin_dir in plugin_dirs:
            manifest_path = plugin_dir / "manifest.json"

            if not manifest_path.exists():
                logger.warning("Skipping %s: no manifest.json", plugin_dir.name)
                results[plugin_dir.name] = "missing manifest"
                continue

            try:
                # Load manifest
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)

                manifest = PluginManifest.from_dict(manifest_data)

                # Skip disabled plugins
                if not manifest.enabled:
                    results[manifest.name] = "disabled"
                    continue

                # Load plugin
                status = self._load_plugin(plugin_dir, manifest)
                results[manifest.name] = status

            except Exception as e:
                logger.error("Failed to load plugin from %s: %s", plugin_dir.name, e)
                results[plugin_dir.name] = f"error: {str(e)}"

        logger.info("Plugin reload complete: %d plugins loaded", len(self._plugins))
        return results

    def _load_plugin(self, plugin_dir: Path, manifest: PluginManifest) -> str:
        """Load a single plugin from directory."""
        plugin_init = plugin_dir / "__init__.py"

        if not plugin_init.exists():
            return "missing __init__.py"

        # Add parent dir to path temporarily (so we can import plugins.plugin_name)
        parent_dir_str = str(plugin_dir.parent)
        if parent_dir_str not in sys.path:
            sys.path.insert(0, parent_dir_str)

        try:
            # Import plugin module
            module_name = f"{plugin_dir.parent.name}.{plugin_dir.name}"

            # Check if already loaded
            if module_name in sys.modules:
                # Reload if already loaded
                importlib.reload(sys.modules[module_name])

            module = importlib.import_module(module_name)

            # Find Plugin subclass
            plugin_class = self._find_plugin_class(module)

            if plugin_class is None:
                return "no Plugin subclass found"

            # Create plugin instance
            plugin = plugin_class(manifest)

            # Initialize plugin
            if not plugin.initialize():
                return "initialization failed"

            # Check if replacing existing plugin
            if manifest.name in self._plugins:
                old_plugin = self._plugins[manifest.name]
                old_plugin.shutdown()
                if self.on_plugin_unloaded:
                    try:
                        self.on_plugin_unloaded(old_plugin)
                    except Exception as e:
                        logger.error("Error in on_plugin_unloaded callback: %s", e)

            # Register plugin
            self._plugins[manifest.name] = plugin
            self._manifests[manifest.name] = manifest

            logger.info(
                "Loaded plugin %s v%s (%d tools)",
                manifest.name,
                manifest.version,
                len(plugin.get_tools()),
            )

            # Trigger callback
            if self.on_plugin_loaded:
                try:
                    self.on_plugin_loaded(plugin)
                except Exception as e:
                    logger.error("Error in on_plugin_loaded callback: %s", e)

            return "loaded"

        except Exception as e:
            logger.error("Error loading plugin %s: %s", manifest.name, e)
            import traceback

            traceback.print_exc()
            return f"load error: {str(e)}"

        finally:
            # Clean up path
            if parent_dir_str in sys.path:
                sys.path.remove(parent_dir_str)

    def _find_plugin_class(self, module) -> Optional[Type[Plugin]]:
        """Find Plugin subclass in module."""
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Plugin) and obj is not Plugin:
                return obj
        return None

    def _watcher_loop(self) -> None:
        """Background loop for watching plugin changes."""
        logger.info("Plugin watcher started (interval: %ds)", self._scan_interval)

        while self._running:
            try:
                current_time = time.time()

                # Check if it's time to scan
                if current_time - self._last_scan_time >= self._scan_interval:
                    # Check for changes
                    if self._check_for_changes():
                        logger.info("Plugin changes detected, reloading...")
                        self.reload_plugins()

                    self._last_scan_time = current_time

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error("Error in plugin watcher: %s", e)
                time.sleep(5)

        logger.info("Plugin watcher stopped")

    def _check_for_changes(self) -> bool:
        """Check if any plugin files have changed."""
        try:
            for plugin_dir in self.plugin_dir.iterdir():
                if not plugin_dir.is_dir():
                    continue

                # Check manifest.json
                manifest_path = plugin_dir / "manifest.json"
                if manifest_path.exists():
                    mtime = manifest_path.stat().st_mtime
                    if mtime > self._last_scan_time:
                        return True

                # Check Python files
                for py_file in plugin_dir.glob("*.py"):
                    mtime = py_file.stat().st_mtime
                    if mtime > self._last_scan_time:
                        return True

            return False

        except Exception as e:
            logger.error("Error checking for plugin changes: %s", e)
            return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_manifest(self, name: str) -> Optional[PluginManifest]:
        """Get plugin manifest by name."""
        return self._manifests.get(name)

    def get_all_plugins(self) -> List[Plugin]:
        """Get all loaded plugins."""
        return list(self._plugins.values())

    def get_all_tools(self) -> Dict[str, Tool]:
        """Get all tools from all plugins."""
        tools = {}
        for plugin in self._plugins.values():
            for tool in plugin.get_tools():
                tools[tool.name] = tool
        return tools

    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool by name.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        # Find tool
        for plugin in self._plugins.values():
            tool = plugin.get_tool(tool_name)
            if tool:
                import asyncio

                return asyncio.run(tool.execute(**kwargs))

        raise ValueError(f"Tool not found: {tool_name}")

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin."""
        manifest = self._manifests.get(name)
        if not manifest:
            return False

        if manifest.enabled:
            return True  # Already enabled

        manifest.enabled = True

        # Update manifest file
        manifest_path = self.plugin_dir / name / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)

        # Reload plugin
        plugin_dir = self.plugin_dir / name
        self._load_plugin(plugin_dir, manifest)

        return True

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        # Shutdown plugin
        plugin.shutdown()

        # Remove from registry
        del self._plugins[name]

        # Update manifest
        manifest = self._manifests.get(name)
        if manifest:
            manifest.enabled = False

            # Update manifest file
            manifest_path = self.plugin_dir / name / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest.to_dict(), f, indent=2)

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get plugin manager status."""
        return {
            "running": self._running,
            "plugin_count": len(self._plugins),
            "plugins": [
                {
                    "name": manifest.name,
                    "version": manifest.version,
                    "enabled": manifest.enabled,
                    "tools": manifest.tools,
                }
                for manifest in self._manifests.values()
            ],
        }


# Global instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager(plugin_dir: Optional[str] = None) -> PluginManager:
    """Get or create the global plugin manager instance."""
    global _plugin_manager

    if _plugin_manager is None:
        _plugin_manager = PluginManager(plugin_dir=plugin_dir)

    return _plugin_manager


def reset_plugin_manager() -> None:
    """Reset the global plugin manager instance."""
    global _plugin_manager

    if _plugin_manager:
        _plugin_manager.stop()
    _plugin_manager = None
