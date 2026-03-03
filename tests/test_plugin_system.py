"""Unit tests for Plugin System."""

import pytest
from pathlib import Path
import tempfile
import shutil
import json
import sys

from src.plugins.plugin_manager import (
    PluginManager,
    PluginManifest,
    Tool,
    Plugin,
    get_plugin_manager,
    reset_plugin_manager,
)


class TestPluginManifest:
    """Tests for PluginManifest."""

    def test_from_dict(self):
        """Test creating manifest from dict."""
        data = {
            "name": "test_plugin",
            "version": "1.0.0",
            "description": "Test plugin",
            "author": "Test Author",
            "tools": ["tool1", "tool2"],
            "dependencies": ["dep1"],
            "enabled": True,
        }

        manifest = PluginManifest.from_dict(data)

        assert manifest.name == "test_plugin"
        assert manifest.version == "1.0.0"
        assert len(manifest.tools) == 2

    def test_to_dict(self):
        """Test converting manifest to dict."""
        manifest = PluginManifest(
            name="test",
            version="1.0.0",
            description="Test",
            author="Author",
            tools=["t1"],
        )

        data = manifest.to_dict()

        assert data["name"] == "test"
        assert data["version"] == "1.0.0"


class TestPluginManager:
    """Tests for PluginManager."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary plugin directory."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    def test_init_creates_directory(self, temp_plugin_dir):
        """Test that manager creates plugin directory."""
        manager = PluginManager(plugin_dir=str(temp_plugin_dir))
        assert temp_plugin_dir.exists()

    def test_get_status(self, temp_plugin_dir):
        """Test getting manager status."""
        manager = PluginManager(plugin_dir=str(temp_plugin_dir))

        status = manager.get_status()

        assert "running" in status
        assert "plugin_count" in status
        assert status["plugin_count"] == 0

    def test_reload_empty_directory(self, temp_plugin_dir):
        """Test reloading with no plugins."""
        manager = PluginManager(plugin_dir=str(temp_plugin_dir))

        results = manager.reload_plugins()

        assert len(results) == 0

    def test_start_and_stop(self, temp_plugin_dir):
        """Test starting and stopping manager."""
        manager = PluginManager(plugin_dir=str(temp_plugin_dir))

        manager.start()
        assert manager._running is True

        manager.stop()
        assert manager._running is False


class TestPluginLoading:
    """Tests for plugin loading."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary plugin directory."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    def test_load_valid_plugin(self, temp_plugin_dir):
        """Test loading a valid plugin."""
        # Create plugin directory
        plugin_dir = temp_plugin_dir / "test_plugin"
        plugin_dir.mkdir()

        # Create manifest
        manifest = {
            "name": "test_plugin",
            "version": "1.0.0",
            "description": "Test plugin",
            "author": "Test",
            "tools": ["test_tool"],
            "enabled": True,
        }

        with open(plugin_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create __init__.py with simple plugin
        init_code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.plugins.plugin_manager import Plugin, Tool, PluginManifest


class TestTool(Tool):
    def __init__(self):
        super().__init__("test_tool", "A test tool")
    
    async def execute(self, **kwargs):
        return {"success": True, "message": "Test executed"}


class TestPlugin(Plugin):
    def register_tools(self):
        return [TestTool()]


plugin_class = TestPlugin
"""

        with open(plugin_dir / "__init__.py", "w") as f:
            f.write(init_code)

        # Load plugin
        manager = PluginManager(plugin_dir=str(temp_plugin_dir))
        results = manager.reload_plugins()

        assert "test_plugin" in results
        assert results["test_plugin"] == "loaded"
        assert len(manager._plugins) == 1

    def test_skip_disabled_plugin(self, temp_plugin_dir):
        """Test that disabled plugins are skipped."""
        plugin_dir = temp_plugin_dir / "disabled_plugin"
        plugin_dir.mkdir()

        manifest = {
            "name": "disabled_plugin",
            "version": "1.0.0",
            "description": "Disabled",
            "author": "Test",
            "tools": [],
            "enabled": False,
        }

        with open(plugin_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create minimal __init__.py
        with open(plugin_dir / "__init__.py", "w") as f:
            f.write("")

        manager = PluginManager(plugin_dir=str(temp_plugin_dir))
        results = manager.reload_plugins()

        assert results.get("disabled_plugin") == "disabled"

    def test_skip_missing_manifest(self, temp_plugin_dir):
        """Test that directories without manifest are skipped."""
        plugin_dir = temp_plugin_dir / "no_manifest"
        plugin_dir.mkdir()

        # No manifest.json

        manager = PluginManager(plugin_dir=str(temp_plugin_dir))
        results = manager.reload_plugins()

        assert "no_manifest" in results
        assert "manifest" in results["no_manifest"]


class TestPluginSingleton:
    """Tests for plugin manager singleton."""

    def test_get_plugin_manager(self):
        """Test getting plugin manager instance."""
        reset_plugin_manager()

        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager1 is manager2

    def test_reset_plugin_manager(self):
        """Test resetting singleton."""
        reset_plugin_manager()
        manager1 = get_plugin_manager()
        reset_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager1 is not manager2


class TestExampleCalendarPlugin:
    """Tests for example calendar plugin."""

    def test_calendar_plugin_loads(self):
        """Test that example calendar plugin loads."""
        # Use actual plugin directory
        plugin_dir = Path(__file__).parent.parent / "plugins"

        if not plugin_dir.exists():
            pytest.skip("Plugins directory not found")

        manager = PluginManager(plugin_dir=str(plugin_dir))
        results = manager.reload_plugins()

        # Check if example_calendar loaded
        if "example_calendar" in results:
            assert results["example_calendar"] == "loaded"

        manager.stop()
