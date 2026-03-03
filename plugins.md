# Talkie Voice Assistant - Plugins Documentation

**Last Updated:** 2026-03-02  
**Version:** 1.0.0

---

## Overview

Talkie supports extensible plugins that add new tools and capabilities. Plugins are discovered automatically from the `plugins/` directory and loaded at startup.

### Plugin Structure

```
plugins/
  plugin_name/
    manifest.json      # Plugin metadata
    __init__.py        # Plugin implementation
```

### Plugin Manifest Schema

```json
{
  "name": "plugin_name",
  "version": "1.0.0",
  "description": "What this plugin does",
  "author": "Author name",
  "tools": ["tool1", "tool2"],
  "dependencies": [],
  "config_schema": {},
  "enabled": true
}
```

---

## Available Plugins

### 1. cron_timer

**Version:** 1.0.0  
**Author:** Talkie  
**Status:** ✅ Active

Schedule timed notifications with voice and chat alerts.

#### Tools

| Tool | Description |
|------|-------------|
| `set_timer` | Set a timer for a specified duration |
| `list_timers` | List all active timers |
| `cancel_timer` | Cancel a timer by ID |

#### Usage

**Via Voice/Chat:**
```
"Set a timer for 5 minutes"
"Set a timer for 2 hours called lunch break"
"List my timers"
"Cancel timer 1"
```

**Via API:**

Set a timer:
```bash
curl -X POST http://localhost:8082/api/timer/set \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 5,
    "unit": "minutes",
    "description": "Lunch break"
  }'
```

Response:
```json
{
  "type": "timer_set",
  "success": true,
  "message": "Timer set for 5 minutes",
  "timer": {
    "id": 1,
    "description": "Lunch break",
    "duration_seconds": 300,
    "start_time": "2026-03-02T12:00:00",
    "end_time": "2026-03-02T12:05:00",
    "active": true
  }
}
```

List timers:
```bash
curl http://localhost:8082/api/timer/list
```

Response:
```json
{
  "type": "timers_list",
  "success": true,
  "count": 1,
  "timers": [
    {
      "id": 1,
      "description": "Lunch break",
      "duration_seconds": 300,
      "remaining_seconds": 245
    }
  ]
}
```

Cancel timer:
```bash
curl -X POST http://localhost:8082/api/timer/cancel/1
```

**Via Python:**
```python
from plugins.plugin_manager import get_plugin_manager

mgr = get_plugin_manager("plugins")
mgr.start()

# Set a timer
result = mgr.call_tool(
    "set_timer",
    duration=5,
    unit="minutes",
    description="Meeting"
)

# List timers
result = mgr.call_tool("list_timers")

# Cancel timer
result = mgr.call_tool("cancel_timer", timer_id=1)
```

#### Parameters

**set_timer:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `duration` | int | Yes | Timer duration value |
| `unit` | string | No | Time unit: "seconds", "minutes", "hours" (default: "seconds") |
| `description` | string | No | Timer description |

**list_timers:**
- No parameters required

**cancel_timer:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timer_id` | int | Yes | ID of the timer to cancel |

#### Features

- ✅ Background countdown threads
- ✅ Voice notification via TTS when timer completes
- ✅ Web notification broadcast to all connected clients
- ✅ Multiple concurrent timers
- ✅ Timer state persistence during session

---

### 2. example_calendar

**Version:** 1.0.0  
**Author:** Talkie Demo  
**Status:** ✅ Active

Example calendar plugin for managing events.

#### Tools

| Tool | Description |
|------|-------------|
| `create_event` | Create a new calendar event |
| `list_events` | List calendar events |
| `delete_event` | Delete an event by ID |

#### Usage

**Via Voice/Chat:**
```
"Create an event called team meeting for tomorrow at 2pm"
"List my events for 2026-03-02"
"Delete event 1"
```

**Via Python:**
```python
from plugins.plugin_manager import get_plugin_manager

mgr = get_plugin_manager("plugins")
mgr.start()

# Create event
result = mgr.call_tool(
    "create_event",
    title="Team Meeting",
    date="2026-03-02",
    time="14:00",
    description="Weekly sync"
)

# List events
result = mgr.call_tool("list_events", date="2026-03-02")

# Delete event
result = mgr.call_tool("delete_event", event_id=1)
```

#### Parameters

**create_event:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `title` | string | Yes | Event title |
| `date` | string | Yes | Event date (YYYY-MM-DD) |
| `time` | string | No | Event time (HH:MM, default: "09:00") |
| `description` | string | No | Event description |

**list_events:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `date` | string | No | Filter by date (YYYY-MM-DD) |

**delete_event:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `event_id` | int | Yes | ID of the event to delete |

---

## Plugin Development Guide

### Creating a New Plugin

1. **Create plugin directory:**
```bash
mkdir plugins/my_plugin
```

2. **Create manifest.json:**
```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "My custom plugin",
  "author": "Your Name",
  "tools": ["tool1", "tool2"],
  "dependencies": [],
  "enabled": true
}
```

3. **Create __init__.py:**
```python
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from plugins.plugin_manager import Plugin, Tool, PluginManifest


class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="What this tool does"
        )

    async def execute(self, param1: str) -> Dict[str, Any]:
        # Your tool logic here
        return {"success": True, "result": f"Processed: {param1}"}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param1"]
            }
        }


class MyPlugin(Plugin):
    def register_tools(self) -> List[Tool]:
        return [MyTool()]


# Plugin entry point
plugin_class = MyPlugin
```

4. **Restart web server** or wait for hot-reload (30s interval)

### Tool Schema

All tools must:
- Inherit from `Tool` base class
- Implement `async execute(**kwargs) -> Dict[str, Any]`
- Implement `get_schema() -> Dict[str, Any]` for LLM integration

### Plugin Lifecycle

- **Load:** Plugin loaded at startup or on file change
- **Initialize:** `register_tools()` called to register tools
- **Run:** Tools available for LLM to call
- **Shutdown:** `shutdown()` called on plugin unload

---

## API Reference

### Plugin Manager

```python
from plugins.plugin_manager import get_plugin_manager

# Get manager instance
mgr = get_plugin_manager("plugins")

# Start manager (loads plugins)
mgr.start()

# Get status
status = mgr.get_status()
# Returns: {
#   "running": True,
#   "plugin_count": 2,
#   "plugins": [...]
# }

# Call a tool
result = mgr.call_tool("tool_name", param1="value")

# Stop manager
mgr.stop()
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/timer/list` | GET | List all active timers |
| `/api/timer/set` | POST | Set a new timer |
| `/api/timer/cancel/{id}` | POST | Cancel a timer |

---

## Troubleshooting

### Plugin Not Loading

1. Check manifest.json exists and is valid JSON
2. Check __init__.py exists
3. Check plugin defines a `Plugin` subclass
4. Check `enabled: true` in manifest
5. Check logs for error messages

### Tool Not Found

- Ensure tool name matches manifest.json
- Ensure plugin loaded successfully
- Check tool is registered in `register_tools()`

### Hot-Reload Not Working

- Hot-reload checks every 30 seconds
- Check file modification times changed
- Restart web server if needed

---

## Changelog

### 1.0.0 (2026-03-02)

**Added:**
- cron_timer plugin with voice notifications
- example_calendar plugin
- Plugin architecture
- Hot-reload support
- REST API for timer management

---

**For more information, see:**
- [Plugin Manager Source](src/plugins/plugin_manager.py)
- [Example Plugin](plugins/example_calendar/__init__.py)
