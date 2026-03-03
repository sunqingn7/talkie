"""Example Calendar Plugin - Demonstrates plugin system."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datetime import datetime, timedelta
from src.plugins.plugin_manager import Plugin, Tool, PluginManifest


class CreateEventTool(Tool):
    """Tool to create calendar events."""

    def __init__(self):
        super().__init__(
            name="create_event",
            description="Create a new calendar event with title, date, and optional description",
        )
        self._events: List[Dict[str, Any]] = []

    async def execute(
        self,
        title: str,
        date: str,
        time: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a calendar event."""
        event = {
            "id": len(self._events) + 1,
            "title": title,
            "date": date,
            "time": time or "09:00",
            "description": description or "",
            "created_at": datetime.now().isoformat(),
        }
        self._events.append(event)

        return {
            "success": True,
            "message": f"Event '{title}' created successfully",
            "event": event,
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "date": {
                        "type": "string",
                        "description": "Event date (YYYY-MM-DD)",
                    },
                    "time": {
                        "type": "string",
                        "description": "Event time (HH:MM, optional)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description (optional)",
                    },
                },
                "required": ["title", "date"],
            },
        }


class ListEventsTool(Tool):
    """Tool to list calendar events."""

    def __init__(self):
        super().__init__(
            name="list_events", description="List all calendar events or filter by date"
        )
        self._events: List[Dict[str, Any]] = []

    async def execute(self, date: Optional[str] = None) -> Dict[str, Any]:
        """List calendar events."""
        events = self._events

        if date:
            events = [e for e in events if e.get("date") == date]

        return {"success": True, "count": len(events), "events": events}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Filter by date (YYYY-MM-DD, optional)",
                    }
                },
                "required": [],
            },
        }


class DeleteEventTool(Tool):
    """Tool to delete calendar events."""

    def __init__(self):
        super().__init__(
            name="delete_event", description="Delete a calendar event by ID"
        )
        self._events: List[Dict[str, Any]] = []

    async def execute(self, event_id: int) -> Dict[str, Any]:
        """Delete a calendar event."""
        for i, event in enumerate(self._events):
            if event["id"] == event_id:
                deleted = self._events.pop(i)
                return {
                    "success": True,
                    "message": f"Event '{deleted['title']}' deleted",
                    "deleted_event": deleted,
                }

        return {"success": False, "error": f"Event with ID {event_id} not found"}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "integer",
                        "description": "ID of the event to delete",
                    }
                },
                "required": ["event_id"],
            },
        }


class ExampleCalendarPlugin(Plugin):
    """Example calendar plugin."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self._shared_events: List[Dict[str, Any]] = []

    def register_tools(self) -> List[Tool]:
        """Register calendar tools."""
        # Share events list between tools
        create_tool = CreateEventTool()
        list_tool = ListEventsTool()
        delete_tool = DeleteEventTool()

        # Share state (in real plugin, use proper storage)
        create_tool._events = self._shared_events
        list_tool._events = self._shared_events
        delete_tool._events = self._shared_events

        return [create_tool, list_tool, delete_tool]


# Plugin entry point
plugin_class = ExampleCalendarPlugin
