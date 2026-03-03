"""Cron Timer Plugin - Schedule timed notifications with voice alerts."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import time
from datetime import datetime, timedelta

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from plugins.plugin_manager import Plugin, Tool, PluginManifest


class TimerJob:
    """Represents a scheduled timer."""

    def __init__(
        self,
        timer_id: int,
        duration_seconds: int,
        description: Optional[str] = None,
    ):
        self.timer_id = timer_id
        self.duration_seconds = duration_seconds
        self.description = description or f"Timer for {duration_seconds} seconds"
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(seconds=duration_seconds)
        self.active = True
        self._thread: Optional[threading.Thread] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert timer to dictionary."""
        return {
            "id": self.timer_id,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "active": self.active,
        }


class SetTimerTool(Tool):
    """Tool to set a timer."""

    def __init__(self, timer_store: "TimerStore"):
        super().__init__(
            name="set_timer",
            description="Set a timer for a specified duration with optional description. Timer will notify via voice and chat when complete.",
        )
        self._store = timer_store

    async def execute(
        self,
        duration: int,
        unit: str = "seconds",
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set a timer.

        Args:
            duration: Timer duration value
            unit: Time unit (seconds, minutes, hours)
            description: Optional description for the timer

        Returns:
            Timer creation result
        """
        # Convert to seconds
        if unit == "minutes":
            duration_seconds = duration * 60
        elif unit == "hours":
            duration_seconds = duration * 3600
        else:
            duration_seconds = duration

        if duration_seconds <= 0:
            return {
                "success": False,
                "error": "Duration must be positive",
            }

        # Create timer
        timer = self._store.create_timer(duration_seconds, description)

        return {
            "success": True,
            "message": f"Timer set for {duration} {unit}",
            "timer": timer.to_dict(),
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "integer",
                        "description": "Timer duration value",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["seconds", "minutes", "hours"],
                        "description": "Time unit (default: seconds)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional timer description",
                    },
                },
                "required": ["duration"],
            },
        }


class ListTimersTool(Tool):
    """Tool to list active timers."""

    def __init__(self, timer_store: "TimerStore"):
        super().__init__(
            name="list_timers",
            description="List all active timers and their remaining time",
        )
        self._store = timer_store

    async def execute(self) -> Dict[str, Any]:
        """List all active timers."""
        timers = self._store.get_active_timers()

        result_timers = []
        for timer in timers:
            timer_dict = timer.to_dict()
            if timer.active:
                remaining = max(
                    0,
                    int((timer.end_time - datetime.now()).total_seconds()),
                )
                timer_dict["remaining_seconds"] = remaining
            result_timers.append(timer_dict)

        return {
            "success": True,
            "count": len(result_timers),
            "timers": result_timers,
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


class CancelTimerTool(Tool):
    """Tool to cancel a timer."""

    def __init__(self, timer_store: "TimerStore"):
        super().__init__(
            name="cancel_timer",
            description="Cancel an active timer by ID",
        )
        self._store = timer_store

    async def execute(self, timer_id: int) -> Dict[str, Any]:
        """
        Cancel a timer.

        Args:
            timer_id: ID of the timer to cancel

        Returns:
            Cancellation result
        """
        cancelled = self._store.cancel_timer(timer_id)

        if cancelled:
            return {
                "success": True,
                "message": f"Timer {timer_id} cancelled",
            }
        else:
            return {
                "success": False,
                "error": f"Timer {timer_id} not found or already inactive",
            }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "timer_id": {
                        "type": "integer",
                        "description": "ID of the timer to cancel",
                    }
                },
                "required": ["timer_id"],
            },
        }


class TimerStore:
    """Central store for managing timers."""

    def __init__(self):
        self._timers: Dict[int, TimerJob] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._notification_callback: Optional[callable] = None

    def set_notification_callback(self, callback: callable) -> None:
        """Set callback for timer notifications."""
        self._notification_callback = callback

    def create_timer(
        self, duration_seconds: int, description: Optional[str] = None
    ) -> TimerJob:
        """Create and start a new timer."""
        with self._lock:
            timer_id = self._next_id
            self._next_id += 1

            timer = TimerJob(timer_id, duration_seconds, description)
            self._timers[timer_id] = timer

            # Start timer thread
            timer._thread = threading.Thread(
                target=self._timer_loop, args=(timer,), daemon=True
            )
            timer._thread.start()

            return timer

    def _timer_loop(self, timer: TimerJob) -> None:
        """Background loop for a single timer."""
        while timer.active:
            remaining = (timer.end_time - datetime.now()).total_seconds()

            if remaining <= 0:
                # Timer complete
                timer.active = False
                self._notify_timer_complete(timer)
                break

            time.sleep(min(1, remaining))

    def _notify_timer_complete(self, timer: TimerJob) -> None:
        """Notify that a timer has completed."""
        if self._notification_callback:
            try:
                self._notification_callback(timer)
            except Exception as e:
                print(f"Error in timer notification callback: {e}")

    def get_active_timers(self) -> List[TimerJob]:
        """Get all active timers."""
        with self._lock:
            return [t for t in self._timers.values() if t.active]

    def cancel_timer(self, timer_id: int) -> bool:
        """Cancel a timer by ID."""
        with self._lock:
            timer = self._timers.get(timer_id)
            if timer and timer.active:
                timer.active = False
                return True
            return False

    def get_timer(self, timer_id: int) -> Optional[TimerJob]:
        """Get a timer by ID."""
        with self._lock:
            return self._timers.get(timer_id)


# Global timer store
_timer_store: Optional[TimerStore] = None


def get_timer_store() -> TimerStore:
    """Get or create the global timer store."""
    global _timer_store
    if _timer_store is None:
        _timer_store = TimerStore()
    return _timer_store


class CronTimerPlugin(Plugin):
    """Cron timer plugin for scheduled notifications."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self._store = get_timer_store()

    def register_tools(self) -> List[Tool]:
        """Register timer tools."""
        return [
            SetTimerTool(self._store),
            ListTimersTool(self._store),
            CancelTimerTool(self._store),
        ]

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin."""
        super().configure(config)
        # Store notification preferences if needed

    def shutdown(self) -> None:
        """Clean up plugin resources."""
        # Cancel all active timers
        store = get_timer_store()
        for timer in store.get_active_timers():
            timer.active = False
        super().shutdown()


# Plugin entry point
plugin_class = CronTimerPlugin
