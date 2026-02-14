"""
DateTime Tool - Get current date and time information.
"""

from datetime import datetime
from typing import Any, Dict
from tools import BaseTool


class DateTimeTool(BaseTool):
    """
    Get the current date and time from the system clock.
    Use this tool when the user asks about:
    - Current date (e.g., "what's today", "what date is it")
    - Current time (e.g., "what time is it")
    - Day of the week (e.g., "what day is today")
    """
    
    def _get_description(self) -> str:
        return """
        Get the current date, time, and day information from the local system.
        Use this tool when the user asks about the current date, time, or day.
        
        Examples:
        - "What date is today?"
        - "What time is it?"
        - "What day is today?"
        - "What's today's date?"
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Format preference: 'full' (date and time), 'date' (date only), 'time' (time only), 'day' (day of week)",
                    "enum": ["full", "date", "time", "day"],
                    "default": "full"
                }
            },
            "required": []
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
    
    async def execute(self, format: str = "full") -> Dict[str, Any]:
        """Get current date and time."""
        now = datetime.now()
        
        if format == "date":
            result = now.strftime("%A, %B %d, %Y")
        elif format == "time":
            result = now.strftime("%I:%M %p")
        elif format == "day":
            result = now.strftime("%A")
        else:  # full
            result = now.strftime("%A, %B %d, %Y at %I:%M %p")
        
        print(f"📅 DateTime query ({format}): {result}")
        
        return {
            "success": True,
            "datetime": result,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "timestamp": now.isoformat()
        }


class TimerTool(BaseTool):
    """
    Set a timer or alarm.
    """
    
    def _get_description(self) -> str:
        return """
        Set a timer for a specific duration (e.g., "set a 5 minute timer").
        The timer will notify when the time is up.
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "minutes": {
                    "type": "number",
                    "description": "Number of minutes for the timer"
                },
                "seconds": {
                    "type": "number",
                    "description": "Number of seconds for the timer"
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the timer (e.g., 'cooking', 'break')",
                    "default": "Timer"
                }
            },
            "required": ["minutes"]
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
    
    async def execute(self, minutes: float, seconds: int = 0, label: str = "Timer") -> Dict[str, Any]:
        """Set a timer."""
        total_seconds = int(minutes * 60 + seconds)
        
        import asyncio
        
        print(f"⏱️  Timer set: {label} for {minutes} minutes")
        
        # Start timer in background
        asyncio.create_task(self._run_timer(total_seconds, label))
        
        return {
            "success": True,
            "message": f"Timer '{label}' set for {minutes} minutes",
            "duration_seconds": total_seconds,
            "label": label
        }
    
    async def _run_timer(self, seconds: int, label: str):
        """Run the timer in background."""
        await asyncio.sleep(seconds)
        print(f"\n⏰ TIMER DONE: {label}!")
        # Could add audio notification here
