"""
Timer and utility tools for Talkie Voice Assistant.
Provides timer, alarm, and reminder functionality.
"""

import asyncio
import time
import threading
from typing import Any, Dict
from datetime import datetime, timedelta
from tools import BaseTool


class TimerTool(BaseTool):
    """
    Timer tool for setting countdowns and alarms.
    Useful for cooking, workouts, reminders, etc.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.active_timers = {}
        self.timer_counter = 0
        
    def _get_description(self) -> str:
        return """
        Set timers and alarms. Useful for reminders, cooking, workouts, etc.
        
        Examples:
        - "Set a 5 minute timer"
        - "Timer for 30 seconds"
        - "Alarm in 2 hours"
        - "Cancel timer 1"
        
        Returns: Timer details including remaining time and status.
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "list", "status"],
                    "description": "Action to perform"
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration in seconds (for 'start' action)"
                },
                "minutes": {
                    "type": "integer",
                    "description": "Duration in minutes (alternative to seconds)"
                },
                "hours": {
                    "type": "integer",
                    "description": "Duration in hours (alternative to seconds)"
                },
                "timer_id": {
                    "type": "string",
                    "description": "Timer ID (for 'stop' or 'status' actions)"
                },
                "label": {
                    "type": "string",
                    "description": "Optional label/name for the timer"
                }
            },
            "required": ["action"]
        }
    
    async def execute(self, action: str, duration: int = None, minutes: int = None,
                     hours: int = None, timer_id: str = None, label: str = None) -> Dict[str, Any]:
        """Execute timer action."""
        
        if action == "start":
            return await self._start_timer(duration, minutes, hours, label)
        
        elif action == "stop":
            return await self._stop_timer(timer_id)
        
        elif action == "list":
            return await self._list_timers()
        
        elif action == "status":
            return await self._get_status(timer_id)
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _start_timer(self, duration: int = None, minutes: int = None, 
                          hours: int = None, label: str = None) -> Dict[str, Any]:
        """Start a new timer."""
        # Calculate total duration in seconds
        total_seconds = 0
        if duration:
            total_seconds += duration
        if minutes:
            total_seconds += minutes * 60
        if hours:
            total_seconds += hours * 3600
        
        if total_seconds <= 0:
            return {"error": "Invalid duration. Please specify seconds, minutes, or hours."}
        
        # Generate timer ID
        self.timer_counter += 1
        timer_id = f"timer_{self.timer_counter}"
        
        # Calculate end time
        start_time = time.time()
        end_time = start_time + total_seconds
        
        # Store timer info
        self.active_timers[timer_id] = {
            "id": timer_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": total_seconds,
            "label": label or f"Timer {self.timer_counter}",
            "status": "running"
        }
        
        # Start timer thread (non-blocking)
        timer_thread = threading.Thread(
            target=self._timer_worker,
            args=(timer_id, total_seconds),
            daemon=True
        )
        timer_thread.start()
        
        # Format duration for display
        duration_str = self._format_duration(total_seconds)
        
        return {
            "success": True,
            "timer_id": timer_id,
            "label": label or f"Timer {self.timer_counter}",
            "duration": total_seconds,
            "duration_formatted": duration_str,
            "status": "running",
            "message": f"Started {duration_str} timer (ID: {timer_id})"
        }
    
    def _timer_worker(self, timer_id: str, duration: int):
        """Worker thread that waits for timer to complete."""
        time.sleep(duration)
        
        if timer_id in self.active_timers:
            self.active_timers[timer_id]["status"] = "completed"
            # Here you could trigger an alarm sound or notification
            # For now, we just mark it as completed
    
    async def _stop_timer(self, timer_id: str) -> Dict[str, Any]:
        """Stop/cancel a timer."""
        if not timer_id:
            return {"error": "Please specify a timer_id to stop"}
        
        if timer_id not in self.active_timers:
            return {"error": f"Timer {timer_id} not found"}
        
        timer = self.active_timers[timer_id]
        timer["status"] = "stopped"
        
        return {
            "success": True,
            "timer_id": timer_id,
            "status": "stopped",
            "message": f"Stopped timer: {timer['label']}"
        }
    
    async def _list_timers(self) -> Dict[str, Any]:
        """List all active timers."""
        active = []
        completed = []
        
        current_time = time.time()
        
        for timer_id, timer in self.active_timers.items():
            if timer["status"] == "running":
                remaining = max(0, timer["end_time"] - current_time)
                active.append({
                    "id": timer_id,
                    "label": timer["label"],
                    "remaining_seconds": int(remaining),
                    "remaining_formatted": self._format_duration(int(remaining))
                })
            elif timer["status"] == "completed":
                completed.append({
                    "id": timer_id,
                    "label": timer["label"],
                    "status": "completed"
                })
        
        return {
            "active_timers": active,
            "completed_timers": completed,
            "total_count": len(self.active_timers)
        }
    
    async def _get_status(self, timer_id: str) -> Dict[str, Any]:
        """Get status of a specific timer."""
        if not timer_id:
            return {"error": "Please specify a timer_id"}
        
        if timer_id not in self.active_timers:
            return {"error": f"Timer {timer_id} not found"}
        
        timer = self.active_timers[timer_id]
        current_time = time.time()
        
        if timer["status"] == "running":
            remaining = max(0, timer["end_time"] - current_time)
            elapsed = current_time - timer["start_time"]
            
            return {
                "id": timer_id,
                "label": timer["label"],
                "status": "running",
                "remaining_seconds": int(remaining),
                "remaining_formatted": self._format_duration(int(remaining)),
                "elapsed_seconds": int(elapsed),
                "elapsed_formatted": self._format_duration(int(elapsed)),
                "total_duration": timer["duration"],
                "total_duration_formatted": self._format_duration(timer["duration"])
            }
        else:
            return {
                "id": timer_id,
                "label": timer["label"],
                "status": timer["status"]
            }
    
    def _format_duration(self, seconds: int) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            if secs > 0:
                return f"{minutes} minute{'s' if minutes != 1 else ''} and {secs} second{'s' if secs != 1 else ''}"
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            if minutes > 0:
                return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
            return f"{hours} hour{'s' if hours != 1 else ''}"


class CalculatorTool(BaseTool):
    """
    Simple calculator tool for math operations.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def _get_description(self) -> str:
        return """
        Perform basic mathematical calculations.
        Supports: addition, subtraction, multiplication, division, power, square root.
        
        Examples:
        - "Calculate 15 * 23"
        - "What is 100 divided by 4?"
        - "Square root of 144"
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '15 * 23')"
                },
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
                    "description": "Specific operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number (for specific operations)"
                },
                "b": {
                    "type": "number",
                    "description": "Second number (for specific operations)"
                }
            },
            "required": []
        }
    
    async def execute(self, expression: str = None, operation: str = None, 
                     a: float = None, b: float = None) -> Dict[str, Any]:
        """Execute calculation."""
        
        try:
            if expression:
                # Clean the expression for safety
                allowed_chars = set('0123456789+-*/.()^ sqrt')
                if not all(c in allowed_chars for c in expression.replace(' ', '')):
                    return {"error": "Invalid characters in expression"}
                
                # Replace ^ with ** for Python evaluation
                expression = expression.replace('^', '**')
                expression = expression.replace('sqrt', '**(0.5)')
                
                # Evaluate safely
                result = eval(expression, {"__builtins__": {}}, {})
                
                return {
                    "success": True,
                    "expression": expression,
                    "result": result,
                    "formatted": f"{expression} = {result}"
                }
            
            elif operation and a is not None:
                # Specific operation mode
                if operation == "add":
                    result = a + (b or 0)
                elif operation == "subtract":
                    result = a - (b or 0)
                elif operation == "multiply":
                    result = a * (b or 1)
                elif operation == "divide":
                    if b == 0:
                        return {"error": "Cannot divide by zero"}
                    result = a / b
                elif operation == "power":
                    result = a ** (b or 2)
                elif operation == "sqrt":
                    if a < 0:
                        return {"error": "Cannot calculate square root of negative number"}
                    result = a ** 0.5
                else:
                    return {"error": f"Unknown operation: {operation}"}
                
                return {
                    "success": True,
                    "operation": operation,
                    "a": a,
                    "b": b,
                    "result": result
                }
            
            else:
                return {"error": "Please provide either an expression or operation with values"}
                
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
