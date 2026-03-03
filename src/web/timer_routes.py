"""Timer API routes for Talkie Web Control Panel."""

from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime

from plugins.cron_timer import get_timer_store

router = APIRouter(prefix="/api/timer", tags=["Timer"])


@router.get("/list")
async def list_timers():
    """List all active timers."""
    try:
        store = get_timer_store()
        timers = store.get_active_timers()

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
            "type": "timers_list",
            "success": True,
            "count": len(result_timers),
            "timers": result_timers,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "type": "error",
            "success": False,
            "message": f"Failed to list timers: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


@router.post("/set")
async def set_timer(
    duration: int,
    unit: str = "seconds",
    description: Optional[str] = None,
):
    """Set a new timer."""
    try:
        store = get_timer_store()

        # Convert to seconds
        if unit == "minutes":
            duration_seconds = duration * 60
        elif unit == "hours":
            duration_seconds = duration * 3600
        else:
            duration_seconds = duration

        if duration_seconds <= 0:
            return {
                "type": "error",
                "success": False,
                "message": "Duration must be positive",
                "timestamp": datetime.now().isoformat(),
            }

        # Create timer
        timer = store.create_timer(duration_seconds, description)

        return {
            "type": "timer_set",
            "success": True,
            "message": f"Timer set for {duration} {unit}",
            "timer": timer.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "type": "error",
            "success": False,
            "message": f"Failed to set timer: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


@router.post("/cancel/{timer_id}")
async def cancel_timer(timer_id: int):
    """Cancel a timer by ID."""
    try:
        store = get_timer_store()
        cancelled = store.cancel_timer(timer_id)

        if cancelled:
            return {
                "type": "timer_cancelled",
                "success": True,
                "message": f"Timer {timer_id} cancelled",
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "type": "error",
                "success": False,
                "message": f"Timer {timer_id} not found or already inactive",
                "timestamp": datetime.now().isoformat(),
            }
    except Exception as e:
        return {
            "type": "error",
            "success": False,
            "message": f"Failed to cancel timer: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
