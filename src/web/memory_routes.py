"""API endpoints for memory management."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory_manager import get_memory_manager

router = APIRouter(prefix="/api/memory", tags=["Memory"])


@router.get("/dates")
async def get_memory_dates():
    """Get all dates with conversation data."""
    try:
        memory_mgr = get_memory_manager()
        dates = memory_mgr.get_all_dates()
        return {"type": "memory_dates", "dates": dates, "count": len(dates)}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"type": "error", "message": str(e)}
        )


@router.get("/sessions/{date_str}")
async def get_sessions_for_date(date_str: str):
    """Get all sessions for a specific date."""
    try:
        # Validate date format
        datetime.strptime(date_str, "%Y-%m-%d")

        memory_mgr = get_memory_manager()
        sessions = memory_mgr.get_sessions_for_date(date_str)

        return {
            "type": "sessions",
            "date": date_str,
            "sessions": [
                {
                    "session_id": s["session_id"],
                    "file": s["file"],
                    "lines": s["lines"],
                    "preview": s["content"][:500] + "..."
                    if len(s["content"]) > 500
                    else s["content"],
                }
                for s in sessions
            ],
            "count": len(sessions),
        }
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"type": "error", "message": str(e)}
        )


@router.get("/summary/{date_str}")
async def get_summary(date_str: str):
    """Get summary for a specific date."""
    try:
        # Validate date format
        datetime.strptime(date_str, "%Y-%m-%d")

        memory_mgr = get_memory_manager()
        summary = memory_mgr.get_summary(date_str)

        if summary is None:
            return JSONResponse(
                status_code=404,
                content={
                    "type": "not_found",
                    "message": f"No summary found for {date_str}",
                },
            )

        return {"type": "summary", "date": date_str, "content": summary}
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"type": "error", "message": str(e)}
        )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get full content of a specific session."""
    try:
        memory_mgr = get_memory_manager()

        # Search for session across all dates
        dates = memory_mgr.get_all_dates()
        for date_str in dates:
            sessions = memory_mgr.get_sessions_for_date(date_str)
            for session in sessions:
                if session["session_id"] == session_id:
                    return {
                        "type": "session",
                        "session_id": session_id,
                        "date": date_str,
                        "content": session["content"],
                    }

        return JSONResponse(
            status_code=404,
            content={"type": "not_found", "message": f"Session {session_id} not found"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"type": "error", "message": str(e)}
        )


@router.post("/search")
async def search_memory(
    query: str, date_from: str = None, date_to: str = None, limit: int = 10
):
    """Search through conversation history."""
    try:
        if not query:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "message": "Query is required"},
            )

        memory_mgr = get_memory_manager()
        results = memory_mgr.search_conversations(query, date_from, date_to)

        # Limit results
        if limit and len(results) > limit:
            results = results[:limit]

        return {
            "type": "search_results",
            "query": query,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"type": "error", "message": str(e)}
        )


@router.get("/today")
async def get_today_summary():
    """Get today's summary and sessions."""
    try:
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")

        memory_mgr = get_memory_manager()
        sessions = memory_mgr.get_sessions_for_date(today)
        summary = memory_mgr.get_summary(today)

        return {
            "type": "today",
            "date": today,
            "sessions": [
                {"session_id": s["session_id"], "lines": s["lines"]} for s in sessions
            ],
            "has_summary": summary is not None,
            "summary": summary,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"type": "error", "message": str(e)}
        )
