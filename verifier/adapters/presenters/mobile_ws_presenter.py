import json
import asyncio
import threading
from typing import Set, Optional
from ...ports.presenter import IReviewPresenterPort
from ...domain.entities import ReviewSession, DiffHunk


def _session_to_dict(session: ReviewSession) -> dict:
    """Serialize a ReviewSession for JSON transport over WebSocket."""
    return {
        "session_id": session.session_id,
        "workspace_root": session.workspace_root,
        "state": session.state.value,
        "current_hunk_index": session.current_hunk_index,
        "total_hunks": len(session.all_hunks),
        "pending": session.pending_count,
        "accepted": session.accepted_count,
        "rejected": session.rejected_count,
        "files": [
            {
                "file_path": fd.file_path,
                "change_type": fd.change_type.value,
                "hunk_count": len(fd.hunks)
            }
            for fd in session.file_diffs
        ]
    }


def _hunk_to_dict(hunk: DiffHunk) -> dict:
    """Serialize a DiffHunk for JSON transport over WebSocket."""
    return {
        "hunk_id": hunk.hunk_id,
        "file_path": hunk.file_path,
        "old_start": hunk.old_start,
        "old_lines": hunk.old_lines,
        "new_start": hunk.new_start,
        "new_lines": hunk.new_lines,
        "header": hunk.header,
        "diff_text": hunk.diff_text,
        "status": hunk.status.value,
        "explanation": hunk.explanation,
    }


class MobileWSPresenter(IReviewPresenterPort):
    """
    Broadcasts review session state changes to connected iOS / mobile
    clients via WebSocket connections.
    
    This presenter does not own the WebSocket server — it receives a
    set of active WebSocket connections from the gateway API router and
    pushes JSON messages to all connected clients.
    """

    def __init__(self):
        self._connections: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def register_connection(self, ws) -> None:
        """Register a new WebSocket connection from a mobile client."""
        self._connections.add(ws)
        print(f"[MobileWS] Client connected. Total: {len(self._connections)}")

    def unregister_connection(self, ws) -> None:
        """Unregister a disconnected WebSocket connection."""
        self._connections.discard(ws)
        print(f"[MobileWS] Client disconnected. Total: {len(self._connections)}")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop used by the FastAPI server."""
        self._loop = loop

    def on_session_started(self, session: ReviewSession) -> None:
        """Broadcast session start to all connected mobile clients."""
        self._broadcast({
            "event": "session_started",
            "session": _session_to_dict(session),
        })

    def on_hunk_displayed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        """Push the current hunk data to all connected mobile clients."""
        self._broadcast({
            "event": "hunk_displayed",
            "session": _session_to_dict(session),
            "hunk": _hunk_to_dict(hunk),
        })

    def on_hunk_processed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        """Notify mobile clients that a hunk decision was recorded."""
        self._broadcast({
            "event": "hunk_processed",
            "session": _session_to_dict(session),
            "hunk": _hunk_to_dict(hunk),
        })

    def on_session_completed(self, session: ReviewSession) -> None:
        """Broadcast session completion to all connected mobile clients."""
        self._broadcast({
            "event": "session_completed",
            "session": _session_to_dict(session),
        })

    def on_explanation_ready(self, hunk: DiffHunk, explanation: str) -> None:
        """Push a hunk explanation to mobile clients."""
        self._broadcast({
            "event": "explanation_ready",
            "hunk": _hunk_to_dict(hunk),
            "explanation": explanation,
        })

    def _broadcast(self, payload: dict) -> None:
        """Send a JSON message to all connected WebSocket clients."""
        if not self._connections:
            return

        message = json.dumps(payload)
        dead = set()

        for ws in self._connections.copy():
            try:
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.send_text(message), self._loop)
                else:
                    # Fallback: try direct send if we're already in an async context
                    asyncio.ensure_future(ws.send_text(message))
            except Exception as e:
                print(f"[MobileWS] Failed to send to client: {e}")
                dead.add(ws)

        self._connections -= dead
