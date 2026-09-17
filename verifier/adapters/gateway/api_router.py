import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, Callable
import os

from ...use_cases.coordinator import ReviewSessionCoordinator
from ..presenters.mobile_ws_presenter import MobileWSPresenter, _session_to_dict, _hunk_to_dict


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_review_router(
    coordinator: ReviewSessionCoordinator,
    mobile_presenter: MobileWSPresenter
) -> APIRouter:
    """
    Creates a FastAPI APIRouter providing REST and WebSocket endpoints
    for the JARVIS code review gateway.
    
    REST endpoints are used by mobile clients for initial state fetch
    and command submission. The WebSocket channel provides real-time
    push updates for live hunk synchronization.
    """
    router = APIRouter(prefix="/api/review", tags=["Code Review"])

    # ------------------------------------------------------------------
    # REST Endpoints
    # ------------------------------------------------------------------

    @router.get("/status")
    async def get_review_status():
        """Returns the current review session state, or null if idle."""
        session = coordinator.active_session
        if not session:
            return JSONResponse(content={"session": None})
        
        current = session.current_hunk
        return JSONResponse(content={
            "session": _session_to_dict(session),
            "current_hunk": _hunk_to_dict(current) if current else None,
        })

    @router.get("/hunks")
    async def get_all_hunks():
        """Returns all hunks in the active review session."""
        session = coordinator.active_session
        if not session:
            return JSONResponse(content={"hunks": []})
        
        return JSONResponse(content={
            "hunks": [_hunk_to_dict(h) for h in session.all_hunks]
        })

    @router.post("/accept/{hunk_id}")
    async def accept_hunk(hunk_id: str):
        """Accept a specific hunk by ID."""
        coordinator.accept_hunk(hunk_id)
        return JSONResponse(content={"status": "ok", "action": "accepted", "hunk_id": hunk_id})

    @router.post("/reject/{hunk_id}")
    async def reject_hunk(hunk_id: str):
        """Reject a specific hunk by ID."""
        coordinator.reject_hunk(hunk_id)
        return JSONResponse(content={"status": "ok", "action": "rejected", "hunk_id": hunk_id})

    @router.post("/accept-all")
    async def accept_all():
        """Accept all remaining pending hunks."""
        coordinator.accept_all()
        return JSONResponse(content={"status": "ok", "action": "accept_all"})

    @router.post("/reject-all")
    async def reject_all():
        """Reject all remaining pending hunks (discard all changes)."""
        coordinator.reject_all()
        return JSONResponse(content={"status": "ok", "action": "reject_all"})

    @router.post("/skip")
    async def skip_hunk():
        """Skip the current hunk."""
        coordinator.skip_hunk()
        return JSONResponse(content={"status": "ok", "action": "skipped"})

    @router.post("/explain/{hunk_id}")
    async def explain_hunk(hunk_id: str):
        """Request an LLM explanation for a specific hunk."""
        coordinator.explain_hunk(hunk_id)
        return JSONResponse(content={"status": "ok", "action": "explain", "hunk_id": hunk_id})

    @router.post("/voice-command")
    async def voice_command(payload: dict):
        """
        Webhook endpoint for iOS Siri / Apple Shortcuts.
        Expects: {"transcript": "accept this block"}
        """
        transcript = payload.get("transcript", "")
        if not transcript:
            return JSONResponse(content={"error": "No transcript provided"}, status_code=400)
        
        # Route through the voice grammar if available
        return JSONResponse(content={"status": "ok", "transcript": transcript})

    # ------------------------------------------------------------------
    # WebSocket Channel
    # ------------------------------------------------------------------

    @router.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket):
        """
        Real-time WebSocket channel for mobile clients.
        
        On connect: sends current session state.
        Listens for incoming commands: {"action": "accept|reject|...", "hunk_id": "..."}
        Pushes hunk updates via MobileWSPresenter.
        """
        await websocket.accept()
        mobile_presenter.set_event_loop(asyncio.get_event_loop())
        mobile_presenter.register_connection(websocket)

        # Send current state on connect
        session = coordinator.active_session
        if session:
            current = session.current_hunk
            await websocket.send_json({
                "event": "sync",
                "session": _session_to_dict(session),
                "current_hunk": _hunk_to_dict(current) if current else None,
            })

        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    action = msg.get("action", "")
                    hunk_id = msg.get("hunk_id")

                    if action == "accept" and hunk_id:
                        coordinator.accept_hunk(hunk_id)
                    elif action == "reject" and hunk_id:
                        coordinator.reject_hunk(hunk_id)
                    elif action == "accept_all":
                        coordinator.accept_all()
                    elif action == "reject_all":
                        coordinator.reject_all()
                    elif action == "skip":
                        coordinator.skip_hunk()
                    elif action == "explain" and hunk_id:
                        coordinator.explain_hunk(hunk_id)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})
        except WebSocketDisconnect:
            mobile_presenter.unregister_connection(websocket)
        except Exception as e:
            print(f"[ReviewGateway] WebSocket error: {e}")
            mobile_presenter.unregister_connection(websocket)

    return router


def create_mobile_app_router() -> APIRouter:
    """
    Serves the mobile PWA / web client static files.
    Mount this at the root to serve index.html at '/review'.
    """
    router = APIRouter(tags=["Mobile UI"])

    @router.get("/review")
    async def serve_mobile_ui():
        """Serve the mobile review PWA."""
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
        return JSONResponse(content={"error": "Mobile UI not found"}, status_code=404)

    return router
