from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import subprocess
import json
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="JARVIS Code Review")

def get_git_diff():
    """Returns the current git diff output."""
    result = subprocess.run(
        ["git", "diff"],
        cwd=PROJECT_DIR,
        capture_output=True, text=True,
        timeout=30
    )
    return result.stdout

def get_git_status():
    """Returns git status output."""
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_DIR,
        capture_output=True, text=True,
        timeout=10
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []

@app.get("/api/diffs")
async def get_diffs():
    """Returns the current git diff."""
    diff = get_git_diff()
    return JSONResponse(content={"diff": diff})

@app.post("/api/refresh")
async def refresh():
    """Returns updated git status and diff."""
    diff = get_git_diff()
    status = get_git_status()
    return JSONResponse(content={"diff": diff, "status": status})

@app.post("/api/accept")
async def accept_changes():
    """Stage all changes (accept)."""
    subprocess.run(["git", "add", "-A"], cwd=PROJECT_DIR, timeout=30)
    return JSONResponse(content={"status": "Changes staged"})

@app.post("/api/reject")
async def reject_file(request: Request):
    """Restores the file from backup, effectively rejecting the changes."""
    data = await request.json()
    file = data.get("file")
    backup_path = os.path.join(PROJECT_DIR, BACKUP_DIR, file)
    current_path = os.path.join(PROJECT_DIR, file)
    if os.path.exists(backup_path):
        import shutil
        shutil.copy2(backup_path, current_path)
        print(f"[UI] Rejected changes for {file}")
    return JSONResponse(content={"status": "rejected"})

@app.post("/api/reject_all")
async def reject_changes():
    """Discard all working tree changes (reject)."""
    subprocess.run(["git", "checkout", "--", "."], cwd=PROJECT_DIR, timeout=30)
    return JSONResponse(content={"status": "Changes discarded"})

@app.post("/api/feedback")
async def receive_feedback(request: Request):
    """Receives correct intent from the UI and adds it to the FAISS index."""
    from adapters.vector_store import FaissAdapter
    from adapters.embeddings import HuggingFaceAdapter
    from use_cases.routing import LearnFromFeedbackUseCase
    
    data = await request.json()
    text = data.get("text")
    intent = data.get("intent")
    
    if text and intent:
        vector_store = FaissAdapter()
        embedding_engine = HuggingFaceAdapter()
        feedback_uc = LearnFromFeedbackUseCase(vector_store, embedding_engine)
        feedback_uc.execute(text, intent)
        return JSONResponse(content={"status": "success", "message": "Learned new mapping"})
    return JSONResponse(content={"status": "error", "message": "Missing text or intent"}, status_code=400)

# ============================================================
# UI SERVING
# ============================================================

@app.get("/")
async def serve_ui(request: Request):
    """Serves the main UI page."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JARVIS Code Review</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 20px;
            }
            h1 {
                color: #007acc;
                text-align: center;
                margin-bottom: 20px;
                text-shadow: 0 0 10px rgba(0, 122, 204, 0.5);
            }
            #container {
                max-width: 1000px;
                margin: 0 auto;
                background: #252526;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            }
            .diff-content {
                padding: 15px;
                font-family: 'Courier New', Courier, monospace;
                font-size: 13px;
                line-height: 1.5;
                background: #1e1e1e;
                white-space: pre-wrap;
                word-wrap: break-word;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
            }
            .diff-add { color: #b5cea8; }
            .diff-del { color: #f14c4c; }
            .diff-ctx { color: #808080; }
            .status-bar {
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                background: #3c3c3c;
                border-radius: 4px;
                font-size: 14px;
                color: #cccccc;
            }
            .btn {
                background: #ffffff33;
                border: 1px solid #ffffff55;
                color: white;
                padding: 6px 14px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                margin: 4px;
            }
            .btn:hover { background: #ffffff55; }
            .btn-accept { background: #28a745; border-color: #218838; }
            .btn-reject { background: #dc3545; border-color: #c82333; }
        </style>
    </head>
    <body>
        <h1>🧠 JARVIS Code Review</h1>
        <div id="container">
            <div style="margin-bottom: 15px;">
                <button class="btn btn-accept" onclick="acceptChanges()">Accept All</button>
                <button class="btn btn-reject" onclick="rejectChanges()">Reject All</button>
                <button class="btn" onclick="refresh()">Refresh</button>
            </div>
            <div id="diff" class="diff-content">Loading git diffs...</div>
            <div class="status-bar" id="status">Listening for code changes...</div>
        </div>
        <script>
            async function refresh() {
                try {
                    const res = await fetch('/api/refresh');
                    const data = await res.json();
                    const diffEl = document.getElementById('diff');
                    if (data.diff) {
                        diffEl.textContent = data.diff;
                        document.getElementById('status').textContent = `${data.status.length || 0} changed file(s)`;
                    } else {
                        diffEl.textContent = 'No changes detected.';
                        document.getElementById('status').textContent = 'All changes committed.';
                    }
                } catch (e) { console.error('Failed:', e); }
            }
            function acceptChanges() { fetch('/api/accept').then(() => refresh()); }
            function rejectChanges() { fetch('/api/reject').then(() => refresh()); }
            setInterval(refresh, 3000);
            refresh();
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")
