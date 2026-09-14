from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import json
import os
import shutil
import asyncio
import difflib
import time

app = FastAPI(title="JARVIS Code Review UI")

DIFFS_FILE = "diffs.json"
BACKUP_DIR = ".backup"

# ============================================================
# DIFF CALCULATION HELPERS
# ============================================================

def get_python_files():
    """Returns a list of .py, .json, .md, .txt, .js, .ts files in the project."""
    exts = {".py", ".json", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml", ".bat"}
    files = []
    for root, dirs, filenames in os.walk("."):
        if ".backup" in dirs:
            dirs.remove(".backup")
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        if ".venv" in dirs or "venv" in dirs:
            dirs.append("venv") if "venv" in dirs else None
            dirs.remove("venv") if "venv" in dirs else None
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    return sorted(files)

def compute_diffs():
    """Computes unified diffs between BACKUP_DIR and current project."""
    all_diffs = []
    files = get_python_files()
    for filepath in files:
        backup_path = os.path.join(BACKUP_DIR, filepath)
        if not os.path.exists(backup_path):
            continue
        with open(backup_path, "r", encoding="utf-8", errors="ignore") as f:
            backup_lines = f.readlines()
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            current_lines = f.readlines()
        
        if backup_lines != current_lines:
            diff = difflib.unified_diff(
                backup_lines, current_lines,
                fromfile=f"backup/{filepath}",
                tofile=filepath,
                lineterm=""
            )
            diff_text = "\n".join(diff)
            if diff_text.strip():
                all_diffs.append({
                    "file": filepath,
                    "diff": diff_text,
                    "timestamp": time.time()
                })
    return all_diffs

def backup_files():
    """Creates a backup of current project files."""
    if os.path.exists(BACKUP_DIR):
        import shutil
        shutil.rmtree(BACKUP_DIR)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for root, dirs, files in os.walk("."):
        if ".backup" in root or "__pycache__" in root or "venv" in root:
            continue
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(BACKUP_DIR, root, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

# ============================================================
# API ENDPOINTS
# ============================================================

@app.post("/api/backup")
async def backup():
    """Manually triggers a backup of current files."""
    backup_files()
    return JSONResponse(content={"status": "Backup created"})

@app.get("/api/diffs")
async def get_diffs():
    """Returns the latest calculated diffs as JSON."""
    if os.path.exists(DIFFS_FILE):
        with open(DIFFS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content=[])

@app.post("/api/calculate")
async def calculate():
    """Calculates diffs and saves them to diffs.json."""
    diffs = compute_diffs()
    with open(DIFFS_FILE, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=2)
    return JSONResponse(content=diffs)

@app.post("/api/accept")
async def accept_file(request: Request):
    """Restores the file from backup, effectively accepting the changes."""
    data = await request.json()
    file = data.get("file")
    backup_path = os.path.join(PROJECT_DIR, BACKUP_DIR, file)
    current_path = os.path.join(PROJECT_DIR, file)
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, current_path)
        print(f"[UI] Accepted changes for {file}")
    return JSONResponse(content={"status": "accepted"})

@app.post("/api/reject")
async def reject_file(request: Request):
    """Restores the file from backup, effectively rejecting the changes."""
    data = await request.json()
    file = data.get("file")
    backup_path = os.path.join(PROJECT_DIR, BACKUP_DIR, file)
    current_path = os.path.join(PROJECT_DIR, file)
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, current_path)
        print(f"[UI] Rejected changes for {file}")
    return JSONResponse(content={"status": "rejected"})

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
            .file-section {
                margin-bottom: 20px;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                overflow: hidden;
            }
            .file-header {
                background: #007acc;
                color: white;
                padding: 10px 15px;
                font-weight: bold;
                font-size: 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .file-actions {
                display: flex;
                gap: 8px;
            }
            .btn {
                background: #ffffff33;
                border: 1px solid #ffffff55;
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s;
            }
            .btn:hover {
                background: #ffffff55;
            }
            .btn-accept {
                background: #28a745;
                border-color: #218838;
            }
            .btn-accept:hover {
                background: #218838;
            }
            .btn-reject {
                background: #dc3545;
                border-color: #c82333;
            }
            .btn-reject:hover {
                background: #c82333;
            }
            .diff-content {
                display: none;
                padding: 15px;
                font-family: 'Courier New', Courier, monospace;
                font-size: 13px;
                line-height: 1.5;
                background: #1e1e1e;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .diff-line {
                padding: 2px 0;
            }
            .diff-add {
                background: #1e5631;
                color: #cdcaa9;
                padding-left: 10px;
                border-left: 3px solid #28a745;
            }
            .diff-del {
                background: #521d24;
                color: #f0dcc8;
                padding-left: 10px;
                border-left: 3px solid #dc3545;
            }
            .diff-ctx {
                color: #808080;
                padding-left: 10px;
            }
            .status-bar {
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                background: #3c3c3c;
                border-radius: 4px;
                font-size: 14px;
                color: #cccccc;
            }
            .pulse {
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { opacity: 0.6; }
                50% { opacity: 1; }
                100% { opacity: 0.6; }
            }
        </style>
    </head>
    <body>
        <h1>🧠 JARVIS Code Review Dashboard</h1>
        <div id="container">
            <div id="content"></div>
            <div class="status-bar" id="status">Listening for code changes...</div>
        </div>

        <script>
            const contentDiv = document.getElementById('content');
            const statusBar = document.getElementById('status');
            
            async function fetchDiffs() {
                try {
                    const res = await fetch('/api/diffs');
                    const diffs = await res.json();
                    renderDiffs(diffs);
                } catch (e) {
                    console.error('Failed to fetch diffs:', e);
                }
            }

            function renderDiffs(diffs) {
                if (diffs.length === 0) {
                    contentDiv.innerHTML = '<p style="text-align:center; color:#888;">No code changes detected yet. Say "code" to JARVIS to start.</p>';
                    statusBar.innerText = "Waiting for code changes...";
                    return;
                }

                statusBar.innerText = `${diffs.length} file(s) with changes detected. Review one by one.`;
                contentDiv.innerHTML = '';

                diffs.forEach((diff, index) => {
                    const fileSection = document.createElement('div');
                    fileSection.className = 'file-section';
                    
                    const header = document.createElement('div');
                    header.className = 'file-header';
                    header.innerHTML = `
                        <span>📄 ${diff.file}</span>
                        <div class="file-actions">
                            <button class="btn" onclick="toggleDiff(${index})">View Diff</button>
                            <button class="btn btn-accept" onclick="acceptFile('${diff.file}')">Accept</button>
                            <button class="btn btn-reject" onclick="rejectFile('${diff.file}')">Reject</button>
                        </div>
                    `;
                    
                    const diffDiv = document.createElement('div');
                    diffDiv.className = 'diff-content';
                    diffDiv.id = 'diff-' + index;
                    
                    const lines = diff.diff.split('\\n');
                    lines.forEach(line => {
                        const lineDiv = document.createElement('div');
                        lineDiv.className = 'diff-line';
                        if (line.startsWith('+++') || line.startsWith('---')) {
                            lineDiv.className += ' diff-ctx';
                        } else if (line.startsWith('+')) {
                            lineDiv.className += ' diff-add';
                            lineDiv.innerText = line;
                        } else if (line.startsWith('-')) {
                            lineDiv.className += ' diff-del';
                            lineDiv.innerText = line;
                        } else {
                            lineDiv.className += ' diff-ctx';
                            lineDiv.innerText = line;
                        }
                        diffDiv.appendChild(lineDiv);
                    });
                    
                    fileSection.appendChild(header);
                    fileSection.appendChild(diffDiv);
                    contentDiv.appendChild(fileSection);
                });
            }

            function toggleDiff(index) {
                const diffDiv = document.getElementById('diff-' + index);
                if (diffDiv.style.display === 'block') {
                    diffDiv.style.display = 'none';
                } else {
                    diffDiv.style.display = 'block';
                }
            }

            function acceptFile(file) {
                if (confirm(`Accept all changes in ${file}?`)) {
                    fetch('/api/accept?file=' + encodeURIComponent(file))
                        .then(() => alert('Accepted!'));
                }
            }

            function rejectFile(file) {
                if (confirm(`Reject all changes in ${file}?`)) {
                    fetch('/api/reject?file=' + encodeURIComponent(file))
                        .then(() => { location.reload(); });
                }
            }

            // Poll every 3 seconds
            setInterval(fetchDiffs, 3000);
            fetchDiffs();
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

# We need fastapi Response for the HTML response
from fastapi.responses import Response
