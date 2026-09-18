import os
import shutil
import difflib
import json
import time

BACKUP_DIR = ".backup"
DIFFS_FILE = "diffs.json"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_target_files():
    """Returns a list of relative file paths to track changes for."""
    exts = {".py", ".json", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml", ".bat", ".ini", ".cfg"}
    files = []
    for root, dirs, filenames in os.walk(PROJECT_DIR):
        if ".backup" in root or "__pycache__" in root or "venv" in root or ".venv" in root or ".git" in root:
            continue
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, PROJECT_DIR)
                files.append(rel_path)
    return sorted(files)

def backup_files():
    """Creates a backup of current project files."""
    backup_path = os.path.join(PROJECT_DIR, BACKUP_DIR)
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    os.makedirs(backup_path, exist_ok=True)
    for relpath in get_target_files():
        full_path = os.path.join(PROJECT_DIR, relpath)
        backup_full_path = os.path.join(backup_path, relpath)
        os.makedirs(os.path.dirname(backup_full_path), exist_ok=True)
        shutil.copy2(full_path, backup_full_path)
    print(f"[Diff Engine] Backup created at {BACKUP_DIR}/")

def compute_diffs():
    """Computes unified diffs between BACKUP_DIR and current project."""
    all_diffs = []
    backup_path = os.path.join(PROJECT_DIR, BACKUP_DIR)
    for relpath in get_target_files():
        backup_file = os.path.join(backup_path, relpath)
        current_file = os.path.join(PROJECT_DIR, relpath)
        if not os.path.exists(backup_file):
            continue
        with open(backup_file, "r", encoding="utf-8", errors="ignore") as f:
            backup_lines = f.readlines()
        with open(current_file, "r", encoding="utf-8", errors="ignore") as f:
            current_lines = f.readlines()
        
        if backup_lines != current_lines:
            diff = difflib.unified_diff(
                backup_lines, current_lines,
                fromfile=f"backup/{relpath}",
                tofile=relpath,
                lineterm=""
            )
            diff_text = "\n".join(diff)
            if diff_text.strip():
                all_diffs.append({
                    "file": relpath,
                    "diff": diff_text,
                    "timestamp": time.time()
                })
    return all_diffs

def save_diffs(diffs):
    """Saves diffs to diffs.json."""
    diffs_path = os.path.join(PROJECT_DIR, DIFFS_FILE)
    with open(diffs_path, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=2)
    print(f"[Diff Engine] Diffs saved to {DIFFS_FILE}")

def run_diff_pipeline():
    """Runs the full backup -> opencode -> diff pipeline."""
    backup_files()
    diffs = compute_diffs()
    save_diffs(diffs)
