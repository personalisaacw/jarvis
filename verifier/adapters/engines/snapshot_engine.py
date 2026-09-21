import os
import shutil
import difflib
import uuid
import re
from typing import List, Set, Dict, Tuple
from datetime import datetime

from ...ports.diff_engine import IDiffEnginePort
from ...domain.entities import FileDiff, DiffHunk
from ...domain.enums import ChangeType, HunkStatus

HIDDEN_DIRS = {'.git', '.jarvis', '__pycache__', '.venv', 'venv', 'node_modules'}
BINARY_EXTS = {'.onnx', '.wav', '.png', '.jpg', '.exe', '.dll', '.so', '.zip', '.tar', '.gz', '.pyc'}

class SnapshotDiffEngineAdapter(IDiffEnginePort):
    def __init__(self):
        self.current_token = None

    def _should_skip(self, path: str, is_dir: bool = False) -> bool:
        basename = os.path.basename(path)
        if is_dir:
            return basename in HIDDEN_DIRS or basename.startswith('.')
        
        _, ext = os.path.splitext(basename)
        return ext.lower() in BINARY_EXTS or basename.startswith('.')

    def _get_snapshot_dir(self, workspace_root: str, snapshot_id: str) -> str:
        return os.path.join(workspace_root, '.jarvis', 'snapshots', snapshot_id)

    def _get_workspace_files(self, root: str) -> Set[str]:
        files = set()
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune hidden dirs
            dirnames[:] = [d for d in dirnames if not self._should_skip(os.path.join(dirpath, d), is_dir=True)]
            
            for f in filenames:
                if not self._should_skip(f, is_dir=False):
                    full_path = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(full_path, root)
                    files.add(rel_path)
        return files

    def capture_baseline(self, workspace_root: str) -> str:
        snapshot_id = str(uuid.uuid4())
        self.current_token = snapshot_id
        snapshot_dir = self._get_snapshot_dir(workspace_root, snapshot_id)
        
        print(f"[SnapshotDiffEngine] Capturing baseline {snapshot_id}...")
        
        os.makedirs(snapshot_dir, exist_ok=True)
        
        for dirpath, dirnames, filenames in os.walk(workspace_root):
            # Prune hidden dirs
            dirnames[:] = [d for d in dirnames if not self._should_skip(os.path.join(dirpath, d), is_dir=True)]
            
            for f in filenames:
                if not self._should_skip(f, is_dir=False):
                    src = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(src, workspace_root)
                    dst = os.path.join(snapshot_dir, rel_path)
                    
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        
        print(f"[SnapshotDiffEngine] Baseline {snapshot_id} captured.")
        return snapshot_id

    def _parse_hunks(self, diff_lines: List[str], file_path: str) -> List[DiffHunk]:
        hunks = []
        current_hunk_lines = []
        old_start, old_lines, new_start, new_lines = 0, 0, 0, 0
        header = ""
        
        # Regex for unified diff hunk header
        header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
        
        for line in diff_lines:
            if line.startswith('---') or line.startswith('+++'):
                continue
            
            match = header_re.match(line)
            if match:
                # Save previous hunk if exists
                if current_hunk_lines:
                    hunks.append(DiffHunk(
                        hunk_id=str(uuid.uuid4()),
                        file_path=file_path,
                        old_start=old_start,
                        old_lines=old_lines,
                        new_start=new_start,
                        new_lines=new_lines,
                        header=header,
                        diff_text="".join(current_hunk_lines),
                        status=HunkStatus.PENDING,
                        explanation=""
                    ))
                
                # Parse new hunk header
                old_s, old_l, new_s, new_l, hdr_context = match.groups()
                old_start = int(old_s)
                old_lines = int(old_l) if old_l else 1
                new_start = int(new_s)
                new_lines = int(new_l) if new_l else 1
                header = line.strip()
                current_hunk_lines = [line]
            elif current_hunk_lines:
                current_hunk_lines.append(line)
                
        # Save last hunk
        if current_hunk_lines:
            hunks.append(DiffHunk(
                hunk_id=str(uuid.uuid4()),
                file_path=file_path,
                old_start=old_start,
                old_lines=old_lines,
                new_start=new_start,
                new_lines=new_lines,
                header=header,
                diff_text="".join(current_hunk_lines),
                status=HunkStatus.PENDING,
                explanation=""
            ))
            
        return hunks

    def compute_diff(self, workspace_root: str, baseline_token: str) -> List[FileDiff]:
        self.current_token = baseline_token
        snapshot_dir = self._get_snapshot_dir(workspace_root, baseline_token)
        
        if not os.path.exists(snapshot_dir):
            print(f"[SnapshotDiffEngine] Snapshot {baseline_token} not found!")
            return []
            
        workspace_files = self._get_workspace_files(workspace_root)
        snapshot_files = self._get_workspace_files(snapshot_dir)
        
        added_files = workspace_files - snapshot_files
        deleted_files = snapshot_files - workspace_files
        common_files = workspace_files & snapshot_files
        
        file_diffs = []
        
        # Handle added files
        for f in added_files:
            ws_path = os.path.join(workspace_root, f)
            with open(ws_path, 'r', encoding='utf-8', errors='replace') as ws_f:
                lines = ws_f.readlines()
            
            diff_text = f"@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{line}" for line in lines)
            hunk = DiffHunk(
                hunk_id=str(uuid.uuid4()),
                file_path=f,
                old_start=0, old_lines=0,
                new_start=1, new_lines=len(lines),
                header=f"@@ -0,0 +1,{len(lines)} @@",
                diff_text=diff_text,
                status=HunkStatus.PENDING,
                explanation=""
            )
            file_diffs.append(FileDiff(file_path=f, change_type=ChangeType.ADDED, hunks=[hunk]))
            
        # Handle deleted files
        for f in deleted_files:
            sn_path = os.path.join(snapshot_dir, f)
            with open(sn_path, 'r', encoding='utf-8', errors='replace') as sn_f:
                lines = sn_f.readlines()
                
            diff_text = f"@@ -1,{len(lines)} +0,0 @@\n" + "".join(f"-{line}" for line in lines)
            hunk = DiffHunk(
                hunk_id=str(uuid.uuid4()),
                file_path=f,
                old_start=1, old_lines=len(lines),
                new_start=0, new_lines=0,
                header=f"@@ -1,{len(lines)} +0,0 @@",
                diff_text=diff_text,
                status=HunkStatus.PENDING,
                explanation=""
            )
            file_diffs.append(FileDiff(file_path=f, change_type=ChangeType.DELETED, hunks=[hunk]))
            
        # Handle modified files
        for f in common_files:
            ws_path = os.path.join(workspace_root, f)
            sn_path = os.path.join(snapshot_dir, f)
            
            with open(sn_path, 'r', encoding='utf-8', errors='replace') as sn_f:
                sn_lines = sn_f.readlines()
            with open(ws_path, 'r', encoding='utf-8', errors='replace') as ws_f:
                ws_lines = ws_f.readlines()
                
            diff_gen = difflib.unified_diff(sn_lines, ws_lines, fromfile=f"a/{f}", tofile=f"b/{f}", n=3)
            diff_lines = list(diff_gen)
            
            if diff_lines:
                hunks = self._parse_hunks(diff_lines, f)
                if hunks:
                    file_diffs.append(FileDiff(file_path=f, change_type=ChangeType.MODIFIED, hunks=hunks))
                    
        return file_diffs

    def apply_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        if not self.current_token:
            print("[SnapshotDiffEngine] No active baseline token for apply_hunk")
            return False
            
        # Update snapshot to match current workspace so it won't appear as diff again
        ws_path = os.path.join(workspace_root, hunk.file_path)
        sn_path = os.path.join(self._get_snapshot_dir(workspace_root, self.current_token), hunk.file_path)
        
        try:
            if os.path.exists(ws_path):
                os.makedirs(os.path.dirname(sn_path), exist_ok=True)
                shutil.copy2(ws_path, sn_path)
            else:
                if os.path.exists(sn_path):
                    os.remove(sn_path)
            return True
        except Exception as e:
            print(f"[SnapshotDiffEngine] Error applying hunk: {e}")
            return False

    def revert_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        if not self.current_token:
            print("[SnapshotDiffEngine] No active baseline token for revert_hunk")
            return False
            
        ws_path = os.path.join(workspace_root, hunk.file_path)
        sn_path = os.path.join(self._get_snapshot_dir(workspace_root, self.current_token), hunk.file_path)
        
        try:
            # If it's a completely added file, reverting means deleting it
            if hunk.old_lines == 0 and hunk.old_start == 0:
                if os.path.exists(ws_path):
                    os.remove(ws_path)
                return True
                
            # If it's a deleted file, reverting means restoring it
            if hunk.new_lines == 0 and hunk.new_start == 0:
                if os.path.exists(sn_path):
                    os.makedirs(os.path.dirname(ws_path), exist_ok=True)
                    shutil.copy2(sn_path, ws_path)
                return True
                
            # Modified file: we need to parse diff_text and splice old lines back
            if not os.path.exists(ws_path) or not os.path.exists(sn_path):
                return False
                
            with open(ws_path, 'r', encoding='utf-8', errors='replace') as f:
                ws_lines = f.readlines()
                
            # Parse diff text to find what lines were added/removed
            diff_lines = hunk.diff_text.splitlines(keepends=True)
            old_content = []
            
            # Skip header
            start_idx = 1 if diff_lines and diff_lines[0].startswith('@@') else 0
            
            for line in diff_lines[start_idx:]:
                if line.startswith('-') or line.startswith(' '):
                    old_content.append(line[1:])
                    
            # Splice back into workspace lines
            # Adjust indices (1-based to 0-based)
            start_idx = max(0, hunk.new_start - 1)
            end_idx = start_idx + hunk.new_lines
            
            ws_lines[start_idx:end_idx] = old_content
            
            with open(ws_path, 'w', encoding='utf-8') as f:
                f.writelines(ws_lines)
                
            return True
        except Exception as e:
            print(f"[SnapshotDiffEngine] Error reverting hunk: {e}")
            return False

    def discard_all(self, workspace_root: str, baseline_token: str) -> bool:
        snapshot_dir = self._get_snapshot_dir(workspace_root, baseline_token)
        if not os.path.exists(snapshot_dir):
            print(f"[SnapshotDiffEngine] Snapshot {baseline_token} not found for discard")
            return False
            
        try:
            # Delete workspace files not in snapshot (added files)
            ws_files = self._get_workspace_files(workspace_root)
            sn_files = self._get_workspace_files(snapshot_dir)
            
            for f in (ws_files - sn_files):
                f_path = os.path.join(workspace_root, f)
                if os.path.exists(f_path):
                    os.remove(f_path)
                    
            # Copy all snapshot files back to workspace
            for f in sn_files:
                src = os.path.join(snapshot_dir, f)
                dst = os.path.join(workspace_root, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                
            return True
        except Exception as e:
            print(f"[SnapshotDiffEngine] Error discarding changes: {e}")
            return False

    def stage_all(self, workspace_root: str) -> bool:
        if self.current_token:
            snapshot_dir = self._get_snapshot_dir(workspace_root, self.current_token)
            if os.path.exists(snapshot_dir):
                try:
                    shutil.rmtree(snapshot_dir)
                    print(f"[SnapshotDiffEngine] Cleaned up snapshot {self.current_token}")
                except Exception as e:
                    print(f"[SnapshotDiffEngine] Failed to clean up snapshot: {e}")
            self.current_token = None
        return True
