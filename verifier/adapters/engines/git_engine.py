import os
import subprocess
import uuid
import tempfile
import re
import datetime
from typing import List, Dict, Optional

from ...ports.diff_engine import IDiffEnginePort
from ...domain.entities import FileDiff, DiffHunk
from ...domain.enums import ChangeType, HunkStatus

BINARY_EXTENSIONS = {
    '.onnx', '.wav', '.png', '.jpg', '.jpeg', '.exe', '.dll', 
    '.so', '.zip', '.tar', '.gz', '.pyc', '.pyd'
}

class GitDiffEngineAdapter(IDiffEnginePort):
    """Adapter for performing differential operations via Git."""

    def _run(self, cmd: List[str], cwd: str, check: bool = False) -> subprocess.CompletedProcess:
        print(f"[GitDiffEngine] Running command: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=check
        )

    def capture_baseline(self, workspace_root: str) -> str:
        """Capture current HEAD as the baseline token."""
        try:
            res = self._run(["git", "rev-parse", "HEAD"], cwd=workspace_root, check=True)
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"[GitDiffEngine] Error capturing baseline: {e.stderr}")
            raise

    def compute_diff(self, workspace_root: str, baseline_token: str) -> List[FileDiff]:
        """Compute unified diff of all changes against the baseline token."""
        # 1. Get status for change types and untracked files
        status_res = self._run(["git", "status", "--porcelain"], cwd=workspace_root)
        status_map: Dict[str, ChangeType] = {}
        untracked_files: List[str] = []

        for line in status_res.stdout.splitlines():
            if len(line) < 3:
                continue
            code = line[:2]
            path = line[3:].strip()
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1] # Unquote
            
            if code == '??':
                untracked_files.append(path)
                status_map[path] = ChangeType.ADDED
            elif 'D' in code:
                status_map[path] = ChangeType.DELETED
            elif 'A' in code:
                status_map[path] = ChangeType.ADDED
            elif 'R' in code:
                status_map[path] = ChangeType.RENAMED
            else:
                status_map[path] = ChangeType.MODIFIED

        file_diffs_map: Dict[str, FileDiff] = {}

        # 2. Get staged diff
        staged_res = self._run(["git", "diff", "--cached", baseline_token, "-U3"], cwd=workspace_root)
        # 3. Get unstaged diff
        unstaged_res = self._run(["git", "diff", baseline_token, "-U3"], cwd=workspace_root)

        # 4. Merge results
        for d_text in [staged_res.stdout, unstaged_res.stdout]:
            if not d_text.strip():
                continue
            file_parts = d_text.split("diff --git ")
            for part in file_parts:
                if not part.strip():
                    continue
                fd = self._parse_file_diff(part, status_map)
                if fd:
                    if fd.file_path in file_diffs_map:
                        # Merge hunks avoiding complete duplicates
                        existing_hunks = {h.header for h in file_diffs_map[fd.file_path].hunks}
                        for new_hunk in fd.hunks:
                            if new_hunk.header not in existing_hunks:
                                file_diffs_map[fd.file_path].hunks.append(new_hunk)
                                existing_hunks.add(new_hunk.header)
                    else:
                        file_diffs_map[fd.file_path] = fd

        # 5. Process untracked files
        for uf in untracked_files:
            fd = self._create_untracked_diff(workspace_root, uf)
            if fd:
                if fd.file_path not in file_diffs_map:
                    file_diffs_map[fd.file_path] = fd
                else:
                    file_diffs_map[fd.file_path].hunks.extend(fd.hunks)

        return list(file_diffs_map.values())

    def _is_binary(self, filename: str, diff_text: str) -> bool:
        if "Binary files" in diff_text and "differ" in diff_text:
            return True
        ext = os.path.splitext(filename)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return True
        return False

    def _parse_file_diff(self, diff_text: str, status_map: Dict[str, ChangeType]) -> Optional[FileDiff]:
        lines = diff_text.splitlines()
        if not lines:
            return None

        header_line = lines[0]
        parts = header_line.split(" b/")
        if len(parts) != 2:
            return None
        
        # Extract actual file path
        file_path = parts[1].strip()
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        if self._is_binary(file_path, diff_text):
            print(f"[GitDiffEngine] Skipping binary file: {file_path}")
            return None

        change_type = status_map.get(file_path, ChangeType.MODIFIED)
        hunks = []
        
        # Find all hunk boundaries
        hunk_starts = [i for i, line in enumerate(lines) if line.startswith("@@ ")]
        
        for i, start_idx in enumerate(hunk_starts):
            end_idx = hunk_starts[i+1] if i + 1 < len(hunk_starts) else len(lines)
            hunk_lines = lines[start_idx:end_idx]
            hunk = self._parse_hunk(file_path, hunk_lines)
            if hunk:
                hunks.append(hunk)

        return FileDiff(
            file_path=file_path,
            change_type=change_type,
            hunks=hunks
        )

    def _parse_hunk(self, file_path: str, lines: List[str]) -> Optional[DiffHunk]:
        if not lines:
            return None
        
        header = lines[0]
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
        if not m:
            return None

        old_start = int(m.group(1))
        old_lines = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_lines = int(m.group(4)) if m.group(4) is not None else 1

        return DiffHunk(
            hunk_id=str(uuid.uuid4()),
            file_path=file_path,
            old_start=old_start,
            old_lines=old_lines,
            new_start=new_start,
            new_lines=new_lines,
            header=header,
            diff_text="\n".join(lines),
            status=HunkStatus.PENDING,
            explanation="",
            created_at=datetime.datetime.now()
        )

    def _create_untracked_diff(self, workspace_root: str, file_path: str) -> Optional[FileDiff]:
        full_path = os.path.join(workspace_root, file_path)
        if self._is_binary(file_path, ""):
            print(f"[GitDiffEngine] Skipping binary file: {file_path}")
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[GitDiffEngine] Skipping unreadable file {file_path}: {e}")
            return None

        lines = content.splitlines()
        new_lines_count = len(lines)
        
        header = f"@@ -0,0 +1,{new_lines_count} @@"
        diff_text_lines = [header] + [f"+{line}" for line in lines]
        diff_text = "\n".join(diff_text_lines)

        hunk = DiffHunk(
            hunk_id=str(uuid.uuid4()),
            file_path=file_path,
            old_start=0,
            old_lines=0,
            new_start=1,
            new_lines=new_lines_count,
            header=header,
            diff_text=diff_text,
            status=HunkStatus.PENDING,
            explanation="",
            created_at=datetime.datetime.now()
        )

        return FileDiff(
            file_path=file_path,
            change_type=ChangeType.ADDED,
            hunks=[hunk]
        )

    def apply_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        """Apply a hunk to the index (stage it)."""
        patch_content = f"--- a/{hunk.file_path}\n+++ b/{hunk.file_path}\n{hunk.diff_text}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding='utf-8') as f:
            f.write(patch_content)
            temp_path = f.name

        try:
            res = self._run(["git", "apply", "--cached", temp_path], cwd=workspace_root)
            if res.returncode != 0:
                print(f"[GitDiffEngine] Failed to apply hunk: {res.stderr}")
                return False
            return True
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def revert_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        """Reverse a hunk from the working directory."""
        
        # FIX: If this is an entirely new untracked file, reverting the hunk just means deleting the file.
        if hunk.old_start == 0 and hunk.old_lines == 0:
            full_path = os.path.join(workspace_root, hunk.file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                print(f"[GitDiffEngine] Reverted new file creation by deleting {hunk.file_path}")
            return True

        patch_content = f"--- a/{hunk.file_path}\n+++ b/{hunk.file_path}\n{hunk.diff_text}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding='utf-8') as f:
            f.write(patch_content)
            temp_path = f.name

        try:
            res = self._run(["git", "apply", "-R", "--whitespace=nowarn", temp_path], cwd=workspace_root)
            if res.returncode != 0:
                print(f"[GitDiffEngine] Failed to revert hunk: {res.stderr}")
                return False
            return True
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def discard_all(self, workspace_root: str, baseline_token: str) -> bool:
        """Discard all uncommitted changes in the workspace."""
        try:
            self._run(["git", "checkout", "--", "."], cwd=workspace_root, check=True)
            self._run(["git", "clean", "-fd"], cwd=workspace_root, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[GitDiffEngine] Failed to discard changes: {e.stderr}")
            return False

    def stage_all(self, workspace_root: str) -> bool:
        """Stage all changes in the workspace."""
        try:
            self._run(["git", "add", "-A"], cwd=workspace_root, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[GitDiffEngine] Failed to stage changes: {e.stderr}")
            return False
