import subprocess
from typing import List
from ...ports.diff_engine import IDiffEnginePort
from ...domain.entities import FileDiff, DiffHunk

class GitDiffEngineAdapter(IDiffEnginePort):
    def capture_baseline(self, workspace_root: str) -> str:
        # returns branch name or HEAD hash
        return "HEAD"

    def compute_diff(self, workspace_root: str, baseline_token: str) -> List[FileDiff]:
        # Implementation would use subprocess to call `git diff -U3` and parse output
        return []

    def apply_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        # Keep hunk (it's already in working tree). Optionally stage it.
        return True

    def revert_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        # Use `git apply -R` or `patch -R`
        return True

    def discard_all(self, workspace_root: str, baseline_token: str) -> bool:
        subprocess.run(["git", "checkout", "--", "."], cwd=workspace_root)
        return True

    def stage_all(self, workspace_root: str) -> bool:
        subprocess.run(["git", "add", "-A"], cwd=workspace_root)
        return True
