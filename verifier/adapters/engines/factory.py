import os
from ...ports.diff_engine import IDiffEnginePort

class DiffEngineFactory:
    @staticmethod
    def create(workspace_root: str) -> IDiffEnginePort:
        git_dir = os.path.join(workspace_root, ".git")
        if os.path.isdir(git_dir):
            from .git_engine import GitDiffEngineAdapter
            return GitDiffEngineAdapter()
        else:
            from .snapshot_engine import SnapshotDiffEngineAdapter
            return SnapshotDiffEngineAdapter()
