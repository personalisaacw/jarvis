from abc import ABC, abstractmethod
from typing import List
from ..domain.entities import FileDiff, DiffHunk

class IDiffEnginePort(ABC):
    @abstractmethod
    def capture_baseline(self, workspace_root: str) -> str:
        """Captures initial state (commit hash or shadow snapshot token)."""
        pass

    @abstractmethod
    def compute_diff(self, workspace_root: str, baseline_token: str) -> List[FileDiff]:
        """Computes all file diffs and decomposed hunks against baseline."""
        pass

    @abstractmethod
    def apply_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        """Confirms and permanently keeps the hunk."""
        pass

    @abstractmethod
    def revert_hunk(self, workspace_root: str, hunk: DiffHunk) -> bool:
        """Reverts the hunk back to the baseline version."""
        pass

    @abstractmethod
    def discard_all(self, workspace_root: str, baseline_token: str) -> bool:
        """Discards all working tree changes."""
        pass

    @abstractmethod
    def stage_all(self, workspace_root: str) -> bool:
        """Accepts all working tree changes."""
        pass
