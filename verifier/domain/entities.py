from dataclasses import dataclass, field
from typing import List, Optional
import time
from .enums import HunkStatus, ChangeType, SessionState

@dataclass
class DiffHunk:
    hunk_id: str
    file_path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    header: str
    diff_text: str             # Line-by-line diff with + / - prefixes
    status: HunkStatus = HunkStatus.PENDING
    explanation: Optional[str] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class FileDiff:
    file_path: str
    change_type: ChangeType
    hunks: List[DiffHunk] = field(default_factory=list)

@dataclass
class ReviewSession:
    session_id: str
    workspace_root: str
    baseline_token: str
    state: SessionState = SessionState.INITIALIZING
    file_diffs: List[FileDiff] = field(default_factory=list)
    current_hunk_index: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def all_hunks(self) -> List[DiffHunk]:
        return [h for f in self.file_diffs for h in f.hunks]

    @property
    def current_hunk(self) -> Optional[DiffHunk]:
        hunks = self.all_hunks
        if 0 <= self.current_hunk_index < len(hunks):
            return hunks[self.current_hunk_index]
        return None

    @property
    def pending_count(self) -> int:
        return sum(1 for h in self.all_hunks if h.status == HunkStatus.PENDING)

    @property
    def accepted_count(self) -> int:
        return sum(1 for h in self.all_hunks if h.status == HunkStatus.ACCEPTED)

    @property
    def rejected_count(self) -> int:
        return sum(1 for h in self.all_hunks if h.status == HunkStatus.REJECTED)
