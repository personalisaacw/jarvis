from typing import Optional
import uuid
from ..domain.entities import ReviewSession, SessionState, DiffHunk, HunkStatus
from ..domain.enums import InputModality, DeviceType
from ..ports.diff_engine import IDiffEnginePort
from ..ports.presenter import IReviewPresenterPort
from ..ports.inbound import IReviewCommandPort

class ReviewSessionCoordinator(IReviewCommandPort):
    def __init__(self, diff_engine: IDiffEnginePort, presenter: IReviewPresenterPort):
        self.diff_engine = diff_engine
        self.presenter = presenter
        self.active_session: Optional[ReviewSession] = None

    def start_session(self, workspace_root: str) -> None:
        baseline_token = self.diff_engine.capture_baseline(workspace_root)
        self.active_session = ReviewSession(
            session_id=str(uuid.uuid4()),
            workspace_root=workspace_root,
            baseline_token=baseline_token,
            state=SessionState.AGENT_RUNNING
        )
        # We don't have diffs yet. Agent runs.

    def trigger_review(self) -> None:
        if not self.active_session:
            return
        
        file_diffs = self.diff_engine.compute_diff(
            self.active_session.workspace_root, 
            self.active_session.baseline_token
        )
        
        if not file_diffs:
            self.active_session.state = SessionState.COMPLETED
            self.presenter.on_session_completed(self.active_session)
            self.active_session = None
            return

        self.active_session.file_diffs = file_diffs
        self.active_session.state = SessionState.REVIEW_ACTIVE
        self.active_session.current_hunk_index = 0
        
        self.presenter.on_session_started(self.active_session)
        
        hunk = self.active_session.current_hunk
        if hunk:
            self.presenter.on_hunk_displayed(self.active_session, hunk)

    def _get_hunk_by_id(self, hunk_id: str) -> Optional[DiffHunk]:
        if not self.active_session:
            return None
        for h in self.active_session.all_hunks:
            if h.hunk_id == hunk_id:
                return h
        return None

    def accept_hunk(self, hunk_id: str) -> None:
        if not self.active_session: return
        hunk = self._get_hunk_by_id(hunk_id)
        if not hunk or hunk.status != HunkStatus.PENDING: return

        # Apply using engine
        success = self.diff_engine.apply_hunk(self.active_session.workspace_root, hunk)
        if success:
            hunk.status = HunkStatus.ACCEPTED
            self._advance_and_notify(hunk)

    def reject_hunk(self, hunk_id: str) -> None:
        if not self.active_session: return
        hunk = self._get_hunk_by_id(hunk_id)
        if not hunk or hunk.status != HunkStatus.PENDING: return

        success = self.diff_engine.revert_hunk(self.active_session.workspace_root, hunk)
        if success:
            hunk.status = HunkStatus.REJECTED
            self._advance_and_notify(hunk)

    def _advance_and_notify(self, hunk: DiffHunk):
        self.presenter.on_hunk_processed(self.active_session, hunk)
        
        # Advance index to next pending hunk
        self._advance_pointer()

        next_hunk = self.active_session.current_hunk
        if next_hunk:
            self.presenter.on_hunk_displayed(self.active_session, next_hunk)
        else:
            self.active_session.state = SessionState.COMPLETED
            self.presenter.on_session_completed(self.active_session)
            self.active_session = None

    def _advance_pointer(self):
        hunks = self.active_session.all_hunks
        start_idx = self.active_session.current_hunk_index
        for i in range(start_idx + 1, len(hunks)):
            if hunks[i].status == HunkStatus.PENDING:
                self.active_session.current_hunk_index = i
                return
        # If no pending hunks remain, set out of bounds
        self.active_session.current_hunk_index = len(hunks)

    def accept_all(self) -> None:
        if not self.active_session: return
        success = self.diff_engine.stage_all(self.active_session.workspace_root)
        if success:
            for h in self.active_session.all_hunks:
                if h.status == HunkStatus.PENDING:
                    h.status = HunkStatus.ACCEPTED
            self.active_session.state = SessionState.COMPLETED
            self.presenter.on_session_completed(self.active_session)
            self.active_session = None

    def reject_all(self) -> None:
        if not self.active_session: return
        success = self.diff_engine.discard_all(self.active_session.workspace_root, self.active_session.baseline_token)
        if success:
            for h in self.active_session.all_hunks:
                if h.status == HunkStatus.PENDING:
                    h.status = HunkStatus.REJECTED
            self.active_session.state = SessionState.COMPLETED
            self.presenter.on_session_completed(self.active_session)
            self.active_session = None

    def explain_hunk(self, hunk_id: str) -> None:
        pass

    def skip_hunk(self) -> None:
        pass
