from typing import List

from verifier.domain.entities import DiffHunk, FileDiff
from verifier.domain.enums import ChangeType
from verifier.use_cases.coordinator import ReviewSessionCoordinator


def _hunk(hunk_id: str) -> DiffHunk:
    return DiffHunk(
        hunk_id=hunk_id,
        file_path="a.py",
        old_start=1,
        old_lines=1,
        new_start=1,
        new_lines=1,
        header="@@ -1 +1 @@",
        diff_text="-old\n+new",
    )


class FakeDiffEngine:
    def __init__(self, diffs: List[FileDiff]):
        self.diffs = diffs

    def capture_baseline(self, workspace_root: str) -> str:
        return "tok"

    def compute_diff(self, workspace_root: str, baseline_token: str) -> List[FileDiff]:
        return self.diffs

    def apply_hunk(self, workspace_root, hunk) -> bool:
        return True

    def revert_hunk(self, workspace_root, hunk) -> bool:
        return True

    def stage_all(self, workspace_root) -> bool:
        return True

    def discard_all(self, workspace_root, baseline_token) -> bool:
        return True


class FakePresenter:
    def on_session_started(self, session) -> None:
        pass

    def on_hunk_displayed(self, session, hunk) -> None:
        pass

    def on_hunk_processed(self, session, hunk) -> None:
        pass

    def on_session_completed(self, session) -> None:
        pass


def _two_hunk_diff() -> List[FileDiff]:
    return [FileDiff(file_path="a.py", change_type=ChangeType.MODIFIED, hunks=[_hunk("h1"), _hunk("h2")])]


def test_last_completed_session_starts_none():
    coord = ReviewSessionCoordinator(FakeDiffEngine([]), FakePresenter())
    assert coord.last_completed_session is None


def test_accept_all_records_last_completed_session():
    coord = ReviewSessionCoordinator(FakeDiffEngine(_two_hunk_diff()), FakePresenter())
    coord.start_session("/w")
    coord.trigger_review()
    coord.accept_all()
    assert coord.active_session is None
    assert coord.last_completed_session is not None
    assert coord.last_completed_session.accepted_count == 2
    assert coord.last_completed_session.rejected_count == 0


def test_reject_all_records_last_completed_session():
    coord = ReviewSessionCoordinator(FakeDiffEngine(_two_hunk_diff()), FakePresenter())
    coord.start_session("/w")
    coord.trigger_review()
    coord.reject_all()
    assert coord.active_session is None
    assert coord.last_completed_session.rejected_count == 2


def test_hunk_by_hunk_completion_records_last_completed_session():
    coord = ReviewSessionCoordinator(FakeDiffEngine(_two_hunk_diff()), FakePresenter())
    coord.start_session("/w")
    coord.trigger_review()
    coord.accept_hunk("h1")
    coord.reject_hunk("h2")
    assert coord.active_session is None
    assert coord.last_completed_session.accepted_count == 1
    assert coord.last_completed_session.rejected_count == 1


def test_trigger_review_without_diffs_records_last_completed_session():
    coord = ReviewSessionCoordinator(FakeDiffEngine([]), FakePresenter())
    coord.start_session("/w")
    coord.trigger_review()
    assert coord.active_session is None
    assert coord.last_completed_session is not None
    assert coord.last_completed_session.workspace_root == "/w"
