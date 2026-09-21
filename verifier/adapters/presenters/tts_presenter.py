from typing import Callable, Optional
from ...ports.presenter import IReviewPresenterPort
from ...domain.entities import ReviewSession, DiffHunk


class TTSPresenter(IReviewPresenterPort):
    """
    TTS Presenter for spoken voice announcements during sequential diff review sessions.
    Provides natural conversational audio announcements via JARVIS's TTS system.
    """

    def __init__(self, on_speech: Optional[Callable[[str], None]] = None) -> None:
        """
        Initialize the TTS presenter.

        Args:
            on_speech: Callback function accepting spoken message text.
                       If None, spoken output falls back to console printing.
        """
        self.on_speech = on_speech

    def _speak(self, message: str) -> None:
        """
        Deliver spoken text via on_speech callback or fallback to standard output.

        Args:
            message: Spoken text to deliver.
        """
        if self.on_speech is not None:
            try:
                self.on_speech(message)
            except Exception as exc:
                print(f"[TTSPresenter] Speech callback error: {exc}")
                print(message)
        else:
            print(message)

    def on_session_started(self, session: ReviewSession) -> None:
        """
        Announce that the review session has started with file and block counts.
        """
        num_files = len(session.file_diffs) if session and session.file_diffs else 0
        num_hunks = len(session.all_hunks) if session else 0
        message = f"I've made changes to {num_files} files with {num_hunks} code blocks to review. Let's go through them."
        self._speak(message)

    def on_hunk_displayed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        """
        Announce the block currently displayed for review.
        """
        current_index = session.current_hunk_index if session else 0
        total = len(session.all_hunks) if session else 0
        file_path = hunk.file_path if hunk else ""
        message = f"Reviewing block {current_index + 1} of {total}. Changes in {file_path}."
        self._speak(message)

    def on_hunk_processed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        """
        Announce the status update for a processed block and remaining count.
        """
        status = hunk.status.value if hasattr(hunk.status, "value") else str(hunk.status)
        remaining = session.pending_count if session else 0
        message = f"Block {status}. {remaining} blocks remaining."
        self._speak(message)

    def on_session_completed(self, session: ReviewSession) -> None:
        """
        Announce the completion of review with accepted and rejected totals.
        """
        accepted = session.accepted_count if session else 0
        rejected = session.rejected_count if session else 0
        message = f"Review complete. {accepted} changes accepted, {rejected} rejected. Returning to standby."
        self._speak(message)

    def on_explanation_ready(self, hunk: DiffHunk, explanation: str) -> None:
        """
        Speak an explanation of the code hunk directly.
        """
        self._speak(explanation)
