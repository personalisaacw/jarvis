from typing import List
from ...ports.presenter import IReviewPresenterPort
from ...domain.entities import ReviewSession, DiffHunk

class MultiPresenterComposite(IReviewPresenterPort):
    def __init__(self, presenters: List[IReviewPresenterPort]):
        self.presenters = presenters

    def on_session_started(self, session: ReviewSession) -> None:
        for p in self.presenters:
            p.on_session_started(session)

    def on_hunk_displayed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        for p in self.presenters:
            p.on_hunk_displayed(session, hunk)

    def on_hunk_processed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        for p in self.presenters:
            p.on_hunk_processed(session, hunk)

    def on_session_completed(self, session: ReviewSession) -> None:
        for p in self.presenters:
            p.on_session_completed(session)

    def on_explanation_ready(self, hunk: DiffHunk, explanation: str) -> None:
        for p in self.presenters:
            p.on_explanation_ready(hunk, explanation)
