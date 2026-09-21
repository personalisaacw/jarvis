from abc import ABC, abstractmethod
from ..domain.entities import ReviewSession, DiffHunk

class IReviewPresenterPort(ABC):
    @abstractmethod
    def on_session_started(self, session: ReviewSession) -> None:
        pass

    @abstractmethod
    def on_hunk_displayed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        pass

    @abstractmethod
    def on_hunk_processed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        pass

    @abstractmethod
    def on_session_completed(self, session: ReviewSession) -> None:
        pass

    @abstractmethod
    def on_explanation_ready(self, hunk: DiffHunk, explanation: str) -> None:
        pass
