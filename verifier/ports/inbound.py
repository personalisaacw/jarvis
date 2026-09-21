from abc import ABC, abstractmethod

class IReviewCommandPort(ABC):
    @abstractmethod
    def accept_hunk(self, hunk_id: str) -> None:
        pass

    @abstractmethod
    def reject_hunk(self, hunk_id: str) -> None:
        pass

    @abstractmethod
    def accept_all(self) -> None:
        pass

    @abstractmethod
    def reject_all(self) -> None:
        pass

    @abstractmethod
    def explain_hunk(self, hunk_id: str) -> None:
        pass

    @abstractmethod
    def skip_hunk(self) -> None:
        pass

class IVoiceInputPort(ABC):
    @abstractmethod
    def handle_voice_command(self, transcript: str) -> None:
        """Parses the transcript and routes it to the correct command."""
        pass
