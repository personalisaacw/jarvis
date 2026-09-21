from abc import ABC, abstractmethod
from typing import Optional, List
from pydantic import BaseModel, Field


class ParseResult(BaseModel):
    """Structured representation of interpreted CLI output."""
    waiting_for_input: bool = Field(
        default=False, 
        description="Whether the CLI is blocked waiting for human input/decision."
    )
    prompt_type: str = Field(
        default="progress",
        description="Classification: 'confirmation', 'choice', 'text_input', 'progress', 'completed', 'error'"
    )
    tts_prompt: str = Field(
        default="", 
        description="Speech-friendly message to be voiced via TTS (intent, details, questions, options)."
    )
    is_completed: bool = Field(
        default=False, 
        description="Whether the CLI process has completed its task."
    )
    options: Optional[List[str]] = Field(
        default=None, 
        description="List of specific options if the CLI asked a multiple-choice question (e.g. ['A: main', 'B: dev'])."
    )
    recommended_response: Optional[str] = Field(
        default=None,
        description="Instruction on how the user should reply (e.g. 'Say yes or no', 'Say A or B')."
    )
    raw_summary: Optional[str] = Field(
        default=None, 
        description="Short debug summary of the raw terminal chunk."
    )



class BaseParser(ABC):
    """Interface for interpreting CLI terminal output into speech-friendly ParseResult."""

    @abstractmethod
    def parse(self, terminal_buffer: str, cli_name: str = "agent") -> ParseResult:
        """Parses the raw terminal text buffer and returns a structured ParseResult."""
        pass


class BaseAgentSession(ABC):
    """Interface for managing the lifecycle of an agentic CLI session."""

    @abstractmethod
    def start(self) -> None:
        """Starts the underlying agent process."""
        pass

    @abstractmethod
    def send_input(self, text: str) -> None:
        """Sends user input (from speech STT) to the active agent process."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Terminates the active agent process."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Returns True if the agent process is still running."""
        pass
