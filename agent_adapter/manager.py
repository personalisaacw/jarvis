import os
import shutil
from typing import Dict, Callable, List, Optional
from .base import BaseParser, BaseAgentSession
from .parsers.groq_parser import GroqTerminalParser
from .parsers.fallback_parser import RegexFallbackParser
from .sessions.cli_session import CLIAgentSession


def default_antigravity_cmd(prompt: str, cwd: str) -> List[str]:
    agy_bin = shutil.which("agy") or os.path.expandvars(r"%LOCALAPPDATA%\agy\bin\agy.exe")
    return [
        agy_bin,
        "-i", prompt,
        "--model", "gemini-3.8-flash-medium",
        "--dangerously-skip-permissions",
        "--add-dir", cwd
    ]


def default_opencode_cmd(prompt: str, cwd: str) -> List[str]:
    opencode_bin = shutil.which("opencode") or "opencode"
    return [opencode_bin, prompt]


class AgentManager:
    """
    Central registry and coordinator for agentic CLIs.
    Routes user speech to active agent sessions and manages Terminal-to-Speech translation.
    """

    def __init__(
        self,
        parser: Optional[BaseParser] = None,
        on_speech: Optional[Callable[[str], None]] = None
    ):
        # Auto-configure parser: Groq if key is present, otherwise fallback
        if parser:
            self.parser = parser
        else:
            groq = GroqTerminalParser()
            self.parser = groq if groq.is_available() else RegexFallbackParser()

        self.on_speech = on_speech
        self.active_session: Optional[CLIAgentSession] = None
        self._cli_registry: Dict[str, Callable[[str, str], List[str]]] = {}

        # Register default CLIs
        self.register_cli("antigravity", default_antigravity_cmd)
        self.register_cli("opencode", default_opencode_cmd)

    def register_cli(self, name: str, cmd_factory: Callable[[str, str], List[str]]) -> None:
        """Registers a CLI command builder by name."""
        self._cli_registry[name.lower()] = cmd_factory

    def has_active_session(self) -> bool:
        """Returns True if there is an active running agent session."""
        return self.active_session is not None and self.active_session.is_active()

    def start_session(
        self,
        cli_name: str,
        prompt: str,
        cwd: Optional[str] = None
    ) -> Optional[CLIAgentSession]:
        """Launches a new agent session for the specified CLI name and initial prompt."""
        if self.has_active_session():
            print(f"[AgentManager] Active session already running for {self.active_session.cli_name}. Stopping prior session.")
            self.stop_active_session()

        cli_key = cli_name.lower()
        if cli_key not in self._cli_registry:
            err = f"Unknown CLI agent '{cli_name}'. Available: {list(self._cli_registry.keys())}"
            print(f"[AgentManager] {err}")
            if self.on_speech:
                self.on_speech(f"Error: Agent {cli_name} is not registered.")
            return None

        working_dir = cwd or os.path.dirname(os.path.abspath(__file__))
        cmd = self._cli_registry[cli_key](prompt, working_dir)

        session = CLIAgentSession(
            command=cmd,
            cwd=working_dir,
            cli_name=cli_name,
            parser=self.parser,
            on_speech_ready=self.on_speech,
            on_session_finished=self._on_session_finished
        )
        self.active_session = session
        session.start()
        return session

    def send_input(self, text: str) -> None:
        """Routes human voice input to the currently active CLI session."""
        if not self.has_active_session():
            print("[AgentManager] No active session to receive input.")
            return
        self.active_session.send_input(text)

    def stop_active_session(self) -> None:
        """Terminates the active session if one is running."""
        if self.active_session:
            self.active_session.stop()
            self.active_session = None

    def _on_session_finished(self) -> None:
        """Internal callback invoked when a session exits."""
        print("[AgentManager] Session ended. Restoring ambient voice mode.")
        if self.on_speech:
            self.on_speech("Coding task has finished. Returning to voice assistant mode.")
        self.active_session = None
