import os
import shutil
from typing import Dict, Callable, List, Optional
from .base import BaseParser, BaseAgentSession
from .parsers.groq_parser import GroqTerminalParser
from .parsers.fallback_parser import RegexFallbackParser
from .sessions.cli_session import CLIAgentSession

from verifier.use_cases.coordinator import ReviewSessionCoordinator
from verifier.adapters.engines.factory import DiffEngineFactory
from verifier.adapters.presenters.composite import MultiPresenterComposite
from verifier.adapters.presenters.tts_presenter import TTSPresenter
from verifier.adapters.presenters.desktop_hud import DesktopHUDPresenter
from verifier.adapters.presenters.mobile_ws_presenter import MobileWSPresenter
from verifier.adapters.input.voice_grammar import ReviewVoiceGrammar


def default_antigravity_cmd(prompt: str, cwd: str, is_continuation: bool = False) -> List[str]:
    agy_bin = shutil.which("agy") or os.path.expandvars(r"%LOCALAPPDATA%\agy\bin\agy.exe")
    cmd = [
        agy_bin,
        "-p", prompt,
        "--model", "gemini-3.8-flash-medium",
        "--dangerously-skip-permissions",
        "--add-dir", cwd
    ]
    if is_continuation:
        cmd.append("-c")
    return cmd


def default_opencode_cmd(prompt: str, cwd: str, is_continuation: bool = False) -> List[str]:
    opencode_bin = shutil.which("opencode") or "opencode"
    return [opencode_bin, prompt]


from .parsers.summarizer import AudioProgressThrottler

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
            base_parser = parser
        else:
            groq = GroqTerminalParser()
            base_parser = groq if groq.is_available() else RegexFallbackParser()
            
        self.parser = AudioProgressThrottler(base_parser)

        self.on_speech = on_speech
        self.active_session: Optional[CLIAgentSession] = None
        self._cli_registry: Dict[str, Callable[[str, str, bool], List[str]]] = {}
        self._is_reviewing = False

        # ── Verifier subsystem setup ──
        self.review_coordinator: Optional[ReviewSessionCoordinator] = None
        self.mobile_presenter = MobileWSPresenter()
        self.voice_grammar: Optional[ReviewVoiceGrammar] = None

        # Register default CLIs
        self.register_cli("antigravity", default_antigravity_cmd)
        self.register_cli("opencode", default_opencode_cmd)

    def register_cli(self, name: str, cmd_factory: Callable[[str, str, bool], List[str]]) -> None:
        """Registers a CLI command builder by name."""
        self._cli_registry[name.lower()] = cmd_factory

    def has_active_session(self) -> bool:
        """Returns True if there is an active running agent session."""
        return self.active_session is not None and self.active_session.is_active()

    def start_session(
        self,
        cli_name: str,
        prompt: str,
        cwd: Optional[str] = None,
        is_continuation: bool = False
    ) -> Optional[CLIAgentSession]:
        """Launches a new agent session for the specified CLI name and initial prompt."""
        if self.has_active_session() and not is_continuation:
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
        cmd = self._cli_registry[cli_key](prompt, working_dir, is_continuation)

        # ── Capture baseline for diff verification asynchronously ──
        if not is_continuation:
            def capture_baseline():
                try:
                    diff_engine = DiffEngineFactory.create(working_dir)
                    tts_presenter = TTSPresenter(on_speech=self.on_speech)
                    hud_presenter = DesktopHUDPresenter(on_command=self._on_hud_command)
                    composite = MultiPresenterComposite([tts_presenter, hud_presenter, self.mobile_presenter])
                    self.review_coordinator = ReviewSessionCoordinator(diff_engine, composite)
                    self.voice_grammar = ReviewVoiceGrammar(coordinator=self.review_coordinator)
                    self.review_coordinator.start_session(working_dir)
                    print(f"[AgentManager] Baseline captured for diff verification in {working_dir}")
                except Exception as e:
                    print(f"[AgentManager] Warning: Could not capture baseline: {e}")
                    self.review_coordinator = None
            
            import threading
            threading.Thread(target=capture_baseline, daemon=True).start()

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

    @property
    def is_reviewing(self) -> bool:
        """Returns True if a diff review session is currently active."""
        return self._is_reviewing

    def send_input(self, text: str) -> None:
        """Routes human voice input to the active CLI session or review coordinator."""

        # ── If in diff review mode, route voice to the review grammar ──
        if self._is_reviewing and self.voice_grammar:
            print(f"[AgentManager] Routing voice to review grammar: '{text}'")
            self.voice_grammar.handle_voice_command(text)
            return

        if not self.active_session:
            print("[AgentManager] No active session to receive input.")
            return

        if not self.active_session.is_active() and self.active_session.is_waiting_for_input():
            # Stateless continuation: process exited but waiting for input
            print(f"[AgentManager] Statelessly continuing {self.active_session.cli_name}...")
            self.start_session(
                self.active_session.cli_name,
                text,
                cwd=self.active_session.cwd,
                is_continuation=True
            )
        else:
            self.active_session.send_input(text)

    def stop_active_session(self) -> None:
        """Terminates the active session if one is running."""
        if self.active_session:
            self.active_session.stop()
            self.active_session = None

    def _on_hud_command(self, action: str, hunk_id: Optional[str]) -> None:
        """Callback from the Desktop HUD for keyboard/mouse actions."""
        if not self.review_coordinator:
            return
        if action == "accept" and hunk_id:
            self.review_coordinator.accept_hunk(hunk_id)
        elif action == "reject" and hunk_id:
            self.review_coordinator.reject_hunk(hunk_id)
        elif action == "accept_all":
            self.review_coordinator.accept_all()
            self._finish_review()
        elif action == "reject_all":
            self.review_coordinator.reject_all()
            self._finish_review()
        elif action == "explain" and hunk_id:
            self.review_coordinator.explain_hunk(hunk_id)
        elif action == "skip":
            self.review_coordinator.skip_hunk()

        # Check if review completed after hunk-level actions
        if self.review_coordinator and not self.review_coordinator.active_session:
            self._finish_review()

    def _on_session_finished(self, was_waiting_for_input: bool = False) -> None:
        """Internal callback invoked when a CLI agent session exits."""
        if was_waiting_for_input:
            # We purposefully do not reset the active_session or announce finish
            return

        self.active_session = None

        # ── Trigger diff verification ──
        if self.review_coordinator:
            print("[AgentManager] Agent session ended. Starting diff verification...")
            self._is_reviewing = True
            self.review_coordinator.trigger_review()

            # If no changes were found, trigger_review already completed the session
            if not self.review_coordinator.active_session:
                self._finish_review()
        else:
            print("[AgentManager] Session ended. Restoring ambient voice mode.")
            if self.on_speech:
                self.on_speech("Coding task has finished. Returning to voice assistant mode.")

    def _finish_review(self) -> None:
        """Clean up review state and return to ambient mode."""
        self._is_reviewing = False
        self.review_coordinator = None
        self.voice_grammar = None
        print("[AgentManager] Diff review complete. Returning to ambient voice mode.")
