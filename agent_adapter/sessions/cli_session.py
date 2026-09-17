import os
import re
import time
import queue
import threading
import subprocess
from typing import Optional, Callable, List
from ..base import BaseAgentSession, BaseParser, ParseResult


ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def strip_ansi(text: str) -> str:
    """Removes ANSI color codes and terminal formatting sequences."""
    return ANSI_ESCAPE_RE.sub('', text)


class CLIAgentSession(BaseAgentSession):
    """
    Manages an interactive CLI subprocess with background stdout reading,
    ANSI stripping, pause detection, and LLM-powered Terminal-to-Speech parsing.
    """

    def __init__(
        self,
        command: List[str],
        cwd: str,
        cli_name: str,
        parser: BaseParser,
        on_speech_ready: Optional[Callable[[str], None]] = None,
        on_session_finished: Optional[Callable[[], None]] = None,
        pause_threshold_sec: float = 1.2
    ):
        self.command = command
        self.cwd = cwd
        self.cli_name = cli_name
        self.parser = parser
        self.on_speech_ready = on_speech_ready
        self.on_session_finished = on_session_finished
        self.pause_threshold_sec = pause_threshold_sec

        self.process: Optional[subprocess.Popen] = None
        self._buffer: str = ""
        self._last_parsed_len: int = 0
        self._last_output_time: float = 0
        self._is_running: bool = False
        self._is_waiting_for_input: bool = False
        self._lock = threading.Lock()

        self._read_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Launches the CLI subprocess and starts background monitoring."""
        print(f"[{self.cli_name}] Spawning process: {' '.join(self.command)}")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                text=False
            )
            self._is_running = True
            self._last_output_time = time.time()

            self._read_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
            self._read_thread.start()

            self._monitor_thread = threading.Thread(target=self._monitor_output_loop, daemon=True)
            self._monitor_thread.start()

        except Exception as e:
            print(f"[{self.cli_name}] Failed to start process: {e}")
            self._is_running = False
            if self.on_speech_ready:
                self.on_speech_ready(f"Error starting {self.cli_name}: {e}")

    def send_input(self, text: str) -> None:
        """Pipes user speech input directly into the CLI's standard input."""
        if not self.is_active() or not self.process or not self.process.stdin:
            print(f"[{self.cli_name}] Cannot send input, process is not active.")
            return

        print(f"[{self.cli_name}] Sending user input to stdin: '{text}'")
        try:
            payload = (text.strip() + "\n").encode("utf-8")
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            with self._lock:
                self._is_waiting_for_input = False
                self._last_output_time = time.time()
        except Exception as e:
            print(f"[{self.cli_name}] Error writing to stdin: {e}")

    def stop(self) -> None:
        """Terminates the process and cleans up threads."""
        self._stop_event.set()
        self._is_running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def is_active(self) -> bool:
        """Returns True if the process is currently running."""
        if not self._is_running or not self.process:
            return False
        return self.process.poll() is None

    def is_waiting_for_input(self) -> bool:
        """Returns True if the CLI is blocked waiting for human input."""
        return self._is_waiting_for_input

    def get_raw_buffer(self) -> str:
        """Returns the accumulated stripped buffer."""
        with self._lock:
            return self._buffer

    def _read_stdout_loop(self) -> None:
        """Continuously reads bytes from the process stdout."""
        while self.is_active() and not self._stop_event.is_set():
            try:
                chunk = self.process.stdout.read(1024)
                if not chunk:
                    break

                text = chunk.decode("utf-8", errors="replace")
                # Print to terminal for live screen visual if desired
                print(text, end="", flush=True)

                cleaned = strip_ansi(text)
                with self._lock:
                    self._buffer += cleaned
                    self._last_output_time = time.time()
            except Exception as e:
                print(f"[{self.cli_name}] Stdout read error: {e}")
                break

        self._is_running = False

    def _monitor_output_loop(self) -> None:
        """Monitors stdout activity; triggers parser when output pauses or finishes."""
        last_checked_len = 0

        while not self._stop_event.is_set():
            time.sleep(0.3)
            now = time.time()

            with self._lock:
                current_len = len(self._buffer)
                time_since_output = now - self._last_output_time
                has_new_output = current_len > last_checked_len
                is_proc_dead = not self.is_active()

            # Trigger parsing if output paused for threshold or process exited
            if (has_new_output and time_since_output >= self.pause_threshold_sec) or (is_proc_dead and has_new_output):
                with self._lock:
                    snapshot = self._buffer
                    last_checked_len = current_len

                parse_result = self.parser.parse(snapshot, cli_name=self.cli_name)
                print(f"\n[{self.cli_name} Parser] Waiting: {parse_result.waiting_for_input} | TTS: '{parse_result.tts_prompt}'")

                with self._lock:
                    self._is_waiting_for_input = parse_result.waiting_for_input

                if parse_result.tts_prompt and self.on_speech_ready:
                    self.on_speech_ready(parse_result.tts_prompt)

                if parse_result.is_completed:
                    break

            if is_proc_dead:
                break

        # Process exited
        print(f"[{self.cli_name}] Process terminated.")
        self._is_running = False
        if self.on_session_finished:
            self.on_session_finished()
