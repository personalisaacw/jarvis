import os
import sys
import time
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent_adapter.base import ParseResult
from agent_adapter.sessions.cli_session import strip_ansi, CLIAgentSession
from agent_adapter.parsers.fallback_parser import RegexFallbackParser
from agent_adapter.parsers.groq_parser import GroqTerminalParser
from agent_adapter.manager import AgentManager


def test_strip_ansi():
    raw = "\x1b[32mSuccess:\x1b[0m Created branch \x1b[1mfeature-auth\x1b[0m"
    cleaned = strip_ansi(raw)
    assert cleaned == "Success: Created branch feature-auth"


def test_fallback_parser_question():
    parser = RegexFallbackParser()
    buf = "Setting up git repository...\nCould you please provide the branch name?"
    res = parser.parse(buf)
    assert res.waiting_for_input is True
    assert "branch name" in res.tts_prompt.lower()
    assert res.is_completed is False


def test_fallback_parser_completion():
    parser = RegexFallbackParser()
    buf = "Writing file router.py...\nTask execution completed successfully."
    res = parser.parse(buf)
    assert res.is_completed is True
    assert "dashboard" in res.tts_prompt.lower()


def test_groq_parser_mocked_response():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "waiting_for_input": True,
        "prompt_type": "choice",
        "tts_prompt": "Please choose branch: A, main, or B, dev.",
        "is_completed": False,
        "options": ["A: main", "B: dev"],
        "recommended_response": "Say Option A or Option B",
        "raw_summary": "Asking for branch selection"
    })
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    parser = GroqTerminalParser(api_key="mock_key")
    parser.client = mock_client

    sample_terminal = """
    Multiple branches found:
    [1] main
    [2] dev
    Select an option:
    """

    result = parser.parse(sample_terminal, cli_name="antigravity")
    assert result.waiting_for_input is True
    assert result.prompt_type == "choice"
    assert "choose branch" in result.tts_prompt
    assert result.options == ["A: main", "B: dev"]
    assert result.recommended_response == "Say Option A or Option B"
    assert result.is_completed is False



def test_groq_parser_missing_key():
    parser = GroqTerminalParser(api_key="")
    assert parser.is_available() is False
    res = parser.parse("some buffer")
    assert "missing" in res.tts_prompt.lower()


def test_agent_manager_cli_lifecycle():
    spoken_messages = []

    def mock_speak(text: str):
        spoken_messages.append(text)

    manager = AgentManager(parser=RegexFallbackParser(), on_speech=mock_speak)

    # Register a simple echo CLI for testing
    manager.register_cli(
        "test_echo",
        lambda p, cwd, is_continuation: [sys.executable, "-c", "import sys; print('Ready for input?'); sys.stdout.flush(); line = sys.stdin.readline(); print('Finished task execution completed.'); sys.stdout.flush()"]
    )

    session = manager.start_session("test_echo", prompt="test", cwd=os.path.dirname(os.path.abspath(__file__)))
    assert session is not None
    assert manager.has_active_session() is True

    # Allow background monitor thread to detect the prompt
    time.sleep(1.5)

    # Pipe user input
    manager.send_input("my_response")

    # Wait for process to exit
    time.sleep(1.5)

    assert manager.has_active_session() is False
    assert any("finished" in msg.lower() or "completed" in msg.lower() or "input" in msg.lower() for msg in spoken_messages)


def _coding_turn_manager(last_completed_session):
    calls = []

    def spy(prompt, accepted, rejected, files, cwd):
        calls.append((prompt, accepted, rejected, files, cwd))

    manager = AgentManager(parser=RegexFallbackParser(), on_speech=None, on_coding_turn=spy)
    manager._last_prompt = "p"
    manager._last_cwd = "/w"
    manager._is_reviewing = True
    manager.review_coordinator = MagicMock(last_completed_session=last_completed_session)
    return manager, calls


def test_finish_review_reports_coding_turn():
    session = MagicMock(accepted_count=2, rejected_count=1, file_diffs=[MagicMock(file_path="a.py")])
    manager, calls = _coding_turn_manager(session)

    manager._finish_review()

    assert calls == [("p", 2, 1, ["a.py"], "/w")]
    assert manager._is_reviewing is False
    assert manager.review_coordinator is None


def test_finish_review_reports_zeros_without_completed_session():
    manager, calls = _coding_turn_manager(None)

    manager._finish_review()

    assert calls == [("p", 0, 0, [], "/w")]
    assert manager._is_reviewing is False


def test_finish_review_survives_coding_turn_failure(capsys):
    def boom(*args):
        raise RuntimeError("store down")

    manager = AgentManager(parser=RegexFallbackParser(), on_speech=None, on_coding_turn=boom)
    manager._last_prompt = "p"
    manager._last_cwd = "/w"
    manager._is_reviewing = True
    manager.review_coordinator = MagicMock(last_completed_session=None)

    manager._finish_review()

    assert manager._is_reviewing is False
    assert manager.review_coordinator is None
    assert "on_coding_turn failed: store down" in capsys.readouterr().out


def test_start_session_records_memory_prompt_not_prefixed_prompt(tmp_path):
    calls = []
    manager = AgentManager(
        parser=RegexFallbackParser(),
        on_speech=None,
        on_coding_turn=lambda *args: calls.append(args),
    )
    manager.register_cli("stub", lambda p, cwd, is_continuation: ["true"])

    with patch("agent_adapter.manager.CLIAgentSession") as session_cls, \
            patch("threading.Thread"):
        manager.start_session(
            "stub",
            prompt="## Project context\n- uses SQLite\nfix it",
            cwd=str(tmp_path),
            memory_prompt="fix it",
        )
    assert session_cls.call_args.kwargs["command"] == ["true"]
    manager.review_coordinator = MagicMock(last_completed_session=None)
    manager._finish_review()

    assert calls == [("fix it", 0, 0, [], str(tmp_path))]
