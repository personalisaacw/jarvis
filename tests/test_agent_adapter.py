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
