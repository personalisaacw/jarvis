import re
from typing import Optional
from ..base import BaseParser, ParseResult


class RegexFallbackParser(BaseParser):
    """Rule-based fallback parser used when cloud LLM inference is unavailable."""

    QUESTION_PATTERNS = [
        r"\?\s*$",
        r"\[y/n\]",
        r"\(y/n\)",
        r"press enter to continue",
        r"select an option",
        r"could you please provide",
        r"what would you like to name",
        r"enter your choice",
        r"branch name\??",
        r"repository path\??",
        r"please let me know:",
        r"\b1\. "
    ]

    COMPLETION_PATTERNS = [
        r"task execution completed",
        r"successfully completed",
        r"changes committed",
        r"all tasks done"
    ]

    def parse(self, terminal_buffer: str, cli_name: str = "agent") -> ParseResult:
        cleaned = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", terminal_buffer).strip()
        last_lines = "\n".join(cleaned.splitlines()[-6:]).strip()

        # Check completion
        for pat in self.COMPLETION_PATTERNS:
            if re.search(pat, last_lines, re.IGNORECASE):
                return ParseResult(
                    waiting_for_input=False,
                    tts_prompt="The coding task has completed. You can review the changes on your dashboard.",
                    is_completed=True,
                    raw_summary="Matched completion pattern"
                )

        # Check question/input required
        for pat in self.QUESTION_PATTERNS:
            if re.search(pat, last_lines, re.IGNORECASE):
                # Clean up prompt line for speech
                clean_speech = re.sub(r"[*#_`~>]", "", last_lines)
                clean_speech = re.sub(r"\s+", " ", clean_speech).strip()
                return ParseResult(
                    waiting_for_input=True,
                    tts_prompt=f"The agent requires your input: {clean_speech}",
                    is_completed=False,
                    raw_summary=f"Matched question pattern '{pat}'"
                )

        return ParseResult(
            waiting_for_input=False,
            tts_prompt="Agent is working on the task.",
            is_completed=False,
            raw_summary="Default processing state"
        )
