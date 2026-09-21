import os
import re
import json
import time
import random
from enum import Enum
from typing import Optional, List
from openai import OpenAI
from ..base import BaseParser, ParseResult

class AgentPhase(Enum):
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    READING_FILES = "reading_files"
    WRITING_CODE = "writing_code"
    TESTING = "testing"
    FINISHED = "finished"
    UNKNOWN = "unknown"

PHASE_TEMPLATES = {
    AgentPhase.INITIALIZING: ["waking up", "getting started", "booting up"],
    AgentPhase.ANALYZING: ["taking a look", "checking things out", "analyzing the request"],
    AgentPhase.PLANNING: ["making a plan", "thinking it through", "planning the steps"],
    AgentPhase.READING_FILES: ["reading some code", "scanning the repo", "checking files"],
    AgentPhase.WRITING_CODE: ["writing that out", "making the changes", "coding it up"],
    AgentPhase.TESTING: ["running tests", "checking if it works"],
}

SYSTEM_PROMPT = """You are a sub-agent summarizing terminal output for an AI voice assistant.
Your goal is to extract the current action the agent is performing and return a 2 to 4 word, highly conversational summary as if a pair-programmer is muttering to themselves.
Example outputs: "reading the files", "checking router dot py", "making the changes", "running a search", "thinking it through".
Return JSON with 'summary' (the short text) and 'phase' (one of: initializing, analyzing, planning, reading_files, writing_code, testing)."""

class AudioProgressThrottler(BaseParser):
    """
    Wraps an existing base parser (like RegexFallbackParser or a Groq-based one).
    It intercepts progress updates and uses Groq (with fallback to templates) 
    to generate natural 2-4 word bursts, heavily debounced.
    """
    def __init__(self, base_parser: BaseParser, debounce_sec: float = 4.0):
        self.base_parser = base_parser
        self.debounce_sec = debounce_sec
        self.last_spoken_time = 0.0
        self.current_phase = AgentPhase.INITIALIZING
        
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self.api_key) if self.api_key else None

    def _get_fallback_summary(self, buffer: str) -> tuple[str, AgentPhase]:
        last_lines = buffer[-500:].lower()
        if "test" in last_lines or "pytest" in last_lines:
            phase = AgentPhase.TESTING
        elif "write" in last_lines or "edit" in last_lines or "modified" in last_lines or "+++" in last_lines:
            phase = AgentPhase.WRITING_CODE
        elif "cat " in last_lines or "read" in last_lines or "grep" in last_lines:
            phase = AgentPhase.READING_FILES
        elif "plan" in last_lines or "think" in last_lines:
            phase = AgentPhase.PLANNING
        elif "analyz" in last_lines or "search" in last_lines:
            phase = AgentPhase.ANALYZING
        else:
            phase = AgentPhase.UNKNOWN
            
        if phase in PHASE_TEMPLATES:
            return random.choice(PHASE_TEMPLATES[phase]), phase
        return "", phase

    def _get_llm_summary(self, buffer: str) -> tuple[str, AgentPhase]:
        if not self.client:
            return self._get_fallback_summary(buffer)
            
        try:
            cleaned = buffer[-2000:].strip()
            response = self.client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": cleaned}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=50
            )
            data = json.loads(response.choices[0].message.content)
            summary = data.get("summary", "")
            phase_str = data.get("phase", "unknown").upper()
            try:
                phase = AgentPhase[phase_str]
            except KeyError:
                phase = AgentPhase.UNKNOWN
            
            return summary, phase
        except Exception as e:
            print(f"[Throttler] LLM summary failed: {e}")
            return self._get_fallback_summary(buffer)

    def parse(self, terminal_buffer: str, cli_name: str = "agent") -> ParseResult:
        # 1. Let the base parser handle critical things like questions and completion
        result = self.base_parser.parse(terminal_buffer, cli_name)
        
        # If it's waiting for input or completed, we ALWAYS pass it through immediately
        if result.waiting_for_input or result.is_completed:
            self.last_spoken_time = time.time() # Reset debounce
            return result
            
        # 2. It's just a progress update. Let's throttle it.
        now = time.time()
        if (now - self.last_spoken_time) < self.debounce_sec:
            result.tts_prompt = "" # Suppress audio
            return result
            
        # 3. We are cleared to speak progress. Get a dynamic summary.
        summary, phase = self._get_llm_summary(terminal_buffer)
        
        # Don't repeat the same phase within a short window unless it's been a long time
        if phase == self.current_phase and (now - self.last_spoken_time) < (self.debounce_sec * 4):
            result.tts_prompt = ""
        else:
            self.current_phase = phase
            self.last_spoken_time = now
            result.tts_prompt = summary
            
        return result
