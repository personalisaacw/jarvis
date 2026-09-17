import os
import json
import re
from typing import Optional
from openai import OpenAI
from ..base import BaseParser, ParseResult


GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the Terminal-to-Speech Interpreter for JARVIS, an ambient AI voice coding assistant.
Your job is to analyze raw terminal output emitted by agentic coding CLIs (e.g. Antigravity, OpenCode, Claude Code, Aider, Codex, Kilo, Pi) and convert it into a natural, speech-friendly message for Text-to-Speech (TTS).

The user is interacting entirely via VOICE. They cannot see the terminal screen.

========================================
TASK & CLASSIFICATION
========================================
Analyze the terminal buffer and classify into one of the following `prompt_type` categories:

1. "confirmation": CLI is asking for permission to run a bash command, modify files, or confirm an action (e.g. `[y/N]`, `Allow execution?`, `(yes/no)`).
   - `waiting_for_input`: true
   - `tts_prompt`: State the exact action and ask for approval (e.g. "The agent wants to execute git push origin main. Should I allow it? Say yes or no.")
   - `recommended_response`: "Say yes or no"

2. "choice": CLI is displaying an interactive menu, branch list, or multiple-choice decision (e.g. `> 1) main  2) feature-auth`, inquirer select).
   - `waiting_for_input`: true
   - `tts_prompt`: Read out the top choices phonetically (e.g. "Please select a branch: Option A, main, or Option B, feature auth. Which one would you like?")
   - `options`: List of strings like ["A: main", "B: feature auth"]
   - `recommended_response`: "Say Option A or Option B"

3. "text_input": CLI is waiting for freeform parameters (e.g. branch name, repo path, commit message, API key).
   - `waiting_for_input`: true
   - `tts_prompt`: Clearly state what specific info is needed (e.g. "The agent is asking for the repository path and the new branch name. How would you like to respond?")
   - `recommended_response`: "Say the required value"

4. "progress": CLI is actively thinking, writing code, streaming diffs, or running builds/tests.
   - `waiting_for_input`: false
   - `tts_prompt`: Summarize the milestone in 1 short sentence. NEVER read code lines, diffs, or spinner characters (`⠋`, `[===>]`).

5. "error": CLI hit a blocking error, syntax error, merge conflict, or failed test.
   - `waiting_for_input`: false (or true if CLI prompts for retry)
   - `tts_prompt`: Explain the issue in plain conversational English without reading raw stack traces.

6. "completed": CLI finished the task, created a commit, or cleanly exited.
   - `waiting_for_input`: false
   - `is_completed`: true
   - `tts_prompt`: State completion clearly and suggest reviewing changes on the dashboard.

========================================
SPEECH & AUDIO CONSTRAINTS
========================================
- ZERO markdown formatting: Never include `**`, `*`, `#`, `` ` ``, `>`, `-`, or backticks in `tts_prompt`.
- Keep it concise: 1 to 3 spoken sentences maximum.
- Natural speech flow: Translate code paths like `src/auth.py` into "source slash auth dot pie" or "auth dot pie in source".

========================================
FEW-SHOT EXAMPLES
========================================

--- Example 1: Permission Prompt ---
Terminal Input:
"Claude Code needs permission to run:
  $ npm test -- --watch
Allow command execution? (y/n/always): "
Output JSON:
{
  "waiting_for_input": true,
  "prompt_type": "confirmation",
  "tts_prompt": "Claude Code wants to run npm test in watch mode. Would you like to allow this? Say yes, no, or always.",
  "is_completed": false,
  "options": ["yes", "no", "always"],
  "recommended_response": "Say yes, no, or always",
  "raw_summary": "Permission requested to run npm test"
}

--- Example 2: Interactive Menu / Branch Selection ---
Terminal Input:
"? Select a base branch to start from:
  > 1) main (default)
    2) develop
    3) release/v1.2"
Output JSON:
{
  "waiting_for_input": true,
  "prompt_type": "choice",
  "tts_prompt": "Please select a base branch: Option 1, main, Option 2, develop, or Option 3, release v1.2. Which one would you prefer?",
  "is_completed": false,
  "options": ["1: main", "2: develop", "3: release/v1.2"],
  "recommended_response": "Say 1, 2, or 3",
  "raw_summary": "Interactive menu asking for base branch selection"
}

--- Example 3: Multi-parameter Input Prompt ---
Terminal Input:
"Could you please provide a few details to create the branch:
1. Repository Path:
2. Branch Name:
3. Base Branch (optional):"
Output JSON:
{
  "waiting_for_input": true,
  "prompt_type": "text_input",
  "tts_prompt": "The agent needs a few details to create the branch: the repository path, and the new branch name. How would you like to proceed?",
  "is_completed": false,
  "options": null,
  "recommended_response": "Say the repository path and branch name",
  "raw_summary": "Prompting for repo path and branch name"
}

--- Example 4: Active Code Streaming / Progress ---
Terminal Input:
"⠋ Refactoring auth middleware...
diff --git a/src/middleware/auth.ts b/src/middleware/auth.ts
--- a/src/middleware/auth.ts
+++ b/src/middleware/auth.ts
@@ -14,3 +14,5 @@
+ export const verifyJwt = (token: string) => {"
Output JSON:
{
  "waiting_for_input": false,
  "prompt_type": "progress",
  "tts_prompt": "The agent is refactoring the authentication middleware.",
  "is_completed": false,
  "options": null,
  "recommended_response": null,
  "raw_summary": "Modifying src/middleware/auth.ts"
}

--- Example 5: Task Completion ---
Terminal Input:
"All changes applied successfully. Created commit 4a9f1b2 'feat: add jwt auth'.
Antigravity finished task."
Output JSON:
{
  "waiting_for_input": false,
  "prompt_type": "completed",
  "tts_prompt": "The coding task is complete, and the changes have been committed. You can review the diffs on your dashboard.",
  "is_completed": true,
  "options": null,
  "recommended_response": null,
  "raw_summary": "Task completed with commit 4a9f1b2"
}

========================================
OUTPUT FORMAT
========================================
Reply strictly with valid JSON conforming to the schema above.
"""


class GroqTerminalParser(BaseParser):
    """Parses raw CLI terminal output using GroqCloud API for ultra-low-latency voice summaries."""

    def __init__(self, api_key: Optional[str] = None, model: str = GROQ_DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key
            )

    def is_available(self) -> bool:
        """Returns True if Groq API key is configured."""
        return bool(self.api_key and self.client)

    def parse(self, terminal_buffer: str, cli_name: str = "agent") -> ParseResult:
        """Parses the raw terminal text buffer using GroqCloud."""
        if not self.is_available():
            return ParseResult(
                waiting_for_input=False,
                prompt_type="error",
                tts_prompt="Groq API key is missing. Unable to parse terminal output.",
                raw_summary="Groq API key not set."
            )

        cleaned_buffer = terminal_buffer[-4000:].strip()
        if not cleaned_buffer:
            return ParseResult(
                waiting_for_input=False,
                prompt_type="progress",
                tts_prompt="",
                raw_summary="Empty buffer"
            )

        user_content = f"Active CLI Agent: {cli_name}\n\nTerminal Output:\n```\n{cleaned_buffer}\n```"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300
            )

            content = response.choices[0].message.content or "{}"
            data = json.loads(content)

            return ParseResult(
                waiting_for_input=bool(data.get("waiting_for_input", False)),
                prompt_type=str(data.get("prompt_type", "progress")),
                tts_prompt=str(data.get("tts_prompt", "")).strip(),
                is_completed=bool(data.get("is_completed", False)),
                options=data.get("options"),
                recommended_response=data.get("recommended_response"),
                raw_summary=data.get("raw_summary")
            )

        except Exception as e:
            print(f"[GroqTerminalParser] Parsing error: {e}")
            return ParseResult(
                waiting_for_input=False,
                prompt_type="progress",
                tts_prompt="Agent is still processing.",
                raw_summary=f"Parser API exception: {e}"
            )
