import os
from typing import Optional

from memory.adapters.llm_json import JsonLLMExtractor

GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"  # mirrors the default in agent_adapter/parsers/groq_parser.py
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqMemoryExtractor(JsonLLMExtractor):
    def __init__(self, api_key: Optional[str] = None, model: str = GROQ_DEFAULT_MODEL, client=None):
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.client = client
        if self.client is None and self.api_key:
            try:
                from openai import OpenAI
            except ImportError:
                OpenAI = None
            if OpenAI is not None:
                self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key and self.client)

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
