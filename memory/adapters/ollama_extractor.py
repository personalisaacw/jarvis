from memory.adapters.llm_json import JsonLLMExtractor

TIMEOUT_SECONDS = 15  # a stalled local model must not hold the capture lock / a waiting user


class OllamaMemoryExtractor(JsonLLMExtractor):
    def __init__(self, model: str = "llama3.2:3b", chat_fn=None):
        self.model = model
        self.chat_fn = chat_fn
        if self.chat_fn is None:
            try:
                import ollama

                self.chat_fn = ollama.Client(timeout=TIMEOUT_SECONDS).chat
            except ImportError:
                self.chat_fn = None

    def is_available(self) -> bool:
        return self.chat_fn is not None

    def _complete(self, system: str, user: str) -> str:
        resp = self.chat_fn(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format="json",
        )
        return resp["message"]["content"] or ""
