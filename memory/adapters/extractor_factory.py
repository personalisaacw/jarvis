from memory.adapters.groq_extractor import GroqMemoryExtractor
from memory.adapters.ollama_extractor import OllamaMemoryExtractor
from memory.ports.extractor import IMemoryExtractor


def build_extractor() -> IMemoryExtractor:
    """Groq when an API key is configured, otherwise the local Ollama fallback."""
    groq = GroqMemoryExtractor()
    if groq.is_available():
        return groq
    return OllamaMemoryExtractor()
