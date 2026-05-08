"""Local Ollama LLM interface.

Requires: pip install ollama langchain_ollama langchain
"""

from __future__ import annotations

from typing import Any
from ...base import LLM


class LocalOllamaLLM(LLM):
    """Deploy Ollama model on local devices.

    Args:
        model_name: Name of local Ollama model checkpoint.
        ollama_llm_init_params: Initialization params for langchain_ollama.OllamaLLM.
    """

    def __init__(self, model_name: str, **ollama_llm_init_params):
        super().__init__()
        from langchain_ollama import OllamaLLM
        self.model = OllamaLLM(model=model_name, **ollama_llm_init_params)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        response = self.model.invoke(prompt)
        return response
