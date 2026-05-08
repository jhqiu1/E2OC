from __future__ import annotations

from typing import Any

from ...base import LLM


class OpenAIAPI(LLM):
    """OpenAI-compatible API interface.

    Args:
        base_url: API base URL.
        api_key: API key.
        model: LLM model name.
        timeout: API timeout in seconds.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout=60, **kwargs):
        super().__init__()
        import openai
        self._model = model
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, **kwargs
        )

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt.strip()}]
        response = self._client.chat.completions.create(
            model=self._model, messages=prompt, stream=False,
        )
        return response.choices[0].message.content
