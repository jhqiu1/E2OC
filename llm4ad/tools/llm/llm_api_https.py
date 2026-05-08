from __future__ import annotations

import http.client
import json
import time
from typing import Any
import traceback
from ...base import LLM


class HttpsApi(LLM):
    """HTTPS API interface for LLM services.

    Args:
        host: Host name (without 'https://').
        key: API key.
        model: LLM model name.
        timeout: API timeout in seconds.
    """

    def __init__(self, host, key, model, timeout=30, **kwargs):
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0
        self.debug_mode = True

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt.strip()}]

        while True:
            try:
                conn = http.client.HTTPSConnection(
                    self._host, timeout=self._timeout
                )
                payload = json.dumps(
                    {
                        "max_tokens": self._kwargs.get("max_tokens", 4096),
                        "top_p": self._kwargs.get("top_p", None),
                        "temperature": self._kwargs.get("temperature", 1.0),
                        "model": self._model,
                        "messages": prompt,
                    }
                )
                headers = {
                    "Authorization": f"Bearer {self._key}",
                    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                    "Content-Type": "application/json",
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                if res.status != 200:
                    body = res.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"HTTP {res.status}: {body[:500]}"
                    )
                data = res.read().decode("utf-8")
                data = json.loads(data)
                if "error" in data:
                    err = data["error"]
                    raise RuntimeError(
                        f"API error: {err.get('message', str(err))}"
                        f" (type: {err.get('type', 'unknown')})"
                    )
                response = data["choices"][0]["message"]["content"]
                # Handle DeepSeek reasoning mode: content might be empty, use reasoning_content
                if not response:
                    message = data["choices"][0].get("message", {})
                    reasoning_content = message.get("reasoning_content", "")
                    if reasoning_content:
                        response = reasoning_content
                if self.debug_mode:
                    self._cumulative_error = 0
                return response
            except Exception as e:
                self._cumulative_error += 1
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(
                            f"{self.__class__.__name__} error: {traceback.format_exc()}."
                            f"You may check your API host and API key."
                        )
                else:
                    print(
                        f"{self.__class__.__name__} error: {traceback.format_exc()}."
                        f"You may check your API host and API key."
                    )
                    time.sleep(2)
                continue
