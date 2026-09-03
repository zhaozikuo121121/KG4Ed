from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import socket
import urllib.error
import urllib.request

from qg_v2.config import Settings


class LLMError(RuntimeError):
    pass


def extract_json_object(text: str) -> dict[str, Any]:
    value_text = text.strip().strip('`')
    if value_text.lower().startswith('json'):
        value_text = value_text[4:].strip()
    try:
        value = json.loads(value_text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = value_text.find('{')
    if start < 0:
        raise LLMError('No JSON object found in LLM output')
    depth = 0
    in_string = escape = False
    for index in range(start, len(value_text)):
        char = value_text[index]
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                value = json.loads(value_text[start:index + 1])
                if isinstance(value, dict):
                    return value
                break
    raise LLMError('Unbalanced JSON object in LLM output')


@dataclass(slots=True)
class QwenClient:
    api_key: str
    base_url: str
    timeout: float = 60.0

    def chat_json(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps({'model': model, 'messages': messages, 'temperature': temperature, 'response_format': {'type': 'json_object'}}, ensure_ascii=False).encode('utf-8'),
            method='POST',
            headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json', 'Accept': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            raise LLMError(f'Qwen HTTP {exc.code}: {exc.read().decode("utf-8", errors="replace")}') from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise LLMError(f'Qwen request failed: {exc}') from exc
        return extract_json_object(payload['choices'][0]['message']['content'])


class LLMRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = None if settings.mock or not settings.api_key else QwenClient(settings.api_key, settings.base_url, settings.request_timeout)

    @property
    def is_mock(self) -> bool:
        return self.client is None

    def model_for(self, agent_name: str) -> str:
        return self.settings.models.for_agent(agent_name)

    def complete_json(self, agent_name: str, messages: list[dict[str, Any]], temperature: float = 0.2) -> dict[str, Any]:
        if self.client is None:
            raise LLMError('LLMRouter is in mock mode; no real client is available')
        return self.client.chat_json(model=self.model_for(agent_name), messages=messages, temperature=temperature)
