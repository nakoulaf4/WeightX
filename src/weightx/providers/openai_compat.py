# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
OpenAI-Compatible Provider Adapter.

Supports:
  - OpenAI (`https://api.openai.com/v1`)
  - OpenRouter (`https://openrouter.ai/api/v1`)
  - vLLM (`http://localhost:8000/v1`)
  - Ollama (`http://localhost:11434/v1`)
  - LM Studio (`http://localhost:1234/v1`)
  - Any custom `/v1/chat/completions` endpoint.

Handles both native OpenAI JSON-Schema `tools` / `tool_calls` AND fallback
`<weightx_tool_call>` parsing for local models without native function calling.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from weightx.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    parse_embedded_tool_calls,
)
from weightx.tools.registry import ToolRegistry


class OpenAICompatibleProvider(LLMProvider):
    """Universal adapter for OpenAI-compatible HTTP chat completion endpoints."""

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        provider_label: str = "openai-compatible",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_sec: float = 60.0,
        use_native_tools: bool = True,
        http_client: httpx.Client | None = None,
    ):
        self.model_name = model
        self.provider_name = provider_label
        self.base_url = base_url.rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or "local-no-key-required"
        )
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_sec = float(timeout_sec)
        self.use_native_tools = bool(use_native_tools)
        self._client = http_client

    @classmethod
    def for_openai(cls, model: str = "gpt-4o", api_key: str | None = None, **kw: Any) -> OpenAICompatibleProvider:
        return cls(
            model=model,
            base_url="https://api.openai.com/v1",
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            provider_label="openai",
            **kw,
        )

    @classmethod
    def for_openrouter(
        cls,
        model: str = "anthropic/claude-3.7-sonnet",
        api_key: str | None = None,
        **kw: Any,
    ) -> OpenAICompatibleProvider:
        return cls(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            provider_label="openrouter",
            **kw,
        )

    @classmethod
    def for_vllm(
        cls,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        **kw: Any,
    ) -> OpenAICompatibleProvider:
        return cls(
            model=model,
            base_url=base_url,
            api_key="vllm-local",
            provider_label="vllm",
            **kw,
        )

    @classmethod
    def for_ollama(
        cls,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434/v1",
        **kw: Any,
    ) -> OpenAICompatibleProvider:
        return cls(
            model=model,
            base_url=base_url,
            api_key="ollama-local",
            provider_label="ollama",
            **kw,
        )

    @classmethod
    def for_lm_studio(
        cls,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        **kw: Any,
    ) -> OpenAICompatibleProvider:
        return cls(
            model=model,
            base_url=base_url,
            api_key="lm-studio-local",
            provider_label="lm-studio",
            **kw,
        )

    def _format_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id or "call_0",
                        "content": m.content,
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                formatted.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.call_id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                formatted.append({"role": m.role, "content": m.content})
        return formatted

    def generate(
        self,
        messages: list[LLMMessage],
        tools: ToolRegistry | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools is not None and self.use_native_tools and tools.names():
            payload["tools"] = tools.to_openai_tools()
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        if self._client is not None:
            resp = self._client.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        else:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(url, json=payload, headers=headers)

        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls: list[LLMToolCall] = []

        for raw_tc in choice.get("tool_calls") or []:
            fn = raw_tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            tool_calls.append(
                LLMToolCall(
                    call_id=str(raw_tc.get("id", f"call_{len(tool_calls) + 1}")),
                    name=str(fn.get("name", "")),
                    arguments=args,
                )
            )

        if not tool_calls and content:
            tool_calls = parse_embedded_tool_calls(content)

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            provider_name=self.provider_name,
            model_name=self.model_name,
        )
