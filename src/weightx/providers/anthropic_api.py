# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Anthropic Messages API Provider (`/v1/messages`)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from weightx.providers.base import LLMMessage, LLMProvider, LLMResponse, LLMToolCall
from weightx.tools.registry import ToolRegistry


class AnthropicProvider(LLMProvider):
    """Native Anthropic Messages API adapter supporting `tool_use` and `tool_result` blocks."""

    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250219",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_sec: float = 60.0,
        http_client: httpx.Client | None = None,
    ):
        self.provider_name = "anthropic"
        self.model_name = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout_sec = float(timeout_sec)
        self._client = http_client

    def generate(
        self,
        messages: list[LLMMessage],
        tools: ToolRegistry | None = None,
    ) -> LLMResponse:
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                api_messages.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.call_id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                api_messages.append({"role": "assistant", "content": blocks or m.content})
            elif m.role == "tool":
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id or "call_0",
                                "content": m.content,
                            }
                        ],
                    }
                )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": api_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools is not None and tools.names():
            payload["tools"] = tools.to_anthropic_tools()

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        url = f"{self.base_url}/messages"

        if self._client is not None:
            resp = self._client.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        else:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(url, json=payload, headers=headers)

        resp.raise_for_status()
        data = resp.json()

        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    LLMToolCall(
                        call_id=str(block.get("id", f"call_{len(tool_calls) + 1}")),
                        name=str(block.get("name", "")),
                        arguments=dict(block.get("input") or {}),
                    )
                )

        usage = data.get("usage", {})
        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            provider_name=self.provider_name,
            model_name=self.model_name,
        )
