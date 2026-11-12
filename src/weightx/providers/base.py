# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Base protocol and message dataclasses for the WeightX Provider Layer."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from weightx.tools.registry import ToolRegistry


@dataclass
class LLMToolCall:
    """A single structured tool invocation requested by the LLM."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMMessage:
    """A conversation turn (`system`, `user`, `assistant`, or `tool`)."""

    role: str
    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class LLMResponse:
    """Normalized response returned by any `LLMProvider.generate()` implementation."""

    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    provider_name: str = "unknown"
    model_name: str = "unknown"


_TOOL_BLOCK_RE = re.compile(
    r"<weightx_tool_call>\s*(\{.*?\})\s*</weightx_tool_call>",
    re.DOTALL,
)


def parse_embedded_tool_calls(text: str) -> list[LLMToolCall]:
    """Extract `<weightx_tool_call>{"name": "...", "arguments": {...}}</weightx_tool_call>` blocks.

    Used by CLI agent providers and text-only local endpoints that do not support
    native HTTP tool-call schemas.
    """
    calls: list[LLMToolCall] = []
    for idx, match in enumerate(_TOOL_BLOCK_RE.finditer(text)):
        raw_json = match.group(1)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed:
                args = parsed.get("arguments", parsed.get("args", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append(
                    LLMToolCall(
                        call_id=f"call_emb_{idx + 1}",
                        name=str(parsed["name"]),
                        arguments=dict(args or {}),
                    )
                )
        except Exception:
            continue
    return calls


class LLMProvider(ABC):
    """Abstract base interface for connecting any LLM brain to the WeightX harness."""

    provider_name: str = "base"
    model_name: str = "unknown"

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        tools: ToolRegistry | None = None,
    ) -> LLMResponse:
        """Send conversation `messages` and available `tools` to the LLM and return its turn."""
        raise NotImplementedError
