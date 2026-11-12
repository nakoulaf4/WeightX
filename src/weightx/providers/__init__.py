# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Provider Layer (Layer 1): Universal LLM interface for API providers, CLI agents,
local servers (Ollama, LM Studio, vLLM), and custom callables.
"""

from .anthropic_api import AnthropicProvider
from .base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    parse_embedded_tool_calls,
)
from .cli_agent import CLIAgentProvider
from .custom import AutonomousWSPProvider, CustomCallableProvider
from .openai_compat import OpenAICompatibleProvider

__all__ = [
    "AnthropicProvider",
    "AutonomousWSPProvider",
    "CLIAgentProvider",
    "CustomCallableProvider",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMToolCall",
    "OpenAICompatibleProvider",
    "parse_embedded_tool_calls",
]
