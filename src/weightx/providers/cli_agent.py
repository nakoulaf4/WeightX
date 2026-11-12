# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
CLI Agent Provider Adapter.

Supports external CLI coding/reasoning agents:
  - Claude Code (`claude -p`)
  - Codex (`codex exec`)
  - OpenCode (`opencode run`)
  - Antigravity (`agy`)
  - DSH (`dsh`)
  - Any custom CLI agent command template.

Maintains WeightX's strict isolation guarantee: the CLI agent runs in an empty
ephemeral working directory (so it cannot directly touch model checkpoints or
hidden held-out tests) and interacts with the target model solely by emitting
`<weightx_tool_call>{"name": "...", "arguments": {...}}</weightx_tool_call>` blocks.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from typing import Any

from weightx.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    parse_embedded_tool_calls,
)
from weightx.toon_io import estimate_tokens
from weightx.tools.registry import ToolRegistry


CLI_AGENT_PRESETS: dict[str, list[str]] = {
    "claude-code": ["claude", "-p", "{prompt}"],
    "codex": ["codex", "exec", "{prompt}"],
    "opencode": ["opencode", "run", "{prompt}"],
    "antigravity": ["agy", "prompt", "{prompt}"],
    "dsh": ["dsh", "-c", "{prompt}"],
}


class CLIAgentProvider(LLMProvider):
    """Bridges a CLI agent (Claude Code, Codex, OpenCode, Antigravity, DSH) to WeightX tools."""

    def __init__(
        self,
        agent_type: str = "claude-code",
        *,
        command_template: list[str] | None = None,
        timeout_sec: float = 120.0,
        env_overrides: dict[str, str] | None = None,
    ):
        self.provider_name = f"cli:{agent_type}"
        self.model_name = agent_type
        if command_template is not None:
            self.command_template = list(command_template)
        elif agent_type in CLI_AGENT_PRESETS:
            self.command_template = list(CLI_AGENT_PRESETS[agent_type])
        else:
            raise ValueError(
                f"Unknown CLI agent preset {agent_type!r}. "
                f"Choose from {list(CLI_AGENT_PRESETS)} or pass `command_template`."
            )
        self.timeout_sec = float(timeout_sec)
        self.env_overrides = dict(env_overrides or {})

    @staticmethod
    def detect_installed_agents() -> dict[str, str | None]:
        """Return a mapping of preset CLI agent names to their resolved binary paths on PATH."""
        found: dict[str, str | None] = {}
        for preset, cmd in CLI_AGENT_PRESETS.items():
            binary = cmd[0]
            found[preset] = shutil.which(binary)
        return found

    def _build_structured_prompt(
        self,
        messages: list[LLMMessage],
        tools: ToolRegistry | None,
    ) -> str:
        sections: list[str] = [
            "You are connected to the WeightX Weight-Space Programming Harness.",
            "To invoke a WeightX tool, output one or more blocks in the exact format:",
            '<weightx_tool_call>{"name": "<tool_name>", "arguments": {...}}</weightx_tool_call>',
        ]
        if tools is not None:
            sections.append("\nAvailable WeightX Tools (TOON):")
            sections.append(tools.to_prompt_catalog())

        sections.append("\nConversation Transcript:")
        for m in messages:
            if m.role == "tool":
                sections.append(f"[TOOL_RESULT id={m.tool_call_id}]\n{m.content}")
            else:
                sections.append(f"[{m.role.upper()}]\n{m.content}")

        return "\n\n".join(sections)

    def generate(
        self,
        messages: list[LLMMessage],
        tools: ToolRegistry | None = None,
    ) -> LLMResponse:
        full_prompt = self._build_structured_prompt(messages, tools)
        cmd = [
            part.replace("{prompt}", full_prompt) for part in self.command_template
        ]

        # Run inside an empty temporary directory so the CLI agent has no direct file access
        with tempfile.TemporaryDirectory(prefix="weightx_cli_agent_") as tmpdir:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )

        output_text = (proc.stdout or "").strip()
        if proc.returncode != 0 and not output_text:
            raise RuntimeError(
                f"CLI agent {self.model_name!r} exited with code {proc.returncode}: {proc.stderr}"
            )

        tool_calls = parse_embedded_tool_calls(output_text)
        return LLMResponse(
            content=output_text,
            tool_calls=tool_calls,
            input_tokens=estimate_tokens(full_prompt),
            output_tokens=estimate_tokens(output_text),
            provider_name=self.provider_name,
            model_name=self.model_name,
        )
