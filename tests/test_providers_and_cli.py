# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for Layer 1 Providers (OpenAI, Anthropic, CLI Agents, Custom) and End-to-End Harness."""

import json
import httpx

from examples.mvp_end_to_end_demo import build_corrupted_model_and_splits, run_mvp_demo
from weightx.providers import (
    AnthropicProvider,
    CLIAgentProvider,
    CustomCallableProvider,
    LLMMessage,
    OpenAICompatibleProvider,
    parse_embedded_tool_calls,
)


def test_openai_compatible_and_anthropic_providers_with_mock_transport():
    def _openai_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "qwen2.5-coder"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Inspecting architecture",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "l1_autoarch_inspect",
                                        "arguments": '{"resolve_with_calibration": true}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 15},
            },
        )

    client_oai = httpx.Client(transport=httpx.MockTransport(_openai_handler))
    oai = OpenAICompatibleProvider.for_ollama(model="qwen2.5-coder", http_client=client_oai)
    resp_oai = oai.generate([LLMMessage(role="user", content="Hello")])
    assert resp_oai.provider_name == "ollama"
    assert len(resp_oai.tool_calls) == 1
    assert resp_oai.tool_calls[0].name == "l1_autoarch_inspect"
    assert resp_oai.tool_calls[0].arguments == {"resolve_with_calibration": True}

    def _anthropic_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Profiling layers"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "l2_unilayer_profile",
                        "input": {"outlier_row_threshold": 2.5},
                    },
                ],
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        )

    client_ant = httpx.Client(transport=httpx.MockTransport(_anthropic_handler))
    ant = AnthropicProvider(api_key="test-key", http_client=client_ant)
    resp_ant = ant.generate([LLMMessage(role="user", content="Profile")])
    assert resp_ant.provider_name == "anthropic"
    assert len(resp_ant.tool_calls) == 1
    assert resp_ant.tool_calls[0].name == "l2_unilayer_profile"


def test_cli_agent_detection_and_embedded_tool_parsing():
    agents = CLIAgentProvider.detect_installed_agents()
    assert "claude-code" in agents
    assert "codex" in agents
    assert "opencode" in agents
    assert "antigravity" in agents
    assert "dsh" in agents

    sample_cli_output = (
        "I will inspect layer 2.\n"
        '<weightx_tool_call>{"name": "l3_list_tensors", "arguments": {"layer_idx": 2}}</weightx_tool_call>'
    )
    tcs = parse_embedded_tool_calls(sample_cli_output)
    assert len(tcs) == 1
    assert tcs[0].name == "l3_list_tensors"
    assert tcs[0].arguments == {"layer_idx": 2}

    custom = CustomCallableProvider(lambda _msgs, _tools: sample_cli_output)
    resp = custom.generate([LLMMessage(role="user", content="go")])
    assert len(resp.tool_calls) == 1


def test_end_to_end_mvp_demo_cycle_and_rollback():
    summary = run_mvp_demo()
    run_res = summary["run_result"]
    assert run_res["final_best_held_out_score"] > run_res["baseline_held_out_score"] + 0.12
    assert summary["rollback_verdict"]["rolled_back"] is True
    assert summary["memit_result"]["relative_error_reduction_pct"] > 70.0
    assert summary["toon_savings"]["token_reduction_pct"] > 30.0
