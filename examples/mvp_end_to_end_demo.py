# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX End-to-End MVP Demonstration.

Demonstrates the complete 5-stage Weight-Space Programming cycle on a Llama-family
Causal Transformer Language Model (RMSNorm + SwiGLU Gated MLP + Q/K/V/O Attention):
  1. Recon (L1 AutoArch + L2 13-Metric Uni-Layer Profile + L2 Logit Lens & Gini KCI)
  2. Localization (Pinpointing corrupted outlier neurons in Layer 2 without overflowing context)
  3. Declarative YAML Planning (BrainSurgery-WX plan with mandatory shape/dtype/finite assertions)
  4. Execution with IWCKEdit Fisher Protection + MEMIT-MASS Batch Editing (50 simultaneous facts)
  5. Vespasian Held-Out Verification & Automatic Rollback on Regression + TOON Token Savings Report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from weightx import (
    AutonomousWSPProvider,
    BatchFactSpec,
    EvaluationSplit,
    WeightXHarness,
    apply_memit_mass_batch,
    compare_toon_vs_json,
    execute_yaml_plan,
)


@dataclass
class MiniLlamaConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 172
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class LlamaAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)


class LlamaGatedMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: MiniLlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = LlamaAttentionBlock(cfg.hidden_size, cfg.num_attention_heads)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = LlamaGatedMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


@dataclass
class CausalLMOut:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


class MiniLlamaForCausalLM(nn.Module):
    """Canonical HuggingFace-named Llama-family decoder model for fast, reproducible WSP demos."""

    def __init__(self, cfg: MiniLlamaConfig | None = None):
        super().__init__()
        self.config = cfg or MiniLlamaConfig()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.model.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.model.norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        **_kwargs: Any,
    ) -> CausalLMOut:
        h = self.model.embed_tokens(input_ids)
        states = [h] if output_hidden_states else None
        for layer in self.model.layers:
            h = layer(h)
            if states is not None:
                states.append(h)
        h_norm = self.model.norm(h)
        logits = self.lm_head(h_norm)
        return CausalLMOut(
            logits=logits,
            hidden_states=tuple(states) if states is not None else None,
        )


def build_corrupted_model_and_splits(
    seed: int = 2026,
) -> tuple[MiniLlamaForCausalLM, EvaluationSplit, EvaluationSplit, torch.Tensor]:
    """Create a causal LM whose clean mapping is corrupted by outlier spikes in Layer 2."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = MiniLlamaConfig()
    model = MiniLlamaForCausalLM(cfg)
    model.eval()

    # Construct 32 deterministic sequences: 16 for Practice split, 16 for Hidden Held-Out split
    all_inputs = torch.randint(4, cfg.vocab_size - 4, (32, 10), dtype=torch.long)
    with torch.no_grad():
        clean_logits = model(all_inputs).logits[:, -1, :]
        clean_targets = clean_logits.argmax(dim=-1)

        # Inject a severe outlier fault in Layer 2's MLP down_proj rows 5, 11, and 19
        # (simulating a weight corruption / over-edited neuron anomaly)
        w_down = model.model.layers[2].mlp.down_proj.weight
        w_down[5, :] *= 16.0
        w_down[11, :] *= 15.0
        w_down[19, :] *= 14.0

    practice_split = EvaluationSplit(
        name="practice",
        input_ids=all_inputs[:16],
        target_ids=clean_targets[:16],
        reference_input_ids=all_inputs[:8],
    )
    held_out_split = EvaluationSplit(
        name="held_out_hidden",
        input_ids=all_inputs[16:],
        target_ids=clean_targets[16:],
        reference_input_ids=all_inputs[16:24],
    )
    calib_ids = all_inputs[:6]
    return model, practice_split, held_out_split, calib_ids


def run_mvp_demo() -> dict[str, Any]:
    """Execute the complete WeightX MVP demonstration and print a structured report."""
    print("=" * 78)
    print("WeightX (GPLv3) — Weight-Space Programming Harness End-to-End MVP Demo")
    print("=" * 78)

    model, practice_split, held_out_split, calib_ids = build_corrupted_model_and_splits()

    harness = WeightXHarness(
        model,
        provider=AutonomousWSPProvider(zero_outliers=False),
        practice_split=practice_split,
        held_out_split=held_out_split,
        calibration_input_ids=calib_ids,
        fisher_damping_alpha=0.15,
        fisher_freeze_top_fraction=0.05,
        improvement_threshold=0.01,
        max_kl_drift=2.0,
    )

    print(f"\n[1] Initial Corrupted Held-Out Score: {harness.vespasian.baseline_score:.4f}")
    run_res = harness.run_editing_cycle(
        goal="Localize and repair outlier neuron corruption in the causal LM using a declarative BrainSurgery-WX YAML plan.",
        max_turns=6,
    )

    print(f"[2] Completed 5-Stage WSP Loop in {run_res.turns_executed} turns ({run_res.tool_calls_executed} tool calls):")
    print(f"    • Baseline Held-Out Score : {run_res.baseline_held_out_score:.4f}")
    print(f"    • Post-Repair Best Score  : {run_res.final_best_held_out_score:.4f} (Delta: {run_res.net_improvement:+.4f})")
    print(f"    • Changed Parameters      : {run_res.diff_summary.changed_params:,} / {run_res.diff_summary.total_params:,} ({run_res.diff_summary.changed_fraction:.4%})")
    for v in run_res.vespasian_history:
        print(f"    • Vespasian Verdict #{v.iteration}   : {v.verdict_reason}")

    # Demonstrate Automatic Rollback on a bad/destructive YAML edit
    print("\n[3] Demonstrating Vespasian Automatic Rollback on a Destructive Edit:")
    bad_plan = """
transforms:
  - assert:
      exists: { of: "model.layers.0.self_attn.q_proj.weight" }
  - scale_:
      target: "model.layers.0.self_attn.q_proj.weight"
      by: -50.0
  - assert:
      finite: { of: "model.layers.0.self_attn.q_proj.weight" }
"""
    execute_yaml_plan(harness.workspace, bad_plan, apply_fisher_guard=False)
    rollback_verdict = harness.vespasian.verify_and_decide(
        hypothesis="Intentional destructive edit to verify automatic rollback"
    )
    print(f"    • Candidate Score         : {rollback_verdict.candidate_score:.4f}")
    print(f"    • Rollback Triggered      : {rollback_verdict.rolled_back}")
    print(f"    • Restored Best Score     : {rollback_verdict.best_score_after:.4f}")
    print(f"    • Verdict                 : {rollback_verdict.verdict_reason}")

    # Demonstrate MEMIT-MASS Simultaneous Batch Associative Editing (50 facts across layers 1, 2, 3)
    print("\n[4] Demonstrating MEMIT-MASS Batch Editing (50 Facts Across 3 MLP Layers):")
    rng = np.random.default_rng(7)
    d_in = model.config.intermediate_size
    d_out = model.config.hidden_size
    # Orthogonal-like random unit keys
    raw_keys = rng.standard_normal((50, d_in))
    raw_keys /= np.linalg.norm(raw_keys, axis=1, keepdims=True)
    target_vals = rng.standard_normal((50, d_out)) * 0.25

    facts = [
        BatchFactSpec(fact_id=f"fact_{i}", key_vector=raw_keys[i], target_value_vector=target_vals[i])
        for i in range(50)
    ]
    memit_targets = [
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.2.mlp.down_proj.weight",
        "model.layers.3.mlp.down_proj.weight",
    ]
    memit_res = apply_memit_mass_batch(
        harness.workspace,
        memit_targets,
        facts,
        lambda_cov=1e-2,
        apply_fisher_guard=True,
    )
    print(f"    • Batch Size              : {memit_res.num_facts} facts across {len(memit_res.target_tensors)} layers")
    print(f"    • Pre-Edit Memory MSE     : {memit_res.pre_edit_mse:.6f}")
    print(f"    • Post-Edit Memory MSE    : {memit_res.post_edit_mse:.6f} ({memit_res.relative_error_reduction * 100:.1f}% error reduction)")

    # Restore best checkpoint after standalone MEMIT demo
    harness.workspace.restore("vespasian_best")

    # Measure TOON vs JSON token savings on the L3 tensor table
    tensor_table = {"tensors": harness.workspace.list_tensors_table()}
    savings = compare_toon_vs_json(tensor_table)
    print("\n[5] TOON Serialization Token Efficiency:")
    print(
        f"    • JSON: {savings.json_chars:,} chars (~{savings.json_est_tokens} tokens) -> "
        f"TOON: {savings.toon_chars:,} chars (~{savings.toon_est_tokens} tokens) "
        f"[{savings.token_reduction_ratio * 100:.1f}% token reduction]"
    )
    print("=" * 78)

    return {
        "run_result": run_res.to_dict(),
        "rollback_verdict": rollback_verdict.to_dict(),
        "memit_result": memit_res.to_dict(),
        "toon_savings": savings.to_dict(),
    }


if __name__ == "__main__":
    run_mvp_demo()
