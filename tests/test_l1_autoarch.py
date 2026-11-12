# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for L1 AutoArch engine and calibration likelihood disambiguation."""

from pathlib import Path
import torch

from examples.mvp_end_to_end_demo import MiniLlamaConfig, MiniLlamaForCausalLM
from weightx.l1_arch import inspect_architecture
from weightx.l3_weights import WeightWorkspace


def test_l1_autoarch_on_llama_module_and_safetensors(tmp_path: Path):
    cfg = MiniLlamaConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=3,
        num_attention_heads=4,
    )
    model = MiniLlamaForCausalLM(cfg)
    calib_ids = torch.randint(1, 90, (4, 8), dtype=torch.long)

    rep = inspect_architecture(model, calibration_input_ids=calib_ids)
    assert rep.family == "decoder_transformer"
    assert rep.d_model == 64
    assert rep.n_layers == 3
    assert rep.vocab_size == 96
    assert rep.mlp_type == "gated"
    assert rep.mlp_hidden_dim == 160
    assert rep.calibration_resolution is not None
    assert rep.calibration_resolution.chosen_heads == 4
    assert rep.calibration_resolution.chosen_head_dim == 16

    # Check tensor role classification
    roles_by_name = {t.name: t.role for t in rep.tensors}
    assert roles_by_name["model.layers.0.self_attn.q_proj.weight"] == "q"
    assert roles_by_name["model.layers.0.self_attn.k_proj.weight"] == "k"
    assert roles_by_name["model.layers.0.self_attn.v_proj.weight"] == "v"
    assert roles_by_name["model.layers.0.self_attn.o_proj.weight"] == "o"
    assert roles_by_name["model.layers.0.mlp.gate_proj.weight"] == "gate"
    assert roles_by_name["model.layers.0.mlp.up_proj.weight"] == "up"
    assert roles_by_name["model.layers.0.mlp.down_proj.weight"] == "down"
    assert roles_by_name["model.embed_tokens.weight"] == "embedding"
    assert roles_by_name["lm_head.weight"] == "lm_head"

    # Also test inspecting a saved .safetensors file directly
    ws = WeightWorkspace(model)
    st_path = ws.export_safetensors(tmp_path / "model.safetensors")
    rep_st = inspect_architecture(st_path)
    assert rep_st.family == "decoder_transformer"
    assert rep_st.n_layers == 3
    assert rep_st.d_model == 64
