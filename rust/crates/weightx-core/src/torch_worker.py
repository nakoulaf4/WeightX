# SPDX-License-Identifier: GPL-3.0-or-later
"""WeightX hybrid torch sidecar worker.

Persistent JSONL process spawned by the Rust core (`torch_backend.rs`).
Reuses the reference Python implementation (`weightx` package) for heavy
numerics: forward/backward via PyTorch, EAMIT via numpy/LAPACK.

Protocol: one JSON request per stdin line, one JSON response per stdout line.
All diagnostics go to stderr. Large arrays are exchanged via .safetensors
files inside the exchange directory (argv[2]).

Usage: python -I torch_worker.py <repo_src_dir> <exchange_dir> <device>
"""
import json
import os
import sys
import traceback

SRC = sys.argv[1] if len(sys.argv) > 1 else ""
EXCHANGE = sys.argv[2] if len(sys.argv) > 2 else "."
DEVICE_REQ = sys.argv[3] if len(sys.argv) > 3 else "auto"
if SRC and os.path.isdir(SRC):
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
from safetensors.numpy import load_file, save_file  # noqa: E402

STATE = {"model": None, "state": {}, "device": "cpu", "model_type": "llama"}


def _path(name):
    return os.path.join(EXCHANGE, name)


def _resolve_device():
    import torch
    if DEVICE_REQ == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE_REQ


def _build_model(req):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfgd = req["config"]
    mt = cfgd.get("model_type", "llama")
    if mt == "minillama":
        mt = "llama"
    hf = req.get("hf_config")
    if hf:
        hf = {k: v for k, v in hf.items() if k != "model_type"}
        cfg = AutoConfig.for_model(mt, **hf)
    elif mt == "gpt2":
        cfg = AutoConfig.for_model(
            "gpt2",
            vocab_size=cfgd["vocab_size"], n_embd=cfgd["hidden_size"],
            n_head=cfgd["num_attention_heads"], n_layer=cfgd["num_hidden_layers"],
            n_inner=cfgd["intermediate_size"],
            activation_function=cfgd.get("hidden_act", "gelu_new"),
            layer_norm_epsilon=cfgd.get("layer_norm_eps", 1e-5),
        )
    else:
        cfg = AutoConfig.for_model(
            "llama",
            vocab_size=cfgd["vocab_size"], hidden_size=cfgd["hidden_size"],
            intermediate_size=cfgd["intermediate_size"],
            num_hidden_layers=cfgd["num_hidden_layers"],
            num_attention_heads=cfgd["num_attention_heads"],
            num_key_value_heads=cfgd["num_key_value_heads"],
            head_dim=cfgd["head_dim"], rms_norm_eps=cfgd.get("rms_norm_eps", 1e-5),
            rope_theta=cfgd.get("rope_theta", 10000.0),
            hidden_act=cfgd.get("hidden_act", "silu"),
            tie_word_embeddings=cfgd.get("tie_word_embeddings", False),
        )
    model = AutoModelForCausalLM.from_config(cfg)
    model.to(torch.float32)
    dev = _resolve_device()
    model.to(dev)
    model.eval()
    STATE["model"] = model
    STATE["device"] = dev
    STATE["model_type"] = mt
    _load_into_model()
    return {"device": dev, "model_type": mt}


def _load_into_model():
    import torch
    model = STATE["model"]
    if model is None:
        return
    sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in STATE["state"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[sidecar] missing keys: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[sidecar] unexpected keys: {unexpected}", file=sys.stderr)
    model.tie_weights()


def _mlp_out_modules():
    model = STATE["model"]
    if STATE["model_type"] == "gpt2":
        return [(f"transformer.h.{i}.mlp.c_proj.weight", h.mlp.c_proj)
                for i, h in enumerate(model.transformer.h)]
    return [(f"model.layers.{i}.mlp.down_proj.weight", l.mlp.down_proj)
            for i, l in enumerate(model.model.layers)]


def op_init(req):
    STATE["state"] = load_file(req["weights"])
    return _build_model(req)


def op_sync(req):
    STATE["state"].update(load_file(req["weights"]))
    _load_into_model()
    return {}


def op_forward(req):
    import torch
    model = STATE["model"]
    ids = torch.tensor(req["ids"], dtype=torch.long, device=STATE["device"])
    want_hidden = bool(req.get("want_hidden"))
    with torch.no_grad():
        out = model(ids, output_hidden_states=want_hidden)
    logits = np.ascontiguousarray(out.logits.float().cpu().numpy())
    save_file({"logits": logits}, _path("logits.safetensors"))
    n_hidden = 0
    if want_hidden:
        hs = {f"h{i}": np.ascontiguousarray(h.float().cpu().numpy())
              for i, h in enumerate(out.hidden_states[1:])}
        save_file(hs, _path("hidden.safetensors"))
        n_hidden = len(hs)
    return {"logits_shape": list(logits.shape), "n_hidden": n_hidden}


def op_grads(req):
    import torch
    import torch.nn.functional as F
    model = STATE["model"]
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    ids = torch.tensor(req["ids"], dtype=torch.long, device=STATE["device"])
    logits = model(ids).logits
    v = logits.shape[-1]
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, v), ids[:, 1:].reshape(-1),
                           reduction="mean")
    loss.backward()
    grads = {n: np.ascontiguousarray(p.grad.detach().float().cpu().numpy())
             for n, p in model.named_parameters() if p.grad is not None}
    save_file(grads, _path("grads.safetensors"))
    return {"names": list(grads.keys())}


def op_keys_grads(req):
    import torch
    import torch.nn.functional as F
    model = STATE["model"]
    need_grads = bool(req.get("need_grads"))
    for p in model.parameters():
        p.requires_grad_(need_grads)
    keys_all, grads_all = {}, {}
    caps_k, caps_g = {}, {}
    handles = []
    for name, mod in _mlp_out_modules():
        def make_hook(nm):
            def hook(m, inp, out):
                caps_k[nm] = inp[0].detach()
                if need_grads:
                    out.register_hook(lambda g, nm=nm: caps_g.__setitem__(nm, g.detach()))
            return hook
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        for seq in req["seqs"]:
            if len(seq) < 2:
                continue
            caps_k.clear()
            caps_g.clear()
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([seq], dtype=torch.long, device=STATE["device"])
            logits = model(ids).logits
            if need_grads:
                v = logits.shape[-1]
                loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, v),
                                       ids[:, 1:].reshape(-1), reduction="sum")
                loss.backward()
            for nm, k in caps_k.items():
                keys_all.setdefault(nm, []).append(k[0, :-1].double().cpu().numpy())
            if need_grads:
                for nm, g in caps_g.items():
                    grads_all.setdefault(nm, []).append(g[0, :-1].double().cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    keys = {nm: np.ascontiguousarray(np.concatenate(parts, axis=0))
            for nm, parts in keys_all.items()}
    grads = {nm: np.ascontiguousarray(np.concatenate(parts, axis=0))
             for nm, parts in grads_all.items()}
    save_file(keys, _path("keys.safetensors"))
    save_file(grads, _path("kgrads.safetensors"))
    return {"names": list(keys.keys())}


def op_eamit(req):
    from weightx.l3_weights.workspace import WeightWorkspace
    from weightx.l3_weights.eamit import BatchFactSpec, EamitAlignConfig, apply_eamit_batch
    fd = req.get("fisher_diag") or {}

    class _DiagGuard:
        """Adapter: exposes Rust-side Fisher column means through get_fisher()."""
        def get_fisher(self, name, w):
            col = np.asarray(fd[name], dtype=np.float64)
            return np.diag(col * w.shape[0]).astype(np.float32)

    state_np = {k: v.copy() for k, v in STATE["state"].items()}
    ws = WeightWorkspace(state_np, fisher_guard=_DiagGuard() if fd else None)
    facts = [
        BatchFactSpec(
            fact_id=str(f["id"]), key_vector=f["key"], target_value_vector=f["value"],
            layer_inputs={k: np.asarray(v, dtype=np.float64)
                          for k, v in (f.get("layer_inputs") or {}).items()},
        )
        for f in req["facts"]
    ]
    ac = req.get("align_config") or {}
    align_cfg = EamitAlignConfig(
        lambda_kl=float(ac.get("lambda_kl", 0.05)),
        lambda_mse=float(ac.get("lambda_mse", 0.25)),
        top_m=int(ac.get("top_m", 8)),
        tau=float(ac.get("tau", 0.1)),
        align_steps=int(ac.get("align_steps", 20)),
        align_lr=float(ac.get("align_lr", 0.03)),
        max_block_size=int(ac.get("max_block_size", 512)),
    )
    res = apply_eamit_batch(
        ws, req["targets"], facts,
        lambda_cov=float(req.get("lambda_cov", 1e-2)),
        apply_fisher_guard=bool(fd),
        align_config=align_cfg,
    )
    edited = {t: np.ascontiguousarray(ws.get_readonly(t)) for t in req["targets"]}
    save_file(edited, _path("edited.safetensors"))
    return {
        "pre_alignment_kl": res.pre_alignment_kl,
        "post_alignment_kl": res.post_alignment_kl,
        "pre_alignment_top_m_mse": res.pre_alignment_top_m_mse,
        "post_alignment_top_m_mse": res.post_alignment_top_m_mse,
    }


OPS = {
    "init": op_init, "sync": op_sync, "forward": op_forward, "grads": op_grads,
    "keys_grads": op_keys_grads, "eamit": op_eamit,
    "ping": lambda req: {},
}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"bad request: {e}"}), flush=True)
            continue
        op = req.get("op", "")
        if op == "shutdown":
            print(json.dumps({"ok": True}), flush=True)
            return
        fn = OPS.get(op)
        if fn is None:
            print(json.dumps({"ok": False, "error": f"unknown op {op!r}"}), flush=True)
            continue
        try:
            t0 = __import__("time").perf_counter()
            resp = fn(req)
            resp["ok"] = True
            if os.environ.get("WEIGHTX_SIDECAR_LOG"):
                print(f"[sidecar] op={op} {( __import__('time').perf_counter() - t0) * 1000:.1f}ms",
                      file=sys.stderr, flush=True)
        except Exception:
            resp = {"ok": False, "error": traceback.format_exc()}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()