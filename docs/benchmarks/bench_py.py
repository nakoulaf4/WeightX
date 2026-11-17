# SPDX-License-Identifier: GPL-3.0-or-later
"""Python-side benchmark mirror of wx-bench (prints one JSON line per run)."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\runbo\Downloads\WeightX")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
WORK = Path(__file__).parent

import numpy as np
import torch
import torch.nn.functional as F

MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "tinygpt2": "sshleifer/tiny-gpt2",
    "minillama": "minillama-demo",
}


def lcg_ids(vocab, seq):
    state = 42
    data = []
    for _ in range(seq):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        data.append((state >> 33) % vocab)
    return torch.tensor([data], dtype=torch.long)


def stats(times):
    s = sorted(times)
    return {"mean_ms": sum(s) / len(s), "min_ms": s[0], "max_ms": s[-1], "iters": len(s)}


def load_model(slug, device="cpu"):
    from weightx.catalog import ModelCatalog
    t0 = time.perf_counter()
    cat = ModelCatalog()
    model, tok, entry = cat.load_model_and_tokenizer(MODELS[slug], device=device, dtype=torch.float32)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    return model, tok, load_ms


def load_ws(slug):
    from weightx.l3_weights.workspace import WeightWorkspace
    model, _, load_ms = load_model(slug)
    ws = WeightWorkspace(model)
    return ws, model, load_ms


def safetensors_path(slug):
    repo = MODELS[slug].replace("/", "--")
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo}" / "snapshots"
    hits = sorted(base.glob("*/model.safetensors"))
    if not hits:
        raise FileNotFoundError(f"no cached model.safetensors for {slug}")
    return str(hits[0])


def cmd_forward(slug, seq, iters, device):
    model, _, load_ms = load_model(slug, device)
    vocab = model.config.vocab_size
    ids = lcg_ids(vocab, seq).to(device)
    with torch.no_grad():
        for _ in range(3 if device == "cuda" else 1):
            model(ids)
        if device == "cuda":
            torch.cuda.synchronize()
        logits = model(ids).logits
        checksum = float(logits.double().sum().item())
        first5 = logits[0, -1, :5].float().cpu().tolist()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = model(ids)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return {"cmd": "forward", "model": slug, "device": device, "seq": seq, "load_ms": load_ms,
            "logits_sum": checksum, "logits_first5": first5, "timing": stats(times)}


def cmd_grads(slug, seq, iters, device):
    model, _, load_ms = load_model(slug, device)
    vocab = model.config.vocab_size
    ids = lcg_ids(vocab, seq).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    def one():
        model.zero_grad(set_to_none=True)
        logits = model(ids).logits
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab), ids[:, 1:].reshape(-1))
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        return sum(float((p.grad.double() ** 2).sum().item()) for p in model.parameters() if p.grad is not None)
    sq = one()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one()
        times.append((time.perf_counter() - t0) * 1000)
    n = sum(1 for p in model.parameters() if p.grad is not None)
    return {"cmd": "grads", "model": slug, "device": device, "seq": seq, "load_ms": load_ms,
            "grad_sq_sum": sq, "n_tensors": n, "timing": stats(times)}


def cmd_eamit(slug, facts_file, targets, iters):
    from weightx.l3_weights.eamit import BatchFactSpec, apply_eamit_batch
    ws, _, load_ms = load_ws(slug)
    raw = json.loads(Path(facts_file).read_text())
    facts = [BatchFactSpec(fact_id=f["id"], key_vector=f["key"], target_value_vector=f["value"]) for f in raw]
    times, pre, post = [], 0.0, 0.0
    pre_kl, post_kl = 0.0, 0.0
    for _ in range(iters):
        ws.restore("baseline")
        t0 = time.perf_counter()
        r = apply_eamit_batch(ws, targets, facts, lambda_cov=1e-2, apply_fisher_guard=True)
        times.append((time.perf_counter() - t0) * 1000)
        pre, post = r.pre_edit_mse, r.post_edit_mse
        pre_kl, post_kl = r.pre_alignment_kl, r.post_alignment_kl
    return {"cmd": "eamit", "model": slug, "facts": len(facts), "layers": len(targets),
            "load_ms": load_ms, "pre_mse": pre, "post_mse": post,
            "pre_alignment_kl": pre_kl, "post_alignment_kl": post_kl, "timing": stats(times)}


PLAN_YAML = """
transforms:
  - assert:
      exists: { of: "model.layers.0.self_attn.q_proj.weight" }
  - scale_:
      target: "model.layers.0.self_attn.q_proj.weight"
      by: 1.001
  - scale_:
      target: "model.layers.1.self_attn.q_proj.weight"
      by: 0.999
  - scale_:
      target: "model.layers.2.self_attn.q_proj.weight"
      by: 1.001
  - assert:
      finite: { of: "model.layers.0.self_attn.q_proj.weight" }
"""


def cmd_toon(payload_file, iters):
    from weightx.toon_io import dumps_toon, loads_toon
    payload = json.loads(Path(payload_file).read_text())
    enc, dec = [], []
    text = ""
    for _ in range(iters):
        t0 = time.perf_counter()
        text = dumps_toon(payload)
        enc.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        v = loads_toon(text)
        dec.append((time.perf_counter() - t0) * 1000)
    return {"cmd": "toon", "chars": len(text), "encode": stats(enc), "decode": stats(dec)}


def cmd_plan(slug, iters):
    from weightx.declarative.plan_engine import execute_yaml_plan
    ws, _, load_ms = load_ws(slug)
    times, ok = [], True
    for _ in range(iters):
        ws.restore("baseline")
        t0 = time.perf_counter()
        r = execute_yaml_plan(ws, PLAN_YAML, dry_run=False, require_assertions=True, apply_fisher_guard=False)
        times.append((time.perf_counter() - t0) * 1000)
        ok = r.success
    return {"cmd": "plan", "model": slug, "load_ms": load_ms, "success": ok, "timing": stats(times)}


def cmd_wsio(path, iters):
    from safetensors.torch import load_file
    from weightx.l3_weights.workspace import WeightWorkspace
    tmp = WORK / "py_bench_export.safetensors"
    loads, exports = [], []
    params = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        state = load_file(path)
        ws = WeightWorkspace(state)
        loads.append((time.perf_counter() - t0) * 1000)
        params = ws.total_params()
        t0 = time.perf_counter()
        ws.export_safetensors(tmp, only_changed=False)
        exports.append((time.perf_counter() - t0) * 1000)
    return {"cmd": "wsio", "params": params, "load": stats(loads), "export": stats(exports)}


def cmd_describe(slug, name, iters):
    ws, _, load_ms = load_ws(slug)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        v = ws.describe_tensor(name, top_k=8)
        times.append((time.perf_counter() - t0) * 1000)
    return {"cmd": "describe", "model": slug, "load_ms": load_ms, "timing": stats(times)}


def cmd_l2(slug, seq, device):
    from weightx.l2_profile.unilayer import profile_model_layers
    model, _, load_ms = load_model(slug, device)
    ids = lcg_ids(model.config.vocab_size, seq).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    rep = profile_model_layers(model, calibration_input_ids=ids, model_name="bench", outlier_row_threshold=2.5)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return {"cmd": "l2", "model": slug, "device": device, "seq": seq, "load_ms": load_ms,
            "layers": rep.num_layers, "elapsed_ms": ms}


def cmd_sandbox(slug, name, iters):
    from weightx.execution.sandbox import IsolatedSandbox
    ws, _, load_ms = load_ws(slug)
    code = f"W['{name}'] = W['{name}'] * 1.0001\nprint('ok')"
    sb = IsolatedSandbox()
    times, ok = [], True
    for _ in range(iters):
        t0 = time.perf_counter()
        r = sb.run(ws, code, tensor_pattern=name)
        times.append((time.perf_counter() - t0) * 1000)
        ok = r.success
    return {"cmd": "sandbox", "model": slug, "load_ms": load_ms, "success": ok, "timing": stats(times)}


def cmd_fixtures():
    """Generate shared facts + TOON payload so both sides use identical inputs."""
    from safetensors.torch import load_file
    meta = {}
    for slug in ("smollm2", "tinygpt2", "minillama"):
        if slug == "minillama":
            model, _, _ = load_model(slug)
            state = {k: v.detach() for k, v in model.state_dict().items()}
        else:
            state = load_file(safetensors_path(slug))
        if slug == "tinygpt2":
            cands = sorted(k for k in state if k.endswith("mlp.c_proj.weight"))
        else:
            cands = sorted(k for k in state if k.endswith("mlp.down_proj.weight"))
        if slug == "smollm2":
            targets = [f"model.layers.{i}.mlp.down_proj.weight" for i in (13, 17, 22, 26)]
        elif slug == "minillama":
            targets = [f"model.layers.{i}.mlp.down_proj.weight" for i in (1, 2, 3)]
        else:
            targets = cands[:4]
        d_out, d_in = state[targets[-1]].shape
        rng = np.random.default_rng(7)
        facts = []
        for i in range(50):
            key = rng.standard_normal(d_in)
            key = key / np.linalg.norm(key)
            val = rng.standard_normal(d_out) * 0.25
            facts.append({"id": f"fact_{i}", "key": key.tolist(), "value": val.tolist()})
        ff = WORK / f"facts_{slug}.json"
        ff.write_text(json.dumps(facts))
        meta[slug] = {"targets": targets, "facts_file": str(ff), "d_in": int(d_in), "d_out": int(d_out),
                      "safetensors": safetensors_path(slug) if slug != "minillama" else None}
    # TOON payload: real L3 tensor table of SmolLM2
    from weightx.l3_weights.workspace import WeightWorkspace
    ws = WeightWorkspace(load_file(meta["smollm2"]["safetensors"]))
    payload = {"tensors": ws.list_tensors_table()}
    (WORK / "payload.json").write_text(json.dumps(payload))
    meta["payload"] = str(WORK / "payload.json")
    (WORK / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"cmd": "fixtures", "meta": meta}


def main():
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "forward":
        out = cmd_forward(a[1], int(a[2]), int(a[3]), a[4] if len(a) > 4 else "cpu")
    elif cmd == "grads":
        out = cmd_grads(a[1], int(a[2]), int(a[3]), a[4] if len(a) > 4 else "cpu")
    elif cmd == "eamit":
        out = cmd_eamit(a[1], a[2], a[3].split(","), int(a[4]))
    elif cmd == "toon":
        out = cmd_toon(a[1], int(a[2]))
    elif cmd == "plan":
        out = cmd_plan(a[1], int(a[2]))
    elif cmd == "wsio":
        out = cmd_wsio(a[1], int(a[2]))
    elif cmd == "describe":
        out = cmd_describe(a[1], a[2], int(a[3]))
    elif cmd == "l2":
        out = cmd_l2(a[1], int(a[2]), a[3] if len(a) > 3 else "cpu")
    elif cmd == "sandbox":
        out = cmd_sandbox(a[1], a[2], int(a[3]))
    elif cmd == "fixtures":
        out = cmd_fixtures()
    else:
        raise SystemExit(f"unknown cmd {cmd}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()