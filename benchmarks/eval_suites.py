# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Standardized GPU-accelerated evaluation suites for WeightX model benchmarks.

Implements canonical, zero-leakage evaluations:
  1. WikiText-2 sliding-window perplexity (HF strided protocol)
  2. LAMBADA zero-shot exact final-word accuracy & word perplexity
  3. Multiple-choice log-likelihood accuracy (`acc` & `acc_norm`) for ARC-Easy, HellaSwag, SciQ
  4. Domain sequence NLL / Perplexity / Top-1 & Top-5 next-token accuracy (MBPP, HumanEval, SciQ, WSP)
  5. CounterFact batch rewrite Efficacy Score (ES), Magnitude (EM), and Specificity (NS)
  6. HumanEval sandboxed Pass@1 code execution
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def model_max_ctx(model: nn.Module, cap: int = 1024) -> int:
    cfg = getattr(model, "config", None)
    candidates = [
        getattr(cfg, "max_position_embeddings", None),
        getattr(cfg, "n_positions", None),
        getattr(cfg, "n_ctx", None),
    ]
    vals = [int(v) for v in candidates if isinstance(v, int) and v > 0]
    return min(cap, min(vals)) if vals else cap


def _safe_batch_size(model: nn.Module, requested: int) -> int:
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 50000))
    if vocab >= 100000:
        return max(1, min(requested, 2))
    return requested


@torch.no_grad()
def eval_wikitext2_ppl(
    model: nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int = 1024,
    stride: int = 512,
    max_tokens: int | None = None,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Sliding-window strided perplexity on WikiText-2 raw test split (batched for GPU speed)."""
    model.eval()
    dev = next(model.parameters()).device
    ctx_len = model_max_ctx(model, cap=max_length)
    eff_stride = min(stride, ctx_len)
    batch_size = 1 if int(getattr(getattr(model, "config", None), "vocab_size", 50000)) >= 100000 else batch_size

    # Encode per paragraph to avoid >model_max_length warnings while preserving exact token stream
    all_ids: list[int] = []
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    paras = [r["text"] for r in rows if r.get("text", "").strip()]
    for idx, p in enumerate(paras):
        all_ids.extend(tokenizer.encode(p, add_special_tokens=False))
        if idx + 1 < len(paras):
            all_ids.extend(sep_ids)
        if max_tokens is not None and len(all_ids) >= max_tokens:
            all_ids = all_ids[:max_tokens]
            break

    enc = torch.tensor(all_ids, dtype=torch.long)
    seq_len = int(enc.numel())
    nll_sum = 0.0
    tok_count = 0
    prev_end_loc = 0

    full_windows: list[tuple[torch.Tensor, torch.Tensor]] = []
    tail_windows: list[tuple[torch.Tensor, torch.Tensor]] = []

    for begin_loc in range(0, seq_len, eff_stride):
        end_loc = min(begin_loc + ctx_len, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            break
        inp = enc[begin_loc:end_loc]
        if inp.numel() < 2:
            break
        tgt = inp.clone()
        if trg_len < inp.numel():
            tgt[:-trg_len] = -100
        if inp.numel() == ctx_len:
            full_windows.append((inp, tgt))
        else:
            tail_windows.append((inp, tgt))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    for start in range(0, len(full_windows), batch_size):
        chunk = full_windows[start : start + batch_size]
        x_b = torch.stack([w[0] for w in chunk]).to(dev)
        y_b = torch.stack([w[1] for w in chunk]).to(dev)
        logits = model(x_b).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y_b[:, 1:].contiguous()
        n_valid = int((shift_labels != -100).sum().item())
        if n_valid > 0:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            nll_sum += float(loss)
            tok_count += n_valid

    for inp, tgt in tail_windows:
        x_b = inp.unsqueeze(0).to(dev)
        y_b = tgt.unsqueeze(0).to(dev)
        logits = model(x_b).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y_b[:, 1:].contiguous()
        n_valid = int((shift_labels != -100).sum().item())
        if n_valid > 0:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            nll_sum += float(loss)
            tok_count += n_valid

    mean_nll = nll_sum / max(tok_count, 1)
    ppl = float(math.exp(min(mean_nll, 20.0)))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "ppl": round(ppl, 4),
        "mean_nll": round(mean_nll, 5),
        "tokens_evaluated": tok_count,
    }


@torch.no_grad()
def eval_lambada(
    model: nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_items: int = 1000,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Exact final-word greedy prediction accuracy & target word perplexity on LAMBADA."""
    model.eval()
    dev = next(model.parameters()).device
    max_ctx = model_max_ctx(model, cap=1024)
    batch_size = _safe_batch_size(model, batch_size)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    samples: list[tuple[list[int], list[int]]] = []
    for row in rows[:max_items]:
        txt = row["text"].strip()
        if " " not in txt:
            continue
        ctx, word = txt.rsplit(" ", 1)
        full_ids = tokenizer.encode(txt, add_special_tokens=False)
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
        if len(full_ids) > len(ctx_ids) and full_ids[: len(ctx_ids)] == ctx_ids:
            prompt_ids = ctx_ids
            tgt_ids = full_ids[len(ctx_ids) :]
        else:
            prompt_ids = ctx_ids
            tgt_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not prompt_ids or not tgt_ids:
            continue
        if len(prompt_ids) + len(tgt_ids) > max_ctx:
            prompt_ids = prompt_ids[-(max_ctx - len(tgt_ids)) :]
        if not prompt_ids:
            continue
        samples.append((prompt_ids, tgt_ids))

    exact_hits = 0
    word_nll_sum = 0.0
    total = len(samples)

    for start in range(0, total, batch_size):
        chunk = samples[start : start + batch_size]
        seqs = [p + t for p, t in chunk]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=dev)
        attn_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
            attn_mask[i, : len(s)] = 1

        logits = model(input_ids, attention_mask=attn_mask).logits.float()
        for i, (p, t) in enumerate(chunk):
            p_len = len(p)
            t_len = len(t)
            pred_slice = logits[i, p_len - 1 : p_len - 1 + t_len, :]
            preds = pred_slice.argmax(dim=-1).tolist()
            if preds == t:
                exact_hits += 1
            log_probs = F.log_softmax(pred_slice, dim=-1)
            word_lp = sum(float(log_probs[j, tid].item()) for j, tid in enumerate(t))
            word_nll_sum += -word_lp

    acc = exact_hits / max(total, 1)
    mean_word_nll = word_nll_sum / max(total, 1)
    word_ppl = float(math.exp(min(mean_word_nll, 20.0)))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "acc": round(acc, 4),
        "word_ppl": round(word_ppl, 3),
        "n_items": total,
        "exact_hits": exact_hits,
    }


@dataclass
class MCItem:
    prompt: str
    choices: list[str]
    gold_idx: int


def build_arc_items(rows: list[dict[str, Any]], max_items: int | None = None) -> list[MCItem]:
    items: list[MCItem] = []
    subset = rows if max_items is None else rows[:max_items]
    for r in subset:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        ans = r["answerKey"]
        if ans not in labels:
            continue
        gold = labels.index(ans)
        items.append(MCItem(prompt=f"Question: {r['question']}\nAnswer:", choices=[f" {t}" for t in texts], gold_idx=gold))
    return items


def build_hellaswag_items(rows: list[dict[str, Any]], max_items: int = 500) -> list[MCItem]:
    items: list[MCItem] = []
    for r in rows[:max_items]:
        lbl = str(r.get("label", "")).strip()
        if not lbl.isdigit():
            continue
        gold = int(lbl)
        endings = r["endings"]
        if not (0 <= gold < len(endings)):
            continue
        items.append(MCItem(prompt=r["ctx"], choices=[f" {e}" for e in endings], gold_idx=gold))
    return items


def build_sciq_mc_items(rows: list[dict[str, Any]], seed: int = 42, max_items: int | None = None) -> list[MCItem]:
    rng = random.Random(seed)
    items: list[MCItem] = []
    subset = rows if max_items is None else rows[:max_items]
    for r in subset:
        choices = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        shuffled = [f" {choices[i]}" for i in order]
        gold = order.index(0)
        items.append(MCItem(prompt=f"Question: {r['question']}\nAnswer:", choices=shuffled, gold_idx=gold))
    return items


@torch.no_grad()
def eval_mc_acc(
    model: nn.Module,
    tokenizer: Any,
    items: list[MCItem],
    *,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Zero-shot multiple-choice log-likelihood accuracy (`acc` and length-normalized `acc_norm`)."""
    model.eval()
    dev = next(model.parameters()).device
    max_ctx = model_max_ctx(model, cap=1024)
    batch_size = _safe_batch_size(model, batch_size)
    bos = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    flat_entries: list[tuple[int, int, list[int], int, int]] = []
    for item_idx, item in enumerate(items):
        p_ids = tokenizer.encode(item.prompt, add_special_tokens=False)
        if bos is not None:
            p_ids = [bos] + p_ids
        for c_idx, choice in enumerate(item.choices):
            f_ids = tokenizer.encode(item.prompt + choice, add_special_tokens=False)
            if bos is not None:
                f_ids = [bos] + f_ids
            if len(f_ids) > len(p_ids) and f_ids[: len(p_ids)] == p_ids:
                c_len = len(f_ids) - len(p_ids)
                seq = f_ids
            else:
                c_ids = tokenizer.encode(choice, add_special_tokens=False)
                if not c_ids:
                    c_ids = [pad_id]
                seq = p_ids + c_ids
                c_len = len(c_ids)
            if len(seq) > max_ctx:
                overflow = len(seq) - max_ctx
                seq = seq[overflow:]
                c_len = min(c_len, len(seq) - 1)
            char_len = max(len(choice.strip()), 1)
            flat_entries.append((item_idx, c_idx, seq, c_len, char_len))

    raw_scores: list[list[float]] = [[float("-inf")] * len(it.choices) for it in items]
    norm_scores: list[list[float]] = [[float("-inf")] * len(it.choices) for it in items]

    for start in range(0, len(flat_entries), batch_size):
        chunk = flat_entries[start : start + batch_size]
        max_len = max(len(e[2]) for e in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=dev)
        attn_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=dev)
        for i, (_, _, seq, _, _) in enumerate(chunk):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=dev)
            attn_mask[i, : len(seq)] = 1

        logits = model(input_ids, attention_mask=attn_mask).logits.float()
        for i, (item_idx, c_idx, seq, c_len, char_len) in enumerate(chunk):
            seq_len = len(seq)
            if seq_len < 2 or c_len < 1:
                continue
            start_pos = max(0, seq_len - c_len - 1)
            pred_logits = logits[i, start_pos : seq_len - 1, :]
            tgt = torch.tensor(seq[seq_len - c_len : seq_len], dtype=torch.long, device=dev)
            lp = F.log_softmax(pred_logits, dim=-1)
            total_lp = float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item())
            raw_scores[item_idx][c_idx] = total_lp
            norm_scores[item_idx][c_idx] = total_lp / char_len

    hits_raw = 0
    hits_norm = 0
    for idx, it in enumerate(items):
        if int(np.argmax(raw_scores[idx])) == it.gold_idx:
            hits_raw += 1
        if int(np.argmax(norm_scores[idx])) == it.gold_idx:
            hits_norm += 1

    n = max(len(items), 1)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "acc": round(hits_raw / n, 4),
        "acc_norm": round(hits_norm / n, 4),
        "n_items": len(items),
    }


@torch.no_grad()
def eval_text_nll_top1(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    max_seq_len: int = 256,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate token-level NLL, Perplexity, Top-1 and Top-5 next-token accuracy (batched)."""
    model.eval()
    dev = next(model.parameters()).device
    max_ctx = model_max_ctx(model, cap=max_seq_len)
    batch_size = _safe_batch_size(model, batch_size)
    bos = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    seqs: list[list[int]] = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if bos is not None:
            ids = [bos] + ids
        ids = ids[:max_ctx]
        if len(ids) >= 2:
            seqs.append(ids)

    total_nll = 0.0
    total_toks = 0
    top1_hits = 0
    top5_hits = 0

    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        max_len = max(len(s) for s in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=dev)
        attn_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=dev)
        for i, s in enumerate(chunk):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
            attn_mask[i, : len(s)] = 1

        logits_b = model(input_ids, attention_mask=attn_mask).logits.float()
        for i, s in enumerate(chunk):
            L = len(s)
            lg = logits_b[i, : L - 1, :]
            tgt = input_ids[i, 1:L]
            total_nll += float(F.cross_entropy(lg, tgt, reduction="sum").item())
            top1_hits += int((lg.argmax(dim=-1) == tgt).sum().item())
            k = min(5, lg.shape[-1])
            top5_hits += int((torch.topk(lg, k=k, dim=-1).indices == tgt.unsqueeze(-1)).any(dim=-1).sum().item())
            total_toks += L - 1

    mean_nll = total_nll / max(total_toks, 1)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "ppl": round(float(math.exp(min(mean_nll, 20.0))), 4),
        "mean_nll": round(mean_nll, 5),
        "top1_acc": round(top1_hits / max(total_toks, 1), 4),
        "top5_acc": round(top5_hits / max(total_toks, 1), 4),
        "tokens": total_toks,
    }


def _check_humaneval_program(code_str: str, timeout_s: float = 2.5) -> bool:
    try:
        compile(code_str, "<humaneval>", "exec")
    except Exception:
        return False
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code_str],
            capture_output=True,
            timeout=timeout_s,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _extract_python_completion(prompt: str, generated: str) -> str:
    """Trim generated completion cleanly before a second top-level def/class/eof."""
    text = generated
    if "```python" in text:
        text = text.split("```python", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[0]
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if kept and line and not line.startswith((" ", "\t", "#")) and line.startswith(("def ", "class ", "if __name__", "print(", "assert ")):
            break
        kept.append(line)
    body = "\n".join(kept)
    if prompt.strip() in body:
        return body
    return prompt + body


@torch.no_grad()
def eval_humaneval_pass1(
    model: nn.Module,
    tokenizer: Any,
    problems: list[dict[str, Any]],
    *,
    max_items: int | None = None,
    max_new_tokens: int = 192,
) -> dict[str, Any]:
    """Greedy Pass@1 functional execution evaluation on OpenAI HumanEval."""
    model.eval()
    dev = next(model.parameters()).device
    subset = problems if max_items is None else problems[:max_items]
    passed = 0
    total = len(subset)
    eos_id = tokenizer.eos_token_id

    for prob in subset:
        prompt = prob["prompt"]
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(dev)
        attn_mask = torch.ones_like(input_ids)
        out = model.generate(
            input_ids,
            attention_mask=attn_mask,
            tokenizer=tokenizer,
            stop_strings=["\ndef ", "\nclass ", "\nif __name__", "\nassert ", "```"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=eos_id if eos_id is not None else 0,
            eos_token_id=eos_id,
            use_cache=True,
        )
        gen_tokens = out[0, input_ids.shape[1] :]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        program = _extract_python_completion(prompt, gen_text)
        full_test = f"{program}\n\n{prob['test']}\n\ncheck({prob['entry_point']})\n"
        if _check_humaneval_program(full_test, timeout_s=2.5):
            passed += 1

    return {
        "pass_at_1": round(passed / max(total, 1), 4),
        "passed": passed,
        "total": total,
    }


@torch.no_grad()
def eval_counterfact_facts(
    model: nn.Module,
    tokenizer: Any,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate rewrite Efficacy Score (ES), Magnitude (EM), and log-prob margin on CounterFact items.

    Each item must contain `prompt`, `target_new`, `target_true`.
    """
    model.eval()
    dev = next(model.parameters()).device
    es_hits = 0
    margins: list[float] = []
    prob_diffs: list[float] = []
    top1_new_hits = 0

    for item in items:
        prompt = item["prompt"]
        new_str = item["target_new"]
        true_str = item["target_true"]

        new_ids = tokenizer.encode(new_str, add_special_tokens=False)
        true_ids = tokenizer.encode(true_str, add_special_tokens=False)
        if not new_ids or not true_ids:
            continue
        t_new = new_ids[0]
        t_true = true_ids[0]

        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=dev)
        if tokenizer.bos_token_id is not None:
            bos = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=dev)
            ids = torch.cat([bos, ids], dim=1)
        logits = model(ids).logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        p_new = float(probs[t_new].item())
        p_true = float(probs[t_true].item())
        lp_new = float(log_probs[t_new].item())
        lp_true = float(log_probs[t_true].item())

        if p_new > p_true:
            es_hits += 1
        if int(logits.argmax().item()) == t_new:
            top1_new_hits += 1
        margins.append(lp_new - lp_true)
        prob_diffs.append(p_new - p_true)

    n = max(len(margins), 1)
    return {
        "efficacy_score": round(es_hits / n, 4),
        "top1_new_rate": round(top1_new_hits / n, 4),
        "mean_prob_margin": round(float(np.mean(prob_diffs)) if prob_diffs else 0.0, 5),
        "mean_logprob_margin": round(float(np.mean(margins)) if margins else 0.0, 4),
        "n_evaluated": len(margins),
    }
