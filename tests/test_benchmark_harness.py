# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast unit tests verifying benchmark evaluation suite math and edge cases."""

from __future__ import annotations

import torch

from benchmarks.eval_suites import (
    MCItem,
    _extract_python_completion,
    eval_counterfact_facts,
    eval_lambada,
    eval_mc_acc,
    eval_text_nll_top1,
    eval_wikitext2_ppl,
)
from examples.mvp_end_to_end_demo import MiniLlamaConfig, MiniLlamaForCausalLM


class _CharTok:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [(ord(ch) % 60) + 4 for ch in text]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def __call__(self, text: str, return_tensors: str | None = None, add_special_tokens: bool = False):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


def test_eval_suites_smoke():
    model = MiniLlamaForCausalLM(MiniLlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2))
    model.eval()
    tok = _CharTok()

    wt = eval_wikitext2_ppl(model, tok, [{"text": "hello world this is a wikitext line " * 6}], max_length=32, stride=16)
    assert wt["ppl"] > 1.0 and wt["tokens_evaluated"] > 0

    lb = eval_lambada(model, tok, [{"text": "the quick brown fox jumps"}])
    assert lb["n_items"] == 1 and lb["word_ppl"] >= 1.0

    mc = eval_mc_acc(model, tok, [MCItem("Question: sky?\nAnswer:", [" blue", " green"], 0)])
    assert mc["n_items"] == 1 and 0.0 <= mc["acc_norm"] <= 1.0

    tx = eval_text_nll_top1(model, tok, ["alpha beta gamma delta"])
    assert tx["ppl"] > 1.0 and 0.0 <= tx["top1_acc"] <= 1.0

    cf = eval_counterfact_facts(model, tok, [{"prompt": "Paris is in", "target_new": " Italy", "target_true": " France"}])
    assert cf["n_evaluated"] == 1

    comp = _extract_python_completion("def f():\n", "    return 1\n\ndef other():\n    pass\n")
    assert "other" not in comp and "return 1" in comp
