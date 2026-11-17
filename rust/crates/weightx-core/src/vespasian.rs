// SPDX-License-Identifier: GPL-3.0-or-later
//! Vespasian Verification & Automatic Rollback Engine.
//! Port of `weightx.execution.vespasian`.

use ndarray::{Array2, Array3};
use serde_json::{json, Value};

use crate::error::Result;
use crate::model::{cross_entropy_rows, softmax_rows, ForwardOut, ModelArch};
use crate::workspace::WeightWorkspace;

/// A labeled evaluation split. The held-out split is never exposed to LLM tools.
pub enum EvalKind {
    /// Next-token prediction with explicit targets: score = 0.7*acc + 0.3*p(target).
    Targets {
        input_ids: Array2<i64>,
        target_ids: Vec<i64>,
    },
    /// Per-sequence next-token NLL; item score = exp(-mean NLL).
    SeqNll { seqs: Vec<Vec<i64>> },
    /// Custom evaluator over (workspace, model).
    Custom(Box<dyn Fn(&WeightWorkspace, Option<&dyn ModelArch>) -> (f64, Vec<f64>) + Send + Sync>),
}

pub struct EvaluationSplit {
    pub name: String,
    pub kind: EvalKind,
    pub reference_input_ids: Option<Array2<i64>>,
}

impl EvaluationSplit {
    pub fn targets(name: &str, input_ids: Array2<i64>, target_ids: Vec<i64>, reference: Option<Array2<i64>>) -> Self {
        Self {
            name: name.to_string(),
            kind: EvalKind::Targets { input_ids, target_ids },
            reference_input_ids: reference,
        }
    }

    pub fn seq_nll(name: &str, seqs: Vec<Vec<i64>>, reference: Option<Array2<i64>>) -> Self {
        Self {
            name: name.to_string(),
            kind: EvalKind::SeqNll { seqs },
            reference_input_ids: reference,
        }
    }

    /// Clone splits with data-backed kinds (Targets / SeqNll); Custom cannot be cloned.
    pub fn try_clone(&self) -> Option<EvaluationSplit> {
        let kind = match &self.kind {
            EvalKind::Targets { input_ids, target_ids } => EvalKind::Targets {
                input_ids: input_ids.clone(),
                target_ids: target_ids.clone(),
            },
            EvalKind::SeqNll { seqs } => EvalKind::SeqNll { seqs: seqs.clone() },
            EvalKind::Custom(_) => return None,
        };
        Some(EvaluationSplit {
            name: self.name.clone(),
            kind,
            reference_input_ids: self.reference_input_ids.clone(),
        })
    }

    pub fn evaluate(&self, ws: &WeightWorkspace, model: Option<&dyn ModelArch>) -> (f64, Vec<f64>) {
        match &self.kind {
            EvalKind::Custom(f) => f(ws, model),
            EvalKind::Targets { input_ids, target_ids } => {
                let Some(model) = model else { return (0.0, vec![0.0]) };
                let out = match model.forward(ws, input_ids, false) {
                    Ok(o) => o,
                    Err(_) => return (0.0, vec![0.0]),
                };
                let last = last_logits(&out);
                let probs = softmax_rows(&last);
                let mut items = Vec::new();
                for (i, &t) in target_ids.iter().enumerate() {
                    let pred = argmax_row(&probs, i);
                    let acc = if pred as i64 == t { 1.0 } else { 0.0 };
                    let p_target = probs[[i, (t.max(0) as usize).min(probs.ncols() - 1)]] as f64;
                    items.push(0.7 * acc + 0.3 * p_target);
                }
                let mean = items.iter().sum::<f64>() / items.len().max(1) as f64;
                (mean, items)
            }
            EvalKind::SeqNll { seqs } => {
                let Some(model) = model else { return (0.0, vec![0.0]) };
                let mut items = Vec::new();
                for seq in seqs {
                    if seq.len() < 2 {
                        items.push(0.0);
                        continue;
                    }
                    let ids = Array2::from_shape_vec((1, seq.len()), seq.clone()).unwrap();
                    let out = match model.forward(ws, &ids, false) {
                        Ok(o) => o,
                        Err(_) => return (0.0, vec![0.0]),
                    };
                    let logits = shift_logits(&out);
                    let targets: Vec<i64> = seq[1..].to_vec();
                    let losses = cross_entropy_rows(&logits, &targets);
                    let mean_nll = losses.iter().sum::<f32>() / losses.len().max(1) as f32;
                    items.push((-mean_nll as f64).exp());
                }
                let mean = items.iter().sum::<f64>() / items.len().max(1) as f64;
                (mean, items)
            }
        }
    }
}

fn last_logits(out: &ForwardOut) -> Array2<f32> {
    let s = out.logits.shape()[1];
    out.logits
        .slice(ndarray::s![.., s - 1, ..])
        .to_owned()
}

fn shift_logits(out: &ForwardOut) -> Array2<f32> {
    let s = out.logits.shape()[1];
    out.logits
        .slice(ndarray::s![0, ..s - 1, ..])
        .to_owned()
}

fn argmax_row(probs: &Array2<f32>, i: usize) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (j, &v) in probs.row(i).iter().enumerate() {
        if v > bv {
            bv = v;
            best = j;
        }
    }
    best
}

#[derive(Debug, Clone)]
pub struct VespasianIterationResult {
    pub iteration: usize,
    pub accepted: bool,
    pub rolled_back: bool,
    pub baseline_score: f64,
    pub previous_best_score: f64,
    pub candidate_score: f64,
    pub best_score_after: f64,
    pub score_delta_vs_best: f64,
    pub paired_fixed_items: usize,
    pub paired_regressed_items: usize,
    pub kl_divergence_ref: f64,
    pub nonfinite_params: usize,
    pub changed_params_pct: f64,
    pub verdict_reason: String,
    pub hypothesis: String,
}

impl VespasianIterationResult {
    pub fn to_value(&self) -> Value {
        let r5 = |v: f64| (v * 1e5).round() / 1e5;
        json!({
            "iteration": self.iteration,
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "baseline_score": r5(self.baseline_score),
            "previous_best_score": r5(self.previous_best_score),
            "candidate_score": r5(self.candidate_score),
            "best_score_after": r5(self.best_score_after),
            "score_delta_vs_best": r5(self.score_delta_vs_best),
            "paired_fixed_items": self.paired_fixed_items,
            "paired_regressed_items": self.paired_regressed_items,
            "kl_divergence_ref": (self.kl_divergence_ref * 1e6).round() / 1e6,
            "nonfinite_params": self.nonfinite_params,
            "changed_params_pct": (self.changed_params_pct * 1e4).round() / 1e4,
            "verdict_reason": self.verdict_reason,
            "hypothesis": self.hypothesis,
        })
    }
    pub fn to_toon(&self) -> String {
        crate::toon::dumps_toon(&self.to_value())
    }
}

pub struct VespasianEngine {
    pub practice_split: Option<EvaluationSplit>,
    held_out_split: Option<EvaluationSplit>,
    pub improvement_threshold: f64,
    pub max_kl_drift: f64,
    pub max_changed_fraction: f64,
    pub practice_budget: usize,
    pub practice_calls_used: usize,
    pub iteration: usize,
    pub history: Vec<VespasianIterationResult>,
    pub baseline_score: f64,
    pub best_score: f64,
    best_item_scores: Vec<f64>,
    baseline_ref_probs: Option<Array3<f32>>,
}

impl VespasianEngine {
    pub fn new(
        ws: &mut WeightWorkspace,
        model: Option<&dyn ModelArch>,
        practice_split: Option<EvaluationSplit>,
        held_out_split: Option<EvaluationSplit>,
        improvement_threshold: f64,
        max_kl_drift: f64,
        max_changed_fraction: f64,
        practice_budget: usize,
    ) -> Self {
        ws.checkpoint("vespasian_best");
        let baseline_ref_probs = capture_reference_probs(
            ws,
            model,
            held_out_split.as_ref().or(practice_split.as_ref()),
        );
        let (baseline_score, best_item_scores) = if let Some(h) = &held_out_split {
            h.evaluate(ws, model)
        } else if let Some(p) = &practice_split {
            p.evaluate(ws, model)
        } else {
            (0.0, vec![])
        };
        Self {
            practice_split,
            held_out_split,
            improvement_threshold,
            max_kl_drift,
            max_changed_fraction,
            practice_budget,
            practice_calls_used: 0,
            iteration: 0,
            history: Vec::new(),
            baseline_score,
            best_score: baseline_score,
            best_item_scores,
            baseline_ref_probs,
        }
    }

    pub fn run_practice_probe(&mut self, ws: &WeightWorkspace, model: Option<&dyn ModelArch>) -> Value {
        if self.practice_calls_used >= self.practice_budget {
            return json!({
                "error": format!("Practice probe budget exhausted ({}/{})", self.practice_budget, self.practice_budget),
                "budget_remaining": 0,
            });
        }
        self.practice_calls_used += 1;
        let Some(split) = &self.practice_split else {
            return json!({
                "practice_score": 0.0,
                "budget_remaining": self.practice_budget - self.practice_calls_used,
                "note": "No practice split configured",
            });
        };
        let (score, items) = split.evaluate(ws, model);
        json!({
            "practice_score": (score * 1e5).round() / 1e5,
            "num_practice_items": items.len(),
            "per_item_scores": items.iter().take(16).map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(),
            "budget_remaining": self.practice_budget - self.practice_calls_used,
        })
    }

    pub fn verify_and_decide(
        &mut self,
        ws: &mut WeightWorkspace,
        model: Option<&dyn ModelArch>,
        hypothesis: &str,
    ) -> VespasianIterationResult {
        self.iteration += 1;

        let nonfinite: usize = ws
            .iter()
            .map(|(_, t)| t.iter().filter(|v| !v.is_finite()).count())
            .sum();
        let diff = ws.diff();
        let changed_frac = diff.changed_fraction();

        let eval_split = self.held_out_split.as_ref().or(self.practice_split.as_ref());
        let (cand_score, cand_items) = if nonfinite == 0 {
            match eval_split {
                Some(s) => s.evaluate(ws, model),
                None => (-1.0, vec![]),
            }
        } else {
            (-1.0, vec![])
        };

        let mut fixed_items = 0;
        let mut regressed_items = 0;
        if cand_items.len() == self.best_item_scores.len() {
            for (b, a) in self.best_item_scores.iter().zip(cand_items.iter()) {
                if *a > *b + 0.05 {
                    fixed_items += 1;
                } else if *a < *b - 0.05 {
                    regressed_items += 1;
                }
            }
        }

        let kl_drift = if nonfinite == 0 {
            self.compute_reference_kl(ws, model)
        } else {
            f64::INFINITY
        };

        let prev_best = self.best_score;
        let delta_vs_best = cand_score - prev_best;

        let mut reasons: Vec<String> = Vec::new();
        let mut accepted = true;

        if nonfinite > 0 {
            accepted = false;
            reasons.push(format!("REJECTED: {nonfinite} NaN/Inf parameters detected"));
        }
        if changed_frac > self.max_changed_fraction {
            accepted = false;
            reasons.push(format!(
                "REJECTED: changed parameter fraction {:.2}% exceeds limit {:.2}%",
                changed_frac * 100.0,
                self.max_changed_fraction * 100.0
            ));
        }
        if kl_drift > self.max_kl_drift {
            accepted = false;
            reasons.push(format!(
                "REJECTED: reference KL divergence {kl_drift:.4} exceeds locality limit {:.4}",
                self.max_kl_drift
            ));
        }
        if delta_vs_best < self.improvement_threshold {
            accepted = false;
            reasons.push(format!(
                "REJECTED: held-out score {cand_score:.5} did not improve over best {prev_best:.5} (delta={delta_vs_best:+.5}, required >= +{})",
                self.improvement_threshold
            ));
        }

        let (verdict_reason, rolled_back) = if accepted {
            self.best_score = cand_score;
            self.best_item_scores = cand_items;
            ws.checkpoint("vespasian_best");
            (
                format!(
                    "ACCEPTED & COMMITTED: held-out score improved {prev_best:.5} -> {cand_score:.5} ({delta_vs_best:+.5}), KL drift={kl_drift:.4} <= {}",
                    self.max_kl_drift
                ),
                false,
            )
        } else {
            let _ = ws.restore("vespasian_best");
            (
                format!("{} -> AUTOMATIC ROLLBACK to vespasian_best.", reasons.join("; ")),
                true,
            )
        };

        let result = VespasianIterationResult {
            iteration: self.iteration,
            accepted,
            rolled_back,
            baseline_score: self.baseline_score,
            previous_best_score: prev_best,
            candidate_score: cand_score,
            best_score_after: self.best_score,
            score_delta_vs_best: delta_vs_best,
            paired_fixed_items: fixed_items,
            paired_regressed_items: regressed_items,
            kl_divergence_ref: kl_drift,
            nonfinite_params: nonfinite,
            changed_params_pct: changed_frac * 100.0,
            verdict_reason,
            hypothesis: hypothesis.to_string(),
        };
        self.history.push(result.clone());
        result
    }

    fn compute_reference_kl(&self, ws: &WeightWorkspace, model: Option<&dyn ModelArch>) -> f64 {
        let Some(base) = &self.baseline_ref_probs else { return 0.0 };
        let split = self.held_out_split.as_ref().or(self.practice_split.as_ref());
        let cur = capture_reference_probs(ws, model, split);
        let Some(cur) = cur else { return 0.0 };
        if cur.shape() != base.shape() {
            return 0.0;
        }
        let mut kl_sum = 0.0f64;
        let mut count = 0usize;
        for (p, q) in base.iter().zip(cur.iter()) {
            let p = (*p as f64).max(1e-12);
            let q = (*q as f64).max(1e-12);
            kl_sum += p * (p.ln() - q.ln());
            count += 1;
        }
        let vocab = base.shape()[2];
        let n_positions = count / vocab.max(1);
        // sum over vocab, mean over batch*seq
        (kl_sum / n_positions.max(1) as f64).max(0.0)
    }
}

fn capture_reference_probs(
    ws: &WeightWorkspace,
    model: Option<&dyn ModelArch>,
    split: Option<&EvaluationSplit>,
) -> Option<Array3<f32>> {
    let split = split?;
    let model = model?;
    let ref_ids = split.reference_input_ids.as_ref().or_else(|| match &split.kind {
        EvalKind::Targets { input_ids, .. } => Some(input_ids),
        _ => None,
    })?;
    let out = model.forward(ws, ref_ids, false).ok()?;
    Some(softmax_3d(&out.logits))
}

fn softmax_3d(x: &Array3<f32>) -> Array3<f32> {
    let shp = x.shape().to_vec();
    let flat = x
        .clone()
        .into_shape_with_order((shp[0] * shp[1], shp[2]))
        .unwrap()
        .to_owned();
    let sm = softmax_rows(&flat);
    sm.into_shape_with_order((shp[0], shp[1], shp[2])).unwrap().to_owned()
}

#[allow(dead_code)]
fn _result_helper(r: &Result<()>) -> bool {
    r.is_ok()
}