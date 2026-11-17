// SPDX-License-Identifier: GPL-3.0-or-later
//! IWCKEdit / FisherEdit weight importance guard (port of `weightx.safety.iwck_fisher`).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::linalg::{median, quantile};
use crate::tensor::{self, Tensor};

#[derive(Debug, Clone)]
pub struct FisherProtectionReport {
    pub tensor_name: String,
    pub raw_l2_delta: f64,
    pub protected_l2_delta: f64,
    pub damping_ratio: f64,
    pub frozen_critical_elements: usize,
    pub outlier_bypass_elements: usize,
}

impl FisherProtectionReport {
    pub fn to_value(&self) -> Value {
        json!({
            "tensor_name": self.tensor_name,
            "raw_l2_delta": (self.raw_l2_delta * 1e6).round() / 1e6,
            "protected_l2_delta": (self.protected_l2_delta * 1e6).round() / 1e6,
            "damping_ratio": (self.damping_ratio * 1e4).round() / 1e4,
            "frozen_critical_elements": self.frozen_critical_elements,
            "outlier_bypass_elements": self.outlier_bypass_elements,
        })
    }
}

/// Weights-only spectral-magnitude Fisher proxy: `F = W^2 * row_factor`.
pub fn proxy_fisher_for_array(w: &Tensor) -> Tensor {
    let sq = w.mapv(|v| v * v);
    if w.ndim() >= 2 {
        let m2d = tensor::as_2d(w);
        let norms: Vec<f64> = m2d
            .rows()
            .into_iter()
            .map(|r| r.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt())
            .collect();
        let med = median(&norms) + 1e-8;
        let row_factor: Vec<f32> = norms
            .iter()
            .map(|&n| {
                let ratio = n / med;
                if ratio > 3.0 {
                    0.05f32
                } else {
                    (ratio as f32).clamp(0.1, 2.0)
                }
            })
            .collect();
        let shp = w.shape().to_vec();
        let rows = shp[0];
        let cols: usize = shp[1..].iter().product();
        let scaled = ndarray::Array2::from_shape_fn((rows, cols), |(i, j)| {
            sq[[i, j]] * row_factor[i]
        });
        scaled.into_dyn().into_shape_with_order(shp).unwrap().to_owned()
    } else {
        sq
    }
}

/// Diagonal empirical Fisher: mean squared next-token-CE gradients over reference
/// batches (port of `compute_empirical_fisher`).
pub fn compute_empirical_fisher(
    ws: &crate::workspace::WeightWorkspace,
    arch: &dyn crate::model::ModelArch,
    reference_batches: &[ndarray::Array2<i64>],
) -> crate::error::Result<HashMap<String, Tensor>> {
    let mut accum: HashMap<String, Tensor> = HashMap::new();
    let mut n_batches = 0usize;
    for ids in reference_batches {
        if ids.ncols() < 2 {
            continue;
        }
        let grads = arch.weight_grads(ws, ids)?;
        for (name, g) in grads {
            let sq = g.mapv(|v| v * v);
            accum
                .entry(name)
                .and_modify(|a| *a = &*a + &sq)
                .or_insert(sq);
        }
        n_batches += 1;
    }
    let denom = n_batches.max(1) as f32;
    Ok(accum.into_iter().map(|(k, v)| (k, v.mapv(|x| x / denom))).collect())
}

pub struct FisherGuard {
    pub fisher_map: HashMap<String, Tensor>,
    pub damping_alpha: f64,
    pub freeze_top_fraction: f64,
    pub outlier_bypass_ratio: f64,
    pub enabled: bool,
    pub history: Vec<FisherProtectionReport>,
}

impl FisherGuard {
    pub fn new(
        fisher_map: HashMap<String, Tensor>,
        damping_alpha: f64,
        freeze_top_fraction: f64,
        outlier_bypass_ratio: f64,
        enabled: bool,
    ) -> Self {
        Self {
            fisher_map,
            damping_alpha,
            freeze_top_fraction: freeze_top_fraction.clamp(0.0, 0.9),
            outlier_bypass_ratio,
            enabled,
            history: Vec::new(),
        }
    }

    pub fn disabled() -> Self {
        Self::new(HashMap::new(), 0.25, 0.05, 2.5, false)
    }

    /// Build a guard with empirical diagonal Fisher from reference batches
    /// (port of `FisherGuard.from_model`). Falls back to the lazy weights-only
    /// proxy when no batches are given.
    pub fn from_arch(
        ws: &crate::workspace::WeightWorkspace,
        arch: &dyn crate::model::ModelArch,
        reference_batches: Option<&[ndarray::Array2<i64>]>,
        damping_alpha: f64,
        freeze_top_fraction: f64,
        outlier_bypass_ratio: f64,
        enabled: bool,
    ) -> crate::error::Result<Self> {
        let fmap = match reference_batches {
            Some(batches) if !batches.is_empty() => {
                compute_empirical_fisher(ws, arch, batches)?
            }
            _ => HashMap::new(),
        };
        Ok(Self::new(fmap, damping_alpha, freeze_top_fraction, outlier_bypass_ratio, enabled))
    }

    pub fn get_fisher(&mut self, name: &str, original_weight: &Tensor) -> Tensor {
        let needs = match self.fisher_map.get(name) {
            Some(t) => t.shape() != original_weight.shape(),
            None => true,
        };
        if needs {
            let f = proxy_fisher_for_array(original_weight);
            self.fisher_map.insert(name.to_string(), f);
        }
        self.fisher_map[name].clone()
    }

    /// Filter `proposed_weight` against the Fisher importance map of `original_weight`.
    pub fn protect_update(
        &mut self,
        name: &str,
        original_weight: &Tensor,
        proposed_weight: &Tensor,
        allow_outlier_repair: bool,
    ) -> (Tensor, FisherProtectionReport) {
        let orig = original_weight;
        let prop = proposed_weight;
        let n = orig.len();
        let delta: Vec<f32> = orig.iter().zip(prop.iter()).map(|(a, b)| b - a).collect();
        let raw_l2 = delta.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();

        if !self.enabled || raw_l2 <= 1e-12 {
            let rep = FisherProtectionReport {
                tensor_name: name.to_string(),
                raw_l2_delta: raw_l2,
                protected_l2_delta: raw_l2,
                damping_ratio: 1.0,
                frozen_critical_elements: 0,
                outlier_bypass_elements: 0,
            };
            return (prop.clone(), rep);
        }

        let f_mat = self.get_fisher(name, orig);
        let f_mean = f_mat.iter().map(|v| *v as f64).sum::<f64>() / n.max(1) as f64;
        let f_norm: Vec<f32> = if f_mean <= 1e-20 {
            vec![0.0; n]
        } else {
            f_mat.iter().map(|v| ((*v as f64) / f_mean).clamp(0.0, 20.0) as f32).collect()
        };

        // Outlier bypass mask
        let mut outlier_mask = vec![false; n];
        if allow_outlier_repair {
            if orig.ndim() >= 2 {
                let m2d = tensor::as_2d(orig);
                let norms: Vec<f64> = m2d
                    .rows()
                    .into_iter()
                    .map(|r| r.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt())
                    .collect();
                let med = median(&norms);
                if med > 1e-12 {
                    let cols = m2d.ncols();
                    for (i, &rn) in norms.iter().enumerate() {
                        if rn >= self.outlier_bypass_ratio * med {
                            for j in 0..cols {
                                outlier_mask[i * cols + j] = true;
                            }
                        }
                    }
                }
            }
            let abs_vals: Vec<f64> = orig.iter().map(|v| (*v as f64).abs()).collect();
            let elem_med = median(&abs_vals) + 1e-8;
            for (i, v) in orig.iter().enumerate() {
                let vf = *v as f64;
                if !vf.is_finite() || vf.abs() >= self.outlier_bypass_ratio * 3.0 * elem_med {
                    outlier_mask[i] = true;
                }
            }
        }

        // 1. Soft IWCKEdit damping
        let mut damping: Vec<f32> = f_norm
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                if outlier_mask[i] {
                    1.0
                } else {
                    (1.0 / (1.0 + self.damping_alpha * f as f64)) as f32
                }
            })
            .collect();

        // 2. Hard top-p critical freeze
        let mut frozen_count = 0usize;
        if self.freeze_top_fraction > 0.0 && n >= 10 {
            let f64s: Vec<f64> = f_norm.iter().map(|v| *v as f64).collect();
            let thresh = quantile(&f64s, 1.0 - self.freeze_top_fraction);
            if thresh > 0.0 {
                for i in 0..n {
                    if (f_norm[i] as f64) >= thresh && !outlier_mask[i] && delta[i].abs() > 0.0 {
                        frozen_count += 1;
                        damping[i] = 0.0;
                    }
                }
            }
        }

        let safe: Vec<f32> = orig
            .iter()
            .zip(delta.iter())
            .zip(damping.iter())
            .map(|((o, d), g)| o + d * g)
            .collect();
        let prot_l2 = delta
            .iter()
            .zip(damping.iter())
            .map(|(d, g)| {
                let v = (*d as f64) * (*g as f64);
                v * v
            })
            .sum::<f64>()
            .sqrt();
        let changed_outlier = outlier_mask
            .iter()
            .zip(delta.iter())
            .filter(|(&m, &d)| m && d.abs() > 1e-12)
            .count();

        let rep = FisherProtectionReport {
            tensor_name: name.to_string(),
            raw_l2_delta: raw_l2,
            protected_l2_delta: prot_l2,
            damping_ratio: prot_l2 / raw_l2.max(1e-12),
            frozen_critical_elements: frozen_count,
            outlier_bypass_elements: changed_outlier,
        };
        self.history.push(rep.clone());
        (tensor::from_vec(orig.shape(), safe), rep)
    }
}