// SPDX-License-Identifier: GPL-3.0-or-later
//! Tensor storage (f32, row-major), safetensors I/O, and SHA-256 provenance.

use std::collections::HashMap;
use std::path::Path;

use ndarray::{Array2, ArrayD, IxDyn};
use sha2::{Digest, Sha256};

use crate::error::{Result, WeightXError};

pub type Tensor = ArrayD<f32>;

pub fn zeros(shape: &[usize]) -> Tensor {
    Tensor::zeros(IxDyn(shape))
}

pub fn ones(shape: &[usize]) -> Tensor {
    Tensor::ones(IxDyn(shape))
}

pub fn full(shape: &[usize], value: f32) -> Tensor {
    Tensor::from_elem(IxDyn(shape), value)
}

pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_shape_vec(IxDyn(shape), data).expect("shape/data mismatch")
}

pub fn numel(t: &Tensor) -> usize {
    t.len()
}

pub fn shape_of(t: &Tensor) -> Vec<usize> {
    t.shape().to_vec()
}

/// Reshape to 2D `(shape[0], prod(rest))` like `arr.reshape(arr.shape[0], -1)`.
pub fn as_2d(t: &Tensor) -> Array2<f32> {
    let shp = t.shape();
    if shp.len() == 1 {
        return t.clone().into_shape_with_order((1, shp[0])).unwrap().to_owned();
    }
    let rows = shp[0];
    let cols: usize = shp[1..].iter().product();
    t.clone()
        .into_shape_with_order((rows, cols))
        .unwrap()
        .to_owned()
}

pub fn sha256_tensor(t: &Tensor) -> String {
    let mut hasher = Sha256::new();
    for v in t.iter() {
        hasher.update(v.to_le_bytes());
    }
    let out = hasher.finalize();
    out.iter().map(|b| format!("{:02x}", b)).collect()
}

fn dtype_to_safetensors(name: &str) -> Result<safetensors::Dtype> {
    use safetensors::Dtype;
    Ok(match name {
        "float32" | "float" | "fp32" => Dtype::F32,
        "float64" | "double" => Dtype::F64,
        "float16" | "half" => Dtype::F16,
        "bfloat16" => Dtype::BF16,
        "int64" | "long" => Dtype::I64,
        "int32" | "int" => Dtype::I32,
        "int16" => Dtype::I16,
        "int8" => Dtype::I8,
        "uint8" => Dtype::U8,
        "bool" => Dtype::BOOL,
        other => return Err(WeightXError::Value(format!("Unsupported dtype {other:?}"))),
    })
}

/// Load a `.safetensors` file into f32 tensors (any numeric dtype is converted).
pub fn load_safetensors(path: &Path) -> Result<Vec<(String, Tensor)>> {
    Ok(load_safetensors_with_dtypes(path)?.0)
}

/// Like [`load_safetensors`], but also reports the original on-disk dtype names
/// (values are still converted to f32 for computation).
pub fn load_safetensors_with_dtypes(path: &Path) -> Result<(Vec<(String, Tensor)>, HashMap<String, String>)> {
    let data = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&data)?;
    let mut out = Vec::new();
    let mut dtypes = HashMap::new();
    for (name, view) in st.tensors() {
        let dtype_name = match view.dtype() {
            safetensors::Dtype::F32 => "float32",
            safetensors::Dtype::F64 => "float64",
            safetensors::Dtype::F16 => "float16",
            safetensors::Dtype::BF16 => "bfloat16",
            safetensors::Dtype::I64 => "int64",
            safetensors::Dtype::I32 => "int32",
            safetensors::Dtype::I16 => "int16",
            safetensors::Dtype::I8 => "int8",
            safetensors::Dtype::U8 => "uint8",
            safetensors::Dtype::BOOL => "bool",
            other => {
                return Err(WeightXError::Value(format!(
                    "Unsupported safetensors dtype {other:?} for {name}"
                )))
            }
        };
        dtypes.insert(name.to_string(), dtype_name.to_string());
        let shape: Vec<usize> = view.shape().to_vec();
        let n: usize = shape.iter().product();
        let raw = view.data();
        let vals: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => {
                // Fast path: aligned little-endian f32 data is a single memcpy.
                let ptr = raw.as_ptr();
                if cfg!(target_endian = "little") && (ptr as usize) % 4 == 0 && raw.len() == n * 4 {
                    unsafe { std::slice::from_raw_parts(ptr as *const f32, n) }.to_vec()
                } else {
                    raw.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                }
            }
            safetensors::Dtype::F64 => raw
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            safetensors::Dtype::F16 => raw
                .chunks_exact(2)
                .map(|b| half_f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            safetensors::Dtype::BF16 => raw
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect(),
            safetensors::Dtype::I64 => raw
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            safetensors::Dtype::I32 => raw
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            safetensors::Dtype::I16 => raw
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            safetensors::Dtype::I8 => raw.iter().map(|&b| b as i8 as f32).collect(),
            safetensors::Dtype::U8 => raw.iter().map(|&b| b as f32).collect(),
            safetensors::Dtype::BOOL => raw.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
            other => {
                return Err(WeightXError::Value(format!(
                    "Unsupported safetensors dtype {other:?} for {name}"
                )))
            }
        };
        if vals.len() != n {
            return Err(WeightXError::SafeTensors(format!(
                "Tensor {name}: expected {n} elements, got {}",
                vals.len()
            )));
        }
        out.push((name.to_string(), from_vec(&shape, vals)));
    }
    Ok((out, dtypes))
}

fn half_f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            (sign << 31) | (e << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// Save tensors as f32 `.safetensors` (mirrors `safetensors.numpy.save_file`).
/// Zero-copy on little-endian targets: contiguous f32 data is reinterpreted as
/// bytes directly, without per-element conversion or intermediate copies.
pub fn save_safetensors(path: &Path, tensors: &[(String, Tensor)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let contig: Vec<_> = tensors.iter().map(|(_, t)| t.as_standard_layout()).collect();
    // Fallback byte copies (big-endian or non-contiguous); built before views
    // so the borrow checker sees stable storage.
    let mut byte_store: Vec<Option<Vec<u8>>> = vec![None; tensors.len()];
    for (i, t) in contig.iter().enumerate() {
        if !cfg!(target_endian = "little") || t.as_slice().is_none() {
            byte_store[i] = Some(t.iter().flat_map(|v| v.to_le_bytes()).collect());
        }
    }
    let mut views = Vec::new();
    for (i, (name, _)) in tensors.iter().enumerate() {
        let t = &contig[i];
        let bytes: &[u8] = match &byte_store[i] {
            Some(b) => b.as_slice(),
            None => {
                let s = t.as_slice().expect("contiguous");
                unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) }
            }
        };
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            t.shape().to_vec(),
            bytes,
        )?;
        views.push((name.as_str(), view));
    }
    let serialized = safetensors::serialize(views, &None)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

/// Read only shapes/dtypes from a safetensors file without materializing data.
pub fn safetensors_metadata(path: &Path) -> Result<Vec<(String, Vec<usize>, String)>> {
    let data = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&data)?;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            (
                name.to_string(),
                view.shape().to_vec(),
                format!("{:?}", view.dtype()).to_lowercase(),
            )
        })
        .collect())
}

pub fn tensors_to_map(tensors: Vec<(String, Tensor)>) -> HashMap<String, Tensor> {
    tensors.into_iter().collect()
}

#[allow(dead_code)]
fn _unused_dtype_helper(name: &str) -> Result<safetensors::Dtype> {
    dtype_to_safetensors(name)
}