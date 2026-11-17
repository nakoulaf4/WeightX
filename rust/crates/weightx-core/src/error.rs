// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX core error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightXError {
    #[error("Unknown tensor {0:?}")]
    UnknownTensor(String),
    #[error("Unknown checkpoint {0:?}; available={1:?}")]
    UnknownCheckpoint(String, Vec<String>),
    #[error("PlanValidationError: {0}")]
    PlanValidation(String),
    #[error("SandboxSecurityError: {0}")]
    SandboxSecurity(String),
    #[error("ValueError: {0}")]
    Value(String),
    #[error("KeyError: {0}")]
    Key(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    #[error("Provider error: {0}")]
    Provider(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, WeightXError>;

impl From<safetensors::SafeTensorError> for WeightXError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        WeightXError::SafeTensors(e.to_string())
    }
}

impl From<WeightXError> for String {
    fn from(e: WeightXError) -> String {
        e.to_string()
    }
}