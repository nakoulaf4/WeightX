// SPDX-License-Identifier: GPL-3.0-or-later
//! Provider factory shared by the CLI and Desktop front-ends.

use std::sync::Arc;

use weightx_core::error::{Result, WeightXError};
use weightx_core::provider::{AutonomousWSPProvider, BrainProvider};
use weightx_core::studio::{ProviderFactory, ProviderSpec};

use crate::cersei_brain::CerseiBrainProvider;
use crate::cli_agent::CliAgentProvider;

/// Instantiate a brain provider from UI/CLI selections (port of `studio.build_provider`).
pub fn build_provider(spec: &ProviderSpec) -> Result<Box<dyn BrainProvider>> {
    let kind = spec.kind.split(' ').next().unwrap_or("autonomous");
    let model = spec.model.as_str();
    let base_url = spec.base_url.as_str();
    let api_key = if spec.api_key.is_empty() { None } else { Some(spec.api_key.as_str()) };
    match kind {
        "autonomous" => Ok(Box::new(AutonomousWSPProvider::default())),
        "ollama" => Ok(Box::new(CerseiBrainProvider::for_ollama(model, base_url)?)),
        "lm-studio" => Ok(Box::new(CerseiBrainProvider::for_lm_studio(model, base_url)?)),
        "openai" => Ok(Box::new(CerseiBrainProvider::for_openai(model, api_key)?)),
        "openrouter" => Ok(Box::new(CerseiBrainProvider::for_openrouter(model, api_key)?)),
        "vllm" => Ok(Box::new(CerseiBrainProvider::for_vllm(model, base_url)?)),
        "anthropic" => Ok(Box::new(CerseiBrainProvider::for_anthropic(model, api_key)?)),
        k if k.starts_with("cli:") => Ok(Box::new(CliAgentProvider::new(&k[4..])?)),
        _ => Err(WeightXError::Value(format!("Unknown provider kind {kind:?}"))),
    }
}

/// Factory for [`weightx_core::studio::StudioSession::factory`].
pub fn agent_factory() -> ProviderFactory {
    Arc::new(build_provider)
}