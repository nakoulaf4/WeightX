// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX agent layer: cersei-based LLM providers (OpenAI-compatible, Anthropic),
//! CLI-agent bridge, and the provider factory used by CLI/Desktop front-ends.

pub mod cersei_brain;
pub mod cli_agent;
pub mod factory;

pub use cersei_brain::CerseiBrainProvider;
pub use cli_agent::{CliAgentProvider, CLI_AGENT_PRESETS};
pub use factory::{agent_factory, build_provider};