// SPDX-License-Identifier: GPL-3.0-or-later
//! cersei-backed brain providers (OpenAI-compatible, Anthropic).
//!
//! Bridges the async `cersei` SDK to the synchronous [`BrainProvider`] contract
//! used by the WeightX harness.

use std::sync::Arc;

use cersei_provider::{Anthropic, CompletionRequest, OpenAi, Provider};
use cersei_types::{ContentBlock, Message, MessageContent, ToolDefinition, ToolResultContent};
use serde_json::Value;
use weightx_core::error::{Result, WeightXError};
use weightx_core::provider::{parse_embedded_tool_calls, BrainProvider, LLMMessage, LLMResponse, LLMToolCall};
use weightx_core::tools::ToolCatalog;

pub struct CerseiBrainProvider {
    inner: Arc<dyn Provider>,
    model: String,
    label: String,
    rt: Arc<tokio::runtime::Runtime>,
}

impl Clone for CerseiBrainProvider {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            model: self.model.clone(),
            label: self.label.clone(),
            rt: self.rt.clone(),
        }
    }
}

fn runtime() -> Result<Arc<tokio::runtime::Runtime>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .map(Arc::new)
        .map_err(|e| WeightXError::Value(format!("cannot create tokio runtime: {e}")))
}

impl CerseiBrainProvider {
    pub fn openai_compatible(
        label: &str,
        model: &str,
        base_url: &str,
        api_key: &str,
    ) -> Result<Self> {
        let provider = OpenAi::builder()
            .api_key(api_key)
            .base_url(base_url.trim_end_matches('/'))
            .model(model)
            .build()
            .map_err(|e| WeightXError::Value(format!("cersei OpenAI init failed: {e}")))?;
        Ok(Self {
            inner: Arc::new(provider),
            model: model.to_string(),
            label: label.to_string(),
            rt: runtime()?,
        })
    }

    pub fn for_openai(model: &str, api_key: Option<&str>) -> Result<Self> {
        let key = api_key
            .filter(|k| !k.is_empty())
            .map(String::from)
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_else(|| "local-no-key-required".to_string());
        let model = if model.is_empty() { "gpt-4o" } else { model };
        Self::openai_compatible("openai", model, "https://api.openai.com/v1", &key)
    }

    pub fn for_openrouter(model: &str, api_key: Option<&str>) -> Result<Self> {
        let key = api_key
            .filter(|k| !k.is_empty())
            .map(String::from)
            .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
            .unwrap_or_else(|| "local-no-key-required".to_string());
        let model = if model.is_empty() { "anthropic/claude-3.7-sonnet" } else { model };
        Self::openai_compatible("openrouter", model, "https://openrouter.ai/api/v1", &key)
    }

    pub fn for_ollama(model: &str, base_url: &str) -> Result<Self> {
        let model = if model.is_empty() { "qwen2.5-coder:14b" } else { model };
        let url = if base_url.is_empty() { "http://localhost:11434/v1" } else { base_url };
        Self::openai_compatible("ollama", model, url, "ollama-local")
    }

    pub fn for_lm_studio(model: &str, base_url: &str) -> Result<Self> {
        let model = if model.is_empty() { "local-model" } else { model };
        let url = if base_url.is_empty() { "http://localhost:1234/v1" } else { base_url };
        Self::openai_compatible("lm-studio", model, url, "lm-studio-local")
    }

    pub fn for_vllm(model: &str, base_url: &str) -> Result<Self> {
        let model = if model.is_empty() { "default" } else { model };
        let url = if base_url.is_empty() { "http://localhost:8000/v1" } else { base_url };
        Self::openai_compatible("vllm", model, url, "vllm-local")
    }

    pub fn for_anthropic(model: &str, api_key: Option<&str>) -> Result<Self> {
        let key = api_key
            .filter(|k| !k.is_empty())
            .map(String::from)
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .unwrap_or_default();
        let model = if model.is_empty() { "claude-3-7-sonnet-20250219" } else { model };
        let provider = Anthropic::builder()
            .api_key(key)
            .model(model)
            .build()
            .map_err(|e| WeightXError::Value(format!("cersei Anthropic init failed: {e}")))?;
        Ok(Self {
            inner: Arc::new(provider),
            model: model.to_string(),
            label: "anthropic".to_string(),
            rt: runtime()?,
        })
    }

    fn build_request(&self, messages: &[LLMMessage], tools: Option<&ToolCatalog>) -> CompletionRequest {
        let mut req = CompletionRequest::new(self.model.clone());
        req.max_tokens = 4096;
        req.temperature = Some(0.0);
        let mut system_parts: Vec<String> = Vec::new();
        let mut msgs: Vec<Message> = Vec::new();
        for m in messages {
            match m.role.as_str() {
                "system" => system_parts.push(m.content.clone()),
                "user" => msgs.push(Message::user(m.content.clone())),
                "assistant" => {
                    if m.tool_calls.is_empty() {
                        msgs.push(Message::assistant(m.content.clone()));
                    } else {
                        let mut blocks = Vec::new();
                        if !m.content.is_empty() {
                            blocks.push(ContentBlock::Text { text: m.content.clone() });
                        }
                        for tc in &m.tool_calls {
                            blocks.push(ContentBlock::ToolUse {
                                id: tc.call_id.clone(),
                                name: tc.name.clone(),
                                input: tc.arguments.clone(),
                            });
                        }
                        msgs.push(Message::assistant_blocks(blocks));
                    }
                }
                "tool" => {
                    msgs.push(Message::user_blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: m.tool_call_id.clone().unwrap_or_default(),
                        content: ToolResultContent::Text(m.content.clone()),
                        is_error: None,
                    }]));
                }
                _ => msgs.push(Message::user(m.content.clone())),
            }
        }
        if !system_parts.is_empty() {
            req.system = Some(system_parts.join("\n\n"));
        }
        req.messages = msgs;
        if let Some(cat) = tools {
            req.tools = cat
                .tools
                .iter()
                .map(|t| ToolDefinition {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.parameters.clone(),
                })
                .collect();
        }
        req
    }
}

impl BrainProvider for CerseiBrainProvider {
    fn provider_name(&self) -> String {
        self.label.clone()
    }
    fn model_name(&self) -> String {
        self.model.clone()
    }
    fn clone_box(&self) -> Box<dyn BrainProvider> {
        Box::new(self.clone())
    }

    fn generate(
        &mut self,
        messages: &[LLMMessage],
        tool_catalog: Option<&ToolCatalog>,
    ) -> Result<LLMResponse> {
        let req = self.build_request(messages, tool_catalog);
        let resp = self.rt.block_on(async {
            match tokio::time::timeout(
                std::time::Duration::from_secs(120),
                self.inner.complete_blocking(req),
            )
            .await
            {
                Ok(Ok(r)) => Ok(r),
                Ok(Err(e)) => Err(WeightXError::Value(format!("{} provider error: {e}", self.label))),
                Err(_) => Err(WeightXError::Value(format!("{} provider timeout after 120s", self.label))),
            }
        })?;

        let content = resp.message.get_all_text();
        let mut tool_calls: Vec<LLMToolCall> = Vec::new();
        if let MessageContent::Blocks(blocks) = &resp.message.content {
            for b in blocks {
                if let ContentBlock::ToolUse { id, name, input } = b {
                    tool_calls.push(LLMToolCall {
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments: input.clone(),
                    });
                }
            }
        }
        if tool_calls.is_empty() {
            tool_calls = parse_embedded_tool_calls(&content);
        }
        Ok(LLMResponse {
            content,
            tool_calls,
            input_tokens: resp.usage.input_tokens as usize,
            output_tokens: resp.usage.output_tokens as usize,
            provider_name: self.label.clone(),
            model_name: self.model.clone(),
        })
    }
}

/// Serialize a tool catalog as OpenAI function schemas (for diagnostics).
pub fn openai_tool_schemas(cat: &ToolCatalog) -> Value {
    Value::Array(cat.to_openai_tools())
}