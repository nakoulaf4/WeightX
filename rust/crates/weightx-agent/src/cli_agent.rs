// SPDX-License-Identifier: GPL-3.0-or-later
//! CLI Agent Provider Adapter — Rust port of `weightx.providers.cli_agent`.
//!
//! Bridges external CLI agents (Claude Code, Codex, OpenCode, Antigravity, DSH)
//! to WeightX tools. The agent runs in an empty ephemeral working directory and
//! interacts with the model solely via `<weightx_tool_call>` blocks.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use weightx_core::error::{Result, WeightXError};
use weightx_core::provider::{parse_embedded_tool_calls, BrainProvider, LLMMessage, LLMResponse};
use weightx_core::tools::ToolCatalog;
use weightx_core::toon;

pub const CLI_AGENT_PRESETS: &[(&str, &[&str])] = &[
    ("claude-code", &["claude", "-p", "{prompt}"]),
    ("codex", &["codex", "exec", "{prompt}"]),
    ("opencode", &["opencode", "run", "{prompt}"]),
    ("antigravity", &["agy", "prompt", "{prompt}"]),
    ("dsh", &["dsh", "-c", "{prompt}"]),
];

/// `shutil.which`-style lookup (honours PATHEXT on Windows).
pub fn which(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var("PATH").ok()?;
    let sep = if cfg!(windows) { ';' } else { ':' };
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
            .split(';')
            .map(|s| s.to_string())
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in path_var.split(sep) {
        if dir.is_empty() {
            continue;
        }
        if cfg!(windows) {
            for ext in &exts {
                let p = PathBuf::from(dir).join(format!("{binary}{ext}"));
                if p.is_file() {
                    return Some(p);
                }
            }
        } else {
            let p = PathBuf::from(dir).join(binary);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

#[derive(Clone)]
pub struct CliAgentProvider {
    pub agent_type: String,
    pub command_template: Vec<String>,
    pub timeout_sec: f64,
}

impl CliAgentProvider {
    pub fn new(agent_type: &str) -> Result<Self> {
        let preset = CLI_AGENT_PRESETS
            .iter()
            .find(|(name, _)| *name == agent_type)
            .ok_or_else(|| {
                WeightXError::Value(format!(
                    "Unknown CLI agent preset {agent_type:?}. Choose from {:?}",
                    CLI_AGENT_PRESETS.iter().map(|(n, _)| *n).collect::<Vec<_>>()
                ))
            })?;
        Ok(Self {
            agent_type: agent_type.to_string(),
            command_template: preset.1.iter().map(|s| s.to_string()).collect(),
            timeout_sec: 120.0,
        })
    }

    pub fn with_template(agent_type: &str, template: Vec<String>, timeout_sec: f64) -> Self {
        Self {
            agent_type: agent_type.to_string(),
            command_template: template,
            timeout_sec,
        }
    }

    /// Preset name -> resolved binary path (None if not installed).
    pub fn detect_installed_agents() -> HashMap<String, Option<String>> {
        let mut found = HashMap::new();
        for (preset, cmd) in CLI_AGENT_PRESETS {
            found.insert(
                preset.to_string(),
                which(cmd[0]).map(|p| p.display().to_string()),
            );
        }
        found
    }

    fn build_structured_prompt(&self, messages: &[LLMMessage], tools: Option<&ToolCatalog>) -> String {
        let mut sections: Vec<String> = vec![
            "You are connected to the WeightX Weight-Space Programming Harness.".into(),
            "To invoke a WeightX tool, output one or more blocks in the exact format:".into(),
            "<weightx_tool_call>{\"name\": \"<tool_name>\", \"arguments\": {...}}</weightx_tool_call>".into(),
        ];
        if let Some(cat) = tools {
            sections.push("\nAvailable WeightX Tools (TOON):".into());
            sections.push(cat.to_prompt_catalog());
        }
        sections.push("\nConversation Transcript:".into());
        for m in messages {
            if m.role == "tool" {
                sections.push(format!(
                    "[TOOL_RESULT id={}]\n{}",
                    m.tool_call_id.clone().unwrap_or_default(),
                    m.content
                ));
            } else {
                sections.push(format!("[{}]\n{}", m.role.to_uppercase(), m.content));
            }
        }
        sections.join("\n\n")
    }
}

impl BrainProvider for CliAgentProvider {
    fn provider_name(&self) -> String {
        format!("cli:{}", self.agent_type)
    }
    fn model_name(&self) -> String {
        self.agent_type.clone()
    }
    fn clone_box(&self) -> Box<dyn BrainProvider> {
        Box::new(self.clone())
    }

    fn generate(
        &mut self,
        messages: &[LLMMessage],
        tool_catalog: Option<&ToolCatalog>,
    ) -> Result<LLMResponse> {
        let full_prompt = self.build_structured_prompt(messages, tool_catalog);
        let cmd: Vec<String> = self
            .command_template
            .iter()
            .map(|p| p.replace("{prompt}", &full_prompt))
            .collect();
        if cmd.is_empty() {
            return Err(WeightXError::Value("empty CLI agent command template".into()));
        }

        let tmp = tempfile::tempdir()
            .map_err(|e| WeightXError::Value(format!("cannot create agent temp dir: {e}")))?;
        let t0 = Instant::now();
        let mut child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .current_dir(tmp.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| WeightXError::Value(format!("failed to spawn CLI agent {:?}: {e}", cmd[0])))?;

        let timeout = Duration::from_secs_f64(self.timeout_sec.max(1.0));
        let mut timed_out = false;
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if t0.elapsed() >= timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        timed_out = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => break,
            }
        }
        let out = child
            .wait_with_output()
            .map_err(|e| WeightXError::Value(format!("CLI agent wait failed: {e}")))?;
        if timed_out {
            return Err(WeightXError::Value(format!(
                "CLI agent {:?} timed out after {}s",
                self.agent_type, self.timeout_sec
            )));
        }
        let output_text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !out.status.success() && output_text.is_empty() {
            return Err(WeightXError::Value(format!(
                "CLI agent {:?} exited with code {:?}: {}",
                self.agent_type,
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        let tool_calls = parse_embedded_tool_calls(&output_text);
        Ok(LLMResponse {
            input_tokens: toon::estimate_tokens(&full_prompt),
            output_tokens: toon::estimate_tokens(&output_text),
            content: output_text,
            tool_calls,
            provider_name: self.provider_name(),
            model_name: self.model_name(),
        })
    }
}