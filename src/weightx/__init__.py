# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX: Open-Source (GPLv3) Weight-Space Programming Harness.
"""

from .catalog import (
    CODEX_PROMPT_SUGGESTIONS,
    ModelCatalog,
    ModelCatalogEntry,
    PromptSuggestion,
)
from .declarative import (
    PlanExecutionReport,
    PlanValidationError,
    compile_and_validate_plan,
    execute_yaml_plan,
)
from .desktop_installer import install_desktop_shortcut
from .execution import (
    EvaluationSplit,
    IsolatedSandbox,
    SandboxExecutionResult,
    SandboxSecurityError,
    VespasianEngine,
    VespasianIterationResult,
)
from .harness import HarnessRunResult, WeightXHarness
from .improver import (
    ModelImprovementReport,
    PretrainedImprovementReport,
    improve_pretrained_lm,
    improve_small_model,
)
from .studio import StudioConfig, StudioSession
from .l1_arch import (
    ArchitectureReport,
    AutoArchEngine,
    CalibrationResolution,
    inspect_architecture,
    resolve_ambiguities_with_calibration,
)
from .l2_profile import (
    KnowledgeMapReport,
    UniLayerProfileReport,
    compute_gini_kci,
    profile_model_layers,
    run_knowledge_map,
    run_logit_lens,
)
from .l3_weights import (
    BatchFactSpec,
    MemitMassResult,
    WeightWorkspace,
    WorkspaceDiffSummary,
    apply_closed_form_ridge,
    apply_memit_mass_batch,
)
from .providers import (
    AnthropicProvider,
    AutonomousWSPProvider,
    CLIAgentProvider,
    CustomCallableProvider,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    OpenAICompatibleProvider,
)
from .safety import FisherGuard, FisherProtectionReport, compute_empirical_fisher
from .toon_io import TokenSavingsReport, compare_toon_vs_json, dumps_toon, loads_toon
from .tools import ToolDefinition, ToolRegistry, build_default_tool_registry, weightx_tool

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "ArchitectureReport",
    "AutoArchEngine",
    "AutonomousWSPProvider",
    "BatchFactSpec",
    "CODEX_PROMPT_SUGGESTIONS",
    "CLIAgentProvider",
    "CalibrationResolution",
    "CustomCallableProvider",
    "EvaluationSplit",
    "FisherGuard",
    "FisherProtectionReport",
    "HarnessRunResult",
    "IsolatedSandbox",
    "KnowledgeMapReport",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMToolCall",
    "MemitMassResult",
    "ModelCatalog",
    "ModelCatalogEntry",
    "ModelImprovementReport",
    "OpenAICompatibleProvider",
    "PlanExecutionReport",
    "PlanValidationError",
    "PretrainedImprovementReport",
    "PromptSuggestion",
    "SandboxExecutionResult",
    "SandboxSecurityError",
    "StudioConfig",
    "StudioSession",
    "TokenSavingsReport",
    "ToolDefinition",
    "ToolRegistry",
    "UniLayerProfileReport",
    "VespasianEngine",
    "VespasianIterationResult",
    "WeightWorkspace",
    "WeightXHarness",
    "WorkspaceDiffSummary",
    "apply_closed_form_ridge",
    "apply_memit_mass_batch",
    "build_default_tool_registry",
    "compare_toon_vs_json",
    "compile_and_validate_plan",
    "compute_empirical_fisher",
    "compute_gini_kci",
    "dumps_toon",
    "execute_yaml_plan",
    "improve_pretrained_lm",
    "improve_small_model",
    "inspect_architecture",
    "install_desktop_shortcut",
    "loads_toon",
    "profile_model_layers",
    "resolve_ambiguities_with_calibration",
    "run_knowledge_map",
    "run_logit_lens",
    "weightx_tool",
]
