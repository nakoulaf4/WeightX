# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""L3 Weight Reading, Slicing, Checkpointing, and Batch Associative Editing (EAMIT)."""

from .eamit import (
    BatchFactSpec,
    EamitAlignConfig,
    EamitAlignStats,
    EamitResult,
    align_residuals_to_keys,
    apply_closed_form_ridge,
    apply_eamit_batch,
)
from .workspace import (
    TensorDiffRecord,
    WeightWorkspace,
    WorkspaceDiffSummary,
)

__all__ = [
    "BatchFactSpec",
    "EamitAlignConfig",
    "EamitAlignStats",
    "EamitResult",
    "TensorDiffRecord",
    "WeightWorkspace",
    "WorkspaceDiffSummary",
    "align_residuals_to_keys",
    "apply_closed_form_ridge",
    "apply_eamit_batch",
]
