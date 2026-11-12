# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""L3 Weight Reading, Slicing, Checkpointing, and Batch Associative Editing (MEMIT-MASS)."""

from .memit_mass import (
    BatchFactSpec,
    MemitMassResult,
    apply_closed_form_ridge,
    apply_memit_mass_batch,
)
from .workspace import (
    TensorDiffRecord,
    WeightWorkspace,
    WorkspaceDiffSummary,
)

__all__ = [
    "BatchFactSpec",
    "MemitMassResult",
    "TensorDiffRecord",
    "WeightWorkspace",
    "WorkspaceDiffSummary",
    "apply_closed_form_ridge",
    "apply_memit_mass_batch",
]
