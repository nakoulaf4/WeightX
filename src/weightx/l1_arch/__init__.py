# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""L1 Architecture Understanding Layer (AutoArch)."""

from .autoarch import (
    ArchitectureReport,
    AutoArchEngine,
    CalibrationResolution,
    TensorRoleInfo,
    inspect_architecture,
    resolve_ambiguities_with_calibration,
)

__all__ = [
    "ArchitectureReport",
    "AutoArchEngine",
    "CalibrationResolution",
    "TensorRoleInfo",
    "inspect_architecture",
    "resolve_ambiguities_with_calibration",
]
