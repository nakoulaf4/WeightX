# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Safety Mechanisms: IWCKEdit Fisher Information Protection against Catastrophic Over-editing."""

from .iwck_fisher import (
    FisherGuard,
    FisherProtectionReport,
    compute_empirical_fisher,
)

__all__ = [
    "FisherGuard",
    "FisherProtectionReport",
    "compute_empirical_fisher",
]
