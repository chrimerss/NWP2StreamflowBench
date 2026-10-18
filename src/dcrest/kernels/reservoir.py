"""Linear reservoir for the fast-response pathway."""
from __future__ import annotations

import torch

Tensor = torch.Tensor


def linear_reservoir_step(storage: Tensor, inflow_mm: Tensor, rate: Tensor,
                          dt_hours: float = 24.0):
    """Add ``inflow``, release ``rate`` of the storage. Returns ``(release, storage)``.

    ``rate`` is per day and is scaled to the step. It is clamped below 1 so the
    reservoir cannot release more than it holds at long steps.
    """
    kd = dt_hours / 24.0
    storage = storage + inflow_mm
    release = torch.clamp(rate * kd, max=1.0) * storage
    return release, storage - release


__all__ = ["linear_reservoir_step"]
