"""Interception store: fills before the soil sees water, evaporates first."""
from __future__ import annotations

import torch

Tensor = torch.Tensor


def interception_step(wt: Tensor, liquid_mm: Tensor, pet_mm: Tensor,
                      ke: Tensor, capacity_mm: Tensor):
    """One interception step.

    ``capacity_mm`` is ``ft * wm``. The store fills from the liquid input, then
    the atmospheric demand ``ke * pet`` is applied to the store first; whatever
    demand is left is passed to the soil column.

    Returns ``(liquid_to_soil_mm, pet_to_soil_mm, wt_new, et_canopy_mm)``.

    Note the unit of the returned demand: it is divided by ``ke`` because the
    soil kernel re-applies ``ke`` internally. This keeps the total atmospheric
    demand consistent across the two stores and matches the reference chain.
    """
    fill = torch.minimum(liquid_mm, torch.clamp(capacity_mm - wt, min=0.0))
    wt = wt + fill
    liquid = liquid_mm - fill
    demand = ke * pet_mm
    et_canopy = torch.minimum(demand, wt)
    wt = torch.minimum(torch.clamp(wt - et_canopy, min=0.0), capacity_mm)
    pet_soil = torch.clamp(demand - et_canopy, min=0.0) / torch.clamp(ke, min=1e-3)
    return liquid, pet_soil, wt, et_canopy


__all__ = ["interception_step"]
