"""Degree-day snow accumulation and melt."""
from __future__ import annotations

import torch

Tensor = torch.Tensor


def degree_day_step(swe: Tensor, precip_mm: Tensor, tair_c: Tensor,
                    tsnow: Tensor, ddf: Tensor, dt_hours: float = 24.0):
    """One degree-day snow step.

    Precipitation falls as snow when ``tair_c < tsnow`` and melt is
    ``ddf * (tair_c - tsnow)+``, capped by the pack. Returns
    ``(liquid_input_mm, swe_new)`` where the liquid input is rain plus melt.

    The rain/snow split is a hard threshold, matching the reference
    implementation. It is subdifferentiable: ``tsnow`` still receives gradient
    through the melt term, which is where its identifiable signal lives. For a
    smooth split (useful when ``tsnow`` must be inferred from snow observations
    rather than discharge) see :func:`degree_day_step_soft`.
    """
    kd = dt_hours / 24.0
    cold = tair_c < tsnow
    snowfall = torch.where(cold, precip_mm, torch.zeros_like(precip_mm))
    rain = precip_mm - snowfall
    melt = torch.minimum(swe + snowfall,
                         ddf * kd * torch.clamp(tair_c - tsnow, min=0.0))
    swe = swe + snowfall - melt
    return rain + melt, swe


def degree_day_step_soft(swe: Tensor, precip_mm: Tensor, tair_c: Tensor,
                         tsnow: Tensor, ddf: Tensor, dt_hours: float = 24.0,
                         tau: float = 0.5):
    """As :func:`degree_day_step` but with a sigmoid rain/snow partition.

    ``tau`` is the transition width in degC. Gives ``tsnow`` a non-zero gradient
    through the partition itself; changes results, so it is not the default.
    """
    kd = dt_hours / 24.0
    fsnow = torch.sigmoid((tsnow - tair_c) / tau)
    snowfall = fsnow * precip_mm
    rain = precip_mm - snowfall
    melt = torch.minimum(swe + snowfall,
                         ddf * kd * torch.clamp(tair_c - tsnow, min=0.0))
    swe = swe + snowfall - melt
    return rain + melt, swe


__all__ = ["degree_day_step", "degree_day_step_soft"]
