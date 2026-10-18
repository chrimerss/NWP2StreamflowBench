"""Evaluation metrics, differentiable where it matters.

The torch versions are written for calibration: they take a boolean mask, treat
missing observations correctly, and reduce over time to a per-unit vector so a
loss can be a plain mean over units. The numpy versions are for reporting.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


def _masked_moments(sim: Tensor, obs: Tensor, mask: Tensor, eps: float):
    n = mask.sum(0).clamp(min=1)
    sim_m = torch.where(mask, sim, torch.zeros_like(sim)).sum(0) / n
    obs_m = torch.where(mask, obs, torch.zeros_like(obs)).sum(0) / n
    sc = torch.where(mask, sim - sim_m[None, :], torch.zeros_like(sim))
    oc = torch.where(mask, obs - obs_m[None, :], torch.zeros_like(obs))
    ss = torch.sqrt((sc ** 2).sum(0) / n + eps)
    os_ = torch.sqrt((oc ** 2).sum(0) / n + eps)
    return n, sim_m, obs_m, sc, oc, ss, os_


def _build_mask(sim: Tensor, obs: Tensor, mask: Optional[Tensor]) -> Tensor:
    m = torch.isfinite(obs) & torch.isfinite(sim)
    if mask is not None:
        m = m & (mask.unsqueeze(1).expand_as(obs) if mask.ndim == 1 else mask)
    return m


def nse(sim: Tensor, obs: Tensor, mask: Optional[Tensor] = None,
        eps: float = 1e-8, var_floor: float = 0.0) -> Tensor:
    """Nash-Sutcliffe efficiency per unit.

    ``var_floor`` bounds the observed variance in the denominator from below.
    It has no effect on the reported metric when left at zero; set it during
    calibration (e.g. 0.1 mm^2/day^2) so that a handful of very flat-hydrograph
    units cannot dominate a basin-averaged loss.
    """
    m = _build_mask(sim, obs, mask)
    n, _, _, _, oc, _, _ = _masked_moments(sim, obs, m, eps)
    denom = (oc ** 2).sum(0) / n
    if var_floor > 0.0:
        denom = torch.clamp(denom, min=var_floor)
    resid = torch.where(m, sim - obs, torch.zeros_like(sim))
    return 1.0 - ((resid ** 2).sum(0) / n) / (denom + eps)


def kge(sim: Tensor, obs: Tensor, mask: Optional[Tensor] = None,
        eps: float = 1e-8, components: bool = False):
    """Kling-Gupta efficiency (Gupta et al., 2009) per unit.

    With ``components=True`` also returns ``(r, alpha, beta)``.
    """
    m = _build_mask(sim, obs, mask)
    n, sim_m, obs_m, sc, oc, ss, os_ = _masked_moments(sim, obs, m, eps)
    r = ((sc * oc).sum(0) / n) / (ss * os_ + eps)
    alpha = ss / (os_ + eps)
    beta = sim_m / (obs_m + eps)
    value = 1.0 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + eps)
    return (value, (r, alpha, beta)) if components else value


def valid_count(obs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Number of usable observations per unit."""
    m = _build_mask(obs, obs, mask)
    return m.sum(0)


def summary(sim, obs, mask=None, min_days: int = 365) -> dict:
    """Per-unit numpy metrics for reporting: nse, kge, r, alpha, beta, n.

    Units with fewer than ``min_days`` observations return NaN rather than a
    number computed from too little data.
    """
    if isinstance(sim, torch.Tensor):
        sim = sim.detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    m = np.isfinite(obs) & np.isfinite(sim)
    if mask is not None:
        m &= mask[:, None] if mask.ndim == 1 else mask
    n = m.sum(0)
    nz = np.maximum(n, 1)
    sim_m = np.where(m, sim, 0.0).sum(0) / nz
    obs_m = np.where(m, obs, 0.0).sum(0) / nz
    sc = np.where(m, sim - sim_m, 0.0)
    oc = np.where(m, obs - obs_m, 0.0)
    ss = np.sqrt((sc ** 2).sum(0) / nz)
    os_ = np.sqrt((oc ** 2).sum(0) / nz)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = ((sc * oc).sum(0) / nz) / (ss * os_)
        alpha = ss / os_
        beta = sim_m / obs_m
        nse_v = 1.0 - (np.where(m, sim - obs, 0.0) ** 2).sum(0) / (oc ** 2).sum(0)
        kge_v = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    ok = n >= min_days
    out = {k: np.where(ok, v, np.nan) for k, v in
           [("nse", nse_v), ("kge", kge_v), ("r", r), ("alpha", alpha),
            ("beta", beta)]}
    out["n"] = n.astype(float)
    return out


__all__ = ["nse", "kge", "valid_count", "summary"]
