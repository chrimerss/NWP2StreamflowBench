"""Gamma unit hydrograph and causal convolution."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def geomorphic_lag_hours(area_km2, *, dtype=torch.float32, device="cpu",
                         coeff: float = 0.35, exponent: float = 0.38) -> Tensor:
    """Prior mean response lag [hours] from drainage area [km2].

    ``coeff * (A/1000) ** exponent`` days, converted to hours — the classic
    area-to-time-of-concentration scaling. It exists so that ``kmult`` is a
    dimensionless multiplier on a physically sensible prior instead of a free
    time constant, which keeps the parameter comparable across catchment sizes.
    """
    if isinstance(area_km2, torch.Tensor):
        a = area_km2.to(dtype=dtype, device=device)
    else:
        a = torch.as_tensor(np.asarray(area_km2), dtype=dtype, device=device)
    return coeff * torch.pow(torch.clamp(a, min=1.0) / 1000.0, exponent) * 24.0


def gamma_uh(nshape: Tensor, kscale_h: Tensor, *, nstep: int = 30,
             dt_hours: float = 24.0, kmin_h: float | None = None,
             eps: float = 1e-6) -> Tensor:
    """Discrete Gamma unit hydrograph, shape ``(nstep, n_units)``, columns sum to 1.

    Evaluated at step midpoints and normalised, so truncation at ``nstep`` is
    mass-conserving by construction rather than by luck.

    ``kmin_h`` clamps the scale from below. Choose it well under the step
    length: a clamp at ``dt/2`` truncated the scale in 99.2% of CAMELS
    catchments and silenced the ``kmult`` gradient entirely. ``None`` means
    ``dt_hours / 2``, which reproduces that legacy behaviour and is *not*
    recommended for new calibration.
    """
    n = torch.clamp(nshape, min=0.5, max=12.0)
    kmin = dt_hours * 0.5 if kmin_h is None else kmin_h
    k = torch.clamp(kscale_h, min=kmin)
    t = (torch.arange(nstep, dtype=n.dtype, device=n.device) + 0.5) * dt_hours
    tt = t[:, None] / k[None, :]
    logw = (n[None, :] - 1.0) * torch.log(tt + eps) - tt - torch.lgamma(n[None, :])
    w = torch.exp(logw)
    return w / torch.clamp(w.sum(0, keepdim=True), min=eps)


def convolve_uh(q: Tensor, weights: Tensor, method: str = "auto") -> Tensor:
    """Causal per-unit convolution: ``out[t] = sum_l weights[l] * q[t - l]``.

    ``q`` is ``(T, n_units)``, ``weights`` is ``(L, n_units)``; the series is
    treated as starting from rest, so the first ``L`` steps are ramp-up and
    should be inside the spin-up window.

    ``method``
        ``"conv"`` uses a grouped 1-D convolution — one pass, fast on GPU.
        ``"loop"`` accumulates lag by lag; slower, and the summation order
        matches the original reference implementation exactly.
        ``"auto"`` picks ``"conv"``, falling back to ``"loop"`` for dtypes
        ``conv1d`` does not support.
    """
    if q.ndim != 2 or weights.ndim != 2:
        raise ValueError(f"expected 2-D (T, N) and (L, N), got {tuple(q.shape)} "
                         f"and {tuple(weights.shape)}")
    if q.shape[1] != weights.shape[1]:
        raise ValueError(f"unit-count mismatch: {q.shape[1]} vs {weights.shape[1]}")
    if method == "auto":
        method = "loop" if q.dtype not in (torch.float32, torch.float64) else "conv"

    if method == "loop":
        T = q.shape[0]
        out = torch.zeros_like(q)
        for lag in range(weights.shape[0]):
            out = out + F.pad(q[: T - lag] * weights[lag][None, :],
                              (0, 0, lag, 0))
        return out

    if method != "conv":
        raise ValueError(f"unknown method {method!r}")

    T, N = q.shape
    L = weights.shape[0]
    # (1, N, T+L-1) signal, (N, 1, L) time-reversed per-unit kernels, groups=N
    x = F.pad(q.transpose(0, 1).unsqueeze(0), (L - 1, 0))
    k = weights.flip(0).transpose(0, 1).unsqueeze(1)
    return F.conv1d(x, k, groups=N).squeeze(0).transpose(0, 1)


__all__ = ["gamma_uh", "geomorphic_lag_hours", "convolve_uh"]
