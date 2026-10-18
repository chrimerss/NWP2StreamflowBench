"""A gradient-descent calibration driver.

Not a wrapper around ``Adam`` for its own sake: it packages the three things
that make basin-wise calibration of this model behave — per-unit best-iterate
tracking, phase restarts from the best iterate with decaying noise, and a
selection window that can be separated from the fitting window.

Per-unit best-iterate tracking matters because a single optimiser trajectory
serves thousands of units at once and they do not reach their best at the same
iteration. Keeping each unit's best iterate means the result can only improve on
any prefix of the schedule, so a longer run is never worse.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

from . import params as _params
from .model import CrestConfig, simulate

Tensor = torch.Tensor


@dataclass
class CalibrationSchedule:
    """Adam settings for a multi-phase run.

    Each phase restarts from the current per-unit best iterate, perturbed by
    ``jitter`` standard deviations in the unconstrained parameter space, and
    anneals the learning rate on a cosine to a tenth of its starting value. It
    is a basin-wise iterated local search that happens to use gradients.
    """

    iterations: int = 2048
    phases: int = 4
    lr: Sequence[float] = (0.05, 0.030, 0.020, 0.012)
    jitter: Sequence[float] = (0.0, 0.60, 0.40, 0.25)

    def phase_lr(self, i: int) -> float:
        return float(self.lr[i]) if i < len(self.lr) else float(self.lr[-1])

    def phase_jitter(self, i: int) -> float:
        return float(self.jitter[i]) if i < len(self.jitter) else float(self.jitter[-1])


@dataclass
class CalibrationResult:
    params: Dict[str, Tensor]
    raw: Dict[str, Tensor]
    score: Tensor                      #: per-unit best selection score
    history: List[dict] = field(default_factory=list)
    gradient_activity: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def calibrate(precip: Tensor, tair: Tensor, pet: Tensor, obs: Tensor,
              area_km2: Tensor, *,
              loss_fn: Callable[[Tensor, Tensor], tuple],
              score_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
              config: Optional[CrestConfig] = None,
              schedule: Optional[CalibrationSchedule] = None,
              bounds: Optional[Mapping[str, tuple]] = None,
              init: Optional[Mapping[str, float]] = None,
              seed: int = 0, init_jitter: float = 0.0,
              on_phase: Optional[Callable[[int, CalibrationResult], None]] = None,
              log_every: int = 0) -> CalibrationResult:
    """Calibrate one parameter set per unit.

    Parameters
    ----------
    loss_fn
        ``loss_fn(sim, obs) -> (per_unit_loss, valid_mask)``. Only valid units
        contribute to the mean that is back-propagated.
    score_fn
        ``score_fn(sim, obs) -> per_unit_score`` (higher is better) used for
        best-iterate tracking. Defaults to the negative of the loss, i.e.
        selection on the fitting window. Pass a function evaluated on a
        held-out window to select there instead.
    init_jitter
        Standard deviations of noise added to the initial raw parameters. Use a
        non-zero value with different ``seed`` values to build an ensemble.

    Returns
    -------
    :class:`CalibrationResult` holding the per-unit best parameters, both in
    physical and raw form, with the selection score and a per-phase history.
    """
    cfg = CrestConfig() if config is None else config
    sched = CalibrationSchedule() if schedule is None else schedule
    b = _params.BOUNDS if bounds is None else dict(bounds)
    device, dtype = precip.device, precip.dtype
    n = precip.shape[1]
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    raw = _params.default_raw(n, dtype=dtype, device=device, bounds=b,
                              values=init, requires_grad=False)
    if init_jitter:
        for k in raw:
            raw[k] = raw[k] + init_jitter * torch.randn(
                (n,), generator=gen, device=device, dtype=dtype)
    raw = {k: v.clone().requires_grad_(True) for k, v in raw.items()}
    area = torch.as_tensor(area_km2, dtype=dtype, device=device)

    best = torch.full((n,), -torch.inf, device=device, dtype=dtype)
    best_raw = {k: v.detach().clone() for k, v in raw.items()}
    activity = {k: 1.0 for k in raw}
    result = CalibrationResult(params={}, raw=best_raw, score=best)
    per_phase = max(1, sched.iterations // sched.phases)
    t0 = time.time()

    def physical(source):
        p = _params.to_physical(source, b)
        p["area_km2"] = area
        return p

    for phase in range(sched.phases):
        if phase:
            with torch.no_grad():
                jit = sched.phase_jitter(phase)
                for k in raw:
                    raw[k].copy_(best_raw[k] + jit * torch.randn(
                        (n,), generator=gen, device=device, dtype=dtype))
        opt = torch.optim.Adam(list(raw.values()), lr=sched.phase_lr(phase))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=per_phase, eta_min=sched.phase_lr(phase) * 0.1)
        for it in range(per_phase):
            opt.zero_grad(set_to_none=True)
            sim = simulate(precip, tair, pet, physical(raw), cfg, validate=False)
            per_unit, valid = loss_fn(sim, obs)
            loss = per_unit[valid].mean()
            loss.backward()
            if phase == 0 and it == 0:
                activity = _params.gradient_activity(raw)
                dead = [k for k, v in activity.items() if v > 0.99]
                if dead:
                    import warnings
                    warnings.warn(
                        f"parameters receive no gradient in every unit and will "
                        f"not be calibrated: {dead}. Check their bounds.",
                        RuntimeWarning, stacklevel=2)
            opt.step()
            sch.step()
            with torch.no_grad():
                s = (-per_unit.detach() if score_fn is None
                     else score_fn(sim.detach(), obs))
                better = (s > best) & valid
                for k in raw:
                    best_raw[k][better] = raw[k].detach()[better]
                best[better] = s[better]
            if log_every and (it % log_every == 0 or it == per_phase - 1):
                print(f"  phase {phase} iter {phase * per_phase + it:5d} "
                      f"loss {loss.item():+.4f} best {best[best > -torch.inf].median():+.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        result.history.append({"phase": phase,
                               "median_score": float(best[best > -torch.inf].median()),
                               "seconds": time.time() - t0})
        if on_phase is not None:
            result.params = physical(best_raw)
            on_phase(phase, result)

    result.params = {k: v.detach() for k, v in physical(best_raw).items()}
    result.raw = {k: v.detach() for k, v in best_raw.items()}
    result.score = best.detach()
    result.gradient_activity = activity
    result.seconds = time.time() - t0
    return result


def nse_loss(mask: Optional[Tensor] = None, var_floor: float = 0.10,
             min_obs: int = 730, transform: Optional[str] = None):
    """``mean_units (1 - NSE)`` — the objective the CAMELS benchmarks use.

    ``var_floor`` bounds the observed variance in the denominator so that a few
    very flat hydrographs cannot dominate the unit-averaged loss.
    ``transform`` of ``"sqrt"`` or ``"log"`` applies to both series first, which
    shifts weight from flood peaks towards low flows.
    """
    def _fn(sim, obs):
        s, o = _apply_transform(sim, obs, transform)
        m = torch.isfinite(o) & torch.isfinite(s)
        if mask is not None:
            m = m & (mask.unsqueeze(1).expand_as(o) if mask.ndim == 1 else mask)
        n = m.sum(0).clamp(min=1)
        om = torch.where(m, o, torch.zeros_like(o)).sum(0) / n
        oc = torch.where(m, o - om[None, :], torch.zeros_like(o))
        var = torch.clamp((oc ** 2).sum(0) / n, min=var_floor)
        resid = torch.where(m, s - o, torch.zeros_like(s))
        return ((resid ** 2).sum(0) / n) / var, m.sum(0) >= min_obs
    return _fn


def kge_loss(mask: Optional[Tensor] = None, bias_penalty: float = 2.0,
             min_obs: int = 730, transform: Optional[str] = None):
    """``mean_units (1 - KGE) + bias_penalty * (beta - 1)^2``.

    The extra bias term is not decoration: with ``lf`` free the unpenalised KGE
    optimum drifts to a low volume bias in a sizeable minority of catchments.
    """
    from .metrics import kge as _kge

    def _fn(sim, obs):
        s, o = _apply_transform(sim, obs, transform)
        m = torch.isfinite(o) & torch.isfinite(s)
        if mask is not None:
            m = m & (mask.unsqueeze(1).expand_as(o) if mask.ndim == 1 else mask)
        value, (_, _, beta) = _kge(s, o, mask, components=True)
        return (1 - value) + bias_penalty * (beta - 1) ** 2, m.sum(0) >= min_obs
    return _fn


def _apply_transform(sim, obs, transform):
    if transform is None:
        return sim, obs
    if transform == "sqrt":
        return torch.sqrt(torch.clamp(sim, min=0.0)), torch.sqrt(torch.clamp(obs, min=0.0))
    if transform == "log":
        eps = 0.01
        return torch.log(torch.clamp(sim, min=0.0) + eps), torch.log(torch.clamp(obs, min=0.0) + eps)
    raise ValueError(f"unknown transform {transform!r}")


__all__ = ["CalibrationSchedule", "CalibrationResult", "calibrate",
           "nse_loss", "kge_loss"]
