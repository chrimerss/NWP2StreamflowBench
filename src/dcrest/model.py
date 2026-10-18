"""The lumped CREST forward model.

Chain, per step, for every modelling unit in parallel::

    P, T_air  --degree-day snow-->  liquid input
              --interception (ft*wm, evaporates first)-->  input to soil
              --CRESTPhys (VIC curve, aquifer)-->  overland | interflow | baseflow
    overland  --linear reservoir (qc)-->  release
    (release + interflow + baseflow) * (1 - lf)
              --Gamma unit hydrograph (nshape, kmult * geomorphic scale)-->  Q

Everything is differentiable with respect to both the parameters and the
forcing, which is what makes the model usable as a decoder in inverse problems
as well as a forward simulator.

Shapes
------
Forcing is ``(T, N)``; parameters are ``(N,)``; discharge is ``(T, N)`` in mm
per step. ``N`` is the number of modelling units and they never interact, so
the model is parallel over ``N`` on the GPU and memory scales as ``T * N``.

Memory
------
Back-propagating through ``T`` steps stores every intermediate. Set
``CrestConfig.checkpoint_every`` to trade one extra forward pass for roughly a
``checkpoint_every / T`` reduction in activation memory; 96 steps is a good
default for multi-decade daily runs on a 12 GB card.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from . import params as _params
from .kernels import (crestphys_step, degree_day_step, interception_step,
                      linear_reservoir_step)
from .routing import convolve_uh, gamma_uh, geomorphic_lag_hours

Tensor = torch.Tensor

#: Names of the model's prognostic states, in the order they are packed.
STATE_NAMES: Tuple[str, ...] = ("swe", "canopy", "sm", "gw", "fast")


@dataclass(frozen=True)
class CrestConfig:
    """Everything about a run that is not a parameter or a forcing.

    ``uh_kmin_h``
        Lower clamp on the Gamma-UH scale, in hours. The default of 1 h keeps
        ``kmult`` identifiable at a daily step. Passing ``None`` reproduces the
        legacy ``dt_hours / 2`` clamp, under which ``kmult`` is frozen in almost
        every catchment — supported only for reproducing older results.
    ``sanitize``
        Replace non-finite intermediates and clamp states at zero each step.
        On by default: with pathological parameters the VIC curve can produce
        NaN, and without this a single bad unit poisons the whole batch's
        gradient. Turn it off to have such cases fail loudly.
    ``compile_step``
        Wrap the per-step function in ``torch.compile``. Worth it for long runs
        on CUDA (the model is kernel-launch bound, not compute bound); adds a
        one-off compilation cost of a few seconds.
    ``bounds``
        Parameter ranges to validate against. ``None`` means the package
        defaults. A released set calibrated under widened ranges carries its
        own, and :func:`~dcrest.io.config_from_meta` fills this in from the
        file — so a stored set is checked against the ranges it was actually
        fitted under instead of being rejected by the defaults.
    """

    dt_hours: float = 24.0
    uh_nstep: int = 30
    uh_kmin_h: Optional[float] = 1.0
    uh_method: str = "auto"
    checkpoint_every: Optional[int] = None
    sanitize: bool = True
    compile_step: bool = False
    bounds: Optional[Mapping[str, tuple]] = None

    def legacy(self) -> "CrestConfig":
        """The configuration of the first published calibration of this model."""
        return replace(self, uh_kmin_h=None, uh_method="loop")


#: Reproduces ``crest_blackbox.simulate`` from the amortized-inversion study.
LEGACY_CONFIG = CrestConfig(uh_kmin_h=None, uh_method="loop")


def initial_states(params: Mapping[str, Tensor], n_units: int, *,
                   dtype=None, device=None) -> Dict[str, Tensor]:
    """Default initial states: half-full soil, aquifer at 30% of capacity.

    These are spin-up seeds, not calibrated quantities; discard a warm-up period
    rather than tuning them.
    """
    wm = params["wm"]
    dtype = wm.dtype if dtype is None else dtype
    device = wm.device if device is None else device
    z = torch.zeros(n_units, dtype=dtype, device=device)
    return {"swe": z.clone(), "canopy": z.clone(),
            "sm": 0.5 * params["wm"], "gw": 0.3 * params["hmaxaq"],
            "fast": z.clone()}


def _make_step(cfg: CrestConfig, with_et: bool = False):
    def step(p, capacity, swe, wt, sm, gw, qs, P_t, TA_t, PET_t):
        liquid, swe = degree_day_step(swe, P_t, TA_t, p["tsnow"], p["ddf"],
                                      cfg.dt_hours)
        liquid, pet_soil, wt, et_canopy = interception_step(
            wt, liquid, PET_t, p["ke"], capacity)
        out = crestphys_step(p, sm, gw, liquid, pet_soil, cfg.dt_hours,
                             return_et=with_et)
        if with_et:
            fast, inter, base, sm, gw, et_soil = out
        else:
            fast, inter, base, sm, gw = out
        if cfg.sanitize:
            fast = torch.nan_to_num(fast)
            slow = torch.nan_to_num(inter + base)
        else:
            slow = inter + base
        release, qs = linear_reservoir_step(qs, fast, p["qc"], cfg.dt_hours)
        q = (release + slow) * (1.0 - p["lf"])
        if cfg.sanitize:
            sm = torch.clamp(torch.nan_to_num(sm), min=0.0)
            gw = torch.clamp(torch.nan_to_num(gw), min=0.0)
            wt = torch.clamp(torch.nan_to_num(wt), min=0.0)
        if with_et:
            return q, et_canopy + et_soil, swe, wt, sm, gw, qs
        return q, swe, wt, sm, gw, qs

    if cfg.compile_step:
        step = torch.compile(step, dynamic=False)
    return step


def simulate(precip: Tensor, tair: Tensor, pet: Tensor,
             params: Mapping[str, Tensor], config: Optional[CrestConfig] = None,
             *, states: Optional[Mapping[str, Tensor]] = None,
             return_states: bool = False, validate: bool = True,
             bounds: Optional[Mapping[str, tuple]] = None):
    """Run the model.

    Parameters
    ----------
    precip, tair, pet
        ``(T, N)`` tensors: precipitation [mm/step], air temperature [degC],
        potential ET [mm/step]. ``precip`` is differentiable, which is what
        allows the model to be inverted for forcing.
    params
        Mapping of the 16 parameters as ``(N,)`` tensors, plus ``area_km2``.
    config
        A :class:`CrestConfig`; defaults to a daily run with the recommended
        UH clamp.
    states
        Optional initial states; see :func:`initial_states`.
    return_states
        Also return the final states and the runoff before routing.

    Returns
    -------
    ``Q`` of shape ``(T, N)`` in mm per step, or ``(Q, aux)`` if
    ``return_states``.
    """
    cfg = CrestConfig() if config is None else config
    if precip.ndim != 2:
        raise ValueError(f"forcing must be (T, N); got {tuple(precip.shape)}")
    if tair.shape != precip.shape or pet.shape != precip.shape:
        raise ValueError("precip, tair and pet must have the same shape "
                         f"({tuple(precip.shape)}, {tuple(tair.shape)}, "
                         f"{tuple(pet.shape)})")
    if validate:
        _params.validate(params, bounds=bounds if bounds is not None
                         else cfg.bounds)
    T, N = precip.shape
    if params["wm"].shape[0] != N:
        raise ValueError(f"forcing has {N} units but parameters have "
                         f"{params['wm'].shape[0]}")

    st = initial_states(params, N) if states is None else dict(states)
    missing = [k for k in STATE_NAMES if k not in st]
    if missing:
        raise KeyError(f"missing initial states: {missing}")
    packed = tuple(st[k] for k in STATE_NAMES)
    capacity = params["ft"] * params["wm"]
    nout = 2 if return_states else 1
    step = _make_step(cfg, with_et=return_states)
    seg = cfg.checkpoint_every

    def run_segment(t0: int, t1: int, *state):
        acc = [[] for _ in range(nout)]
        s = state
        for t in range(t0, t1):
            res = step(params, capacity, *s, precip[t], tair[t], pet[t])
            for i in range(nout):
                acc[i].append(res[i])
            s = res[nout:]
        return tuple(torch.stack(a) for a in acc) + tuple(s)

    chunks = [[] for _ in range(nout)]
    if seg is None or seg >= T:
        res = run_segment(0, T, *packed)
    else:
        res = None
        for t0 in range(0, T, seg):
            res = _checkpoint(run_segment, t0, min(t0 + seg, T), *packed,
                              use_reentrant=False)
            for i in range(nout):
                chunks[i].append(res[i])
            packed = res[nout:]
    if seg is None or seg >= T:
        for i in range(nout):
            chunks[i].append(res[i])
        packed = res[nout:]
    stacked = [torch.cat(c) if len(c) > 1 else c[0] for c in chunks]
    runoff = stacked[0]

    lag = geomorphic_lag_hours(params["area_km2"], dtype=precip.dtype,
                               device=precip.device)
    weights = gamma_uh(params["nshape"], lag * params["kmult"],
                       nstep=cfg.uh_nstep, dt_hours=cfg.dt_hours,
                       kmin_h=cfg.uh_kmin_h)
    discharge = convolve_uh(runoff, weights, method=cfg.uh_method)

    if return_states:
        aux = dict(zip(STATE_NAMES, packed))
        aux["runoff"] = runoff
        aux["et"] = stacked[1]
        aux["uh"] = weights
        return discharge, aux
    return discharge


class CrestLumped(torch.nn.Module):
    """Calibration-ready wrapper holding one unconstrained parameter per unit.

    The module owns raw (logit-space) parameters, so any ``torch.optim``
    optimiser can be pointed straight at ``.parameters()`` and the physical
    values stay inside their bounds by construction.

    >>> model = CrestLumped(n_units=5, area_km2=area)
    >>> q = model(precip, tair, pet)
    >>> (1 - kge(q, obs)).mean().backward()
    """

    def __init__(self, n_units: int, area_km2, config: Optional[CrestConfig] = None,
                 *, bounds=None, init: Optional[Mapping[str, float]] = None,
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.config = CrestConfig() if config is None else config
        self.bounds = _params.BOUNDS if bounds is None else dict(bounds)
        raw = _params.default_raw(n_units, dtype=dtype, device=device,
                                  bounds=self.bounds, values=init,
                                  requires_grad=False)
        self.raw = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in raw.items()})
        self.register_buffer(
            "area_km2", torch.as_tensor(area_km2, dtype=dtype, device=device))

    @property
    def physical(self) -> Dict[str, Tensor]:
        """Current parameters in physical units, plus ``area_km2``."""
        p = _params.to_physical(dict(self.raw), self.bounds)
        p["area_km2"] = self.area_km2
        return p

    def gradient_activity(self, atol: float = 0.0) -> Dict[str, float]:
        """Fraction of units with zero gradient, per parameter (post-backward)."""
        return _params.gradient_activity(dict(self.raw), atol=atol)

    def forward(self, precip: Tensor, tair: Tensor, pet: Tensor, **kw):
        return simulate(precip, tair, pet, self.physical, self.config, **kw)


__all__ = ["CrestConfig", "LEGACY_CONFIG", "STATE_NAMES", "CrestLumped",
           "initial_states", "simulate"]
