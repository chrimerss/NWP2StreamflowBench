"""Parameter specification for the lumped CREST chain.

Sixteen parameters per modelling unit (a basin in lumped mode, a grid cell in
distributed mode). Each carries a physical range used both to bound calibration
and to document the model; ``to_raw``/``to_physical`` move between the physical
value and the unconstrained space in which gradient descent actually runs.

The ranges below are the *release* ranges. Two of them differ from the values
used in the first calibration of this model and the difference mattered:

``fc``   was (0, 150) mm/h. At that scale ``temX = (SM+Wo)/2/WM * fc * dt``
         exceeded the available runoff on essentially every day, so
         ``min(R, temX)`` never bound, ``fc`` received exactly zero gradient,
         and it stayed at its initial value in 100% of catchments. The range is
         now (0, 5) mm/h, where the term is active.
``kmult`` interacts with ``uh_kmin_h``: a lower clamp of dt/2 = 12 h truncated
         the Gamma-UH scale in 99.2% of catchments, again zeroing the gradient.
         Use ``uh_kmin_h=1.0`` (the default in :class:`~dcrest.model.CrestConfig`
         when ``dt_hours`` is 24) unless reproducing the older set.

Checking that every parameter actually receives gradient is cheap here and
impossible with derivative-free calibration; see :func:`gradient_activity`.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

Tensor = torch.Tensor

#: Physical bounds, in EF5-compatible units. Order is the canonical order.
BOUNDS: Dict[str, Tuple[float, float]] = {
    # --- degree-day snow -------------------------------------------------
    "tsnow":  (-2.0, 3.0),      # rain/snow air-temperature threshold [degC]
    "ddf":    (1.0, 8.0),       # degree-day melt factor [mm/degC/day]
    # --- interception ----------------------------------------------------
    "ft":     (0.0, 0.7),       # canopy capacity as a fraction of wm [-]
    # --- CRESTPhys soil column ------------------------------------------
    "wm":     (1.0, 2000.0),    # mean soil water capacity [mm]
    "b":      (0.01, 6.0),      # VIC infiltration-curve exponent [-]
    "im":     (0.0, 0.5),       # impervious area fraction [-]
    "ke":     (0.2, 2.5),       # PET multiplier [-]
    "fc":     (0.0, 5.0),       # interflow rate [mm/h]
    "ksoil":  (0.0, 0.5),       # soil -> aquifer drainage rate [1/day]
    # --- aquifer ---------------------------------------------------------
    "hmaxaq": (50.0, 4000.0),   # aquifer capacity [mm]
    "gwc":    (0.01, 5.0),      # baseflow coefficient [mm/day]
    "gwe":    (0.5, 6.0),       # baseflow exponent [-]
    # --- response --------------------------------------------------------
    "qc":     (0.05, 1.0),      # fast linear-reservoir release rate [1/day]
    "lf":     (0.0, 0.5),       # deep loss / abstraction fraction [-]
    "nshape": (1.0, 8.0),       # Gamma unit-hydrograph shape [-]
    "kmult":  (0.1, 4.0),       # multiplier on the geomorphic UH scale [-]
}

#: Optional Sacramento-style tension-water extension. ``ptens`` protects a
#: fraction of soil storage from drainage (it can still evaporate); ``plp``
#: holds evapotranspiration at the full atmospheric demand until storage falls
#: below that fraction, as HBV's ``lp`` does. The defaults below are the
#: identity: ``ptens = 0`` and ``plp = 1`` reproduce the 16-parameter model
#: exactly, so a calibration on these bounds strictly contains the base model.
BOUNDS_TENSION: Dict[str, Tuple[float, float]] = dict(
    BOUNDS, ptens=(0.0, 0.9), plp=(0.05, 1.0))

#: Legacy ranges of the first published calibration. Kept so that older
#: parameter files can be re-expanded; do not use for new calibration.
LEGACY_BOUNDS: Dict[str, Tuple[float, float]] = dict(BOUNDS, fc=(0.0, 150.0))

PARAM_NAMES: Tuple[str, ...] = tuple(BOUNDS)

#: Starting values for calibration; chosen to be inside the active region of
#: every term (see the module docstring).
DEFAULTS: Dict[str, float] = {
    "tsnow": 0.5, "ddf": 3.0, "ft": 0.1, "wm": 200.0, "b": 1.0, "im": 0.05,
    "ke": 1.0, "fc": 0.5, "ksoil": 0.05, "hmaxaq": 500.0, "gwc": 0.5,
    "gwe": 3.0, "qc": 0.3, "lf": 0.05, "nshape": 2.0, "kmult": 0.6,
    "ptens": 0.2, "plp": 0.6,
}

UNITS: Dict[str, str] = {
    "tsnow": "degC", "ddf": "mm/degC/day", "ft": "-", "wm": "mm", "b": "-",
    "im": "-", "ke": "-", "fc": "mm/h", "ksoil": "1/day", "hmaxaq": "mm",
    "gwc": "mm/day", "gwe": "-", "qc": "1/day", "lf": "-", "nshape": "-",
    "kmult": "-", "ptens": "-", "plp": "-",
}


def _bounds(bounds: Mapping[str, Tuple[float, float]] | None):
    return BOUNDS if bounds is None else bounds


def to_raw(physical: Mapping[str, Tensor], bounds=None) -> Dict[str, Tensor]:
    """Physical values -> unconstrained logit space (the calibration space)."""
    b = _bounds(bounds)
    out = {}
    for k, v in physical.items():
        if k not in b:
            continue
        lo, hi = b[k]
        v = torch.as_tensor(v)
        u = ((v - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
        out[k] = torch.log(u / (1 - u))
    return out


def to_physical(raw: Mapping[str, Tensor], bounds=None) -> Dict[str, Tensor]:
    """Unconstrained logit space -> bounded physical values."""
    b = _bounds(bounds)
    return {k: b[k][0] + (b[k][1] - b[k][0]) * torch.sigmoid(r)
            for k, r in raw.items() if k in b}


def default_raw(n_units: int, *, dtype=torch.float32, device="cpu",
                bounds=None, values: Mapping[str, float] | None = None,
                requires_grad: bool = True) -> Dict[str, Tensor]:
    """A fresh set of raw parameters, one row per modelling unit."""
    b = _bounds(bounds)
    vals = dict(DEFAULTS)
    if values:
        vals.update(values)
    out = {}
    for k in b:
        lo, hi = b[k]
        u = float(np.clip((vals[k] - lo) / (hi - lo), 1e-4, 1 - 1e-4))
        t = torch.full((n_units,), float(np.log(u / (1 - u))),
                       dtype=dtype, device=device)
        out[k] = t.requires_grad_(requires_grad)
    return out


def validate(params: Mapping[str, Tensor], *, bounds=None,
             require_area: bool = True) -> None:
    """Raise if a parameter is missing, mis-shaped, non-finite or out of range.

    Extension parameters present in params but not in bounds (e.g.
    ptens/plp when validating against the base 16) are checked against
    :data:`BOUNDS_TENSION` so a tension-variant set validates out of the box.
    """
    b = dict(_bounds(bounds))
    for k in ("ptens", "plp"):
        if k in params and k not in b:
            b[k] = BOUNDS_TENSION[k]
    missing = [k for k in b if k not in params and k in _bounds(bounds)]
    if missing:
        raise KeyError(f"missing parameters: {missing}")
    shapes = {tuple(torch.as_tensor(params[k]).shape) for k in b}
    if len(shapes) != 1:
        raise ValueError(f"parameters have inconsistent shapes: {shapes}")
    if require_area and "area_km2" not in params:
        raise KeyError("missing 'area_km2' (needed for the geomorphic UH scale)")
    for k in b:
        v = torch.as_tensor(params[k]).detach()
        if not torch.isfinite(v).all():
            raise ValueError(f"parameter '{k}' contains non-finite values")
        lo, hi = b[k]
        tol = 1e-4 * (hi - lo)
        if float(v.min()) < lo - tol or float(v.max()) > hi + tol:
            raise ValueError(
                f"parameter '{k}' outside bounds [{lo}, {hi}]: "
                f"[{float(v.min())}, {float(v.max())}]")


def gradient_activity(raw: Mapping[str, Tensor], *, atol: float = 0.0
                      ) -> Dict[str, float]:
    """Fraction of units whose gradient is (numerically) zero, per parameter.

    Call after ``loss.backward()``. A value near 1 means the parameter is not
    being calibrated at all — usually because a ``min``/``clamp`` upstream never
    binds under the current bounds. This diagnostic is the reason to prefer a
    differentiable model even when the optimiser itself is not the point.
    """
    out = {}
    for k, t in raw.items():
        if t.grad is None:
            out[k] = 1.0
        else:
            out[k] = float((t.grad.abs() <= atol).float().mean())
    return out


def describe() -> str:
    """A printable table of the parameter specification."""
    w = max(len(k) for k in BOUNDS)
    lines = [f"{'name'.ljust(w)}  {'units':<12} {'low':>10} {'high':>10} "
             f"{'default':>10}", "-" * (w + 48)]
    for k, (lo, hi) in BOUNDS.items():
        lines.append(f"{k.ljust(w)}  {UNITS[k]:<12} {lo:>10.4g} {hi:>10.4g} "
                     f"{DEFAULTS[k]:>10.4g}")
    return "\n".join(lines)


__all__ = ["BOUNDS", "BOUNDS_TENSION", "LEGACY_BOUNDS", "PARAM_NAMES", "DEFAULTS", "UNITS",
           "to_raw", "to_physical", "default_raw", "validate",
           "gradient_activity", "describe"]
