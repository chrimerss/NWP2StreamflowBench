"""dcrest — a differentiable, GPU-parallel implementation of CREST.

CREST (Coupled Routing and Excess Storage) is the hydrologic model behind the
EF5 operational forecasting system. This package provides its water-balance
chain in PyTorch, so discharge is differentiable with respect to both the
parameters and the meteorological forcing.

Two things follow from that. Parameters can be calibrated by gradient descent
over thousands of catchments at once, and — because the forcing is on the same
footing — the model can be *inverted*: held fixed while the input is solved for.

Quick start
-----------
>>> import torch, dcrest
>>> params = dcrest.default_parameters(n_units=3, area_km2=[120., 800., 4300.])
>>> P = torch.rand(365, 3) * 10          # mm/day
>>> T = 10 + 8 * torch.randn(365, 3)     # degC
>>> E = torch.full((365, 3), 3.0)        # mm/day
>>> Q = dcrest.simulate(P, T, E, params)
>>> Q.shape
torch.Size([365, 3])

Calibration
-----------
>>> model = dcrest.CrestLumped(n_units=3, area_km2=[120., 800., 4300.])
>>> opt = torch.optim.Adam(model.parameters(), lr=0.05)
>>> loss = (1 - dcrest.metrics.kge(model(P, T, E), obs)).mean()  # doctest: +SKIP
>>> loss.backward(); opt.step()                                   # doctest: +SKIP
>>> model.gradient_activity()   # any value near 1.0 is a dead parameter

Scope
-----
Lumped: one modelling unit per tensor column, no lateral coupling, response
translated by a per-unit Gamma unit hydrograph. Cell-to-cell channel routing —
the "coupled routing" half of the full model — is not implemented yet.
"""
from . import calibrate as _calibrate_mod  # noqa: F401
from . import io, kernels, metrics, params, routing
from .calibrate import (CalibrationResult, CalibrationSchedule, calibrate,
                        kge_loss, nse_loss)
from .io import config_from_meta, load_parameters, save_parameters
from .model import (CrestConfig, CrestLumped, LEGACY_CONFIG, STATE_NAMES,
                    initial_states, simulate)
from .params import (BOUNDS, BOUNDS_TENSION, DEFAULTS, PARAM_NAMES,
                     gradient_activity)
from .routing import gamma_uh, geomorphic_lag_hours

__version__ = "1.0.0"


def default_parameters(n_units: int, area_km2, *, dtype=None, device="cpu",
                       values=None, bounds=None):
    """A physical parameter set at the default values — handy for smoke tests.

    Not a prior and not calibrated: every unit gets the same values.

    Emits exactly the parameters in ``bounds`` (the base 16 by default). Pass
    ``bounds=dcrest.BOUNDS_TENSION`` to include the optional Sacramento-style
    tension parameters; without that the model is the base one, so adding the
    extension can never change existing behaviour by accident.
    """
    import torch as _torch
    dtype = _torch.get_default_dtype() if dtype is None else dtype
    names = BOUNDS if bounds is None else bounds
    vals = dict(DEFAULTS)
    if values:
        vals.update(values)
    out = {k: _torch.full((n_units,), float(vals[k]), dtype=dtype, device=device)
           for k in names}
    out["area_km2"] = _torch.as_tensor(area_km2, dtype=dtype, device=device)
    return out


__all__ = [
    "BOUNDS", "BOUNDS_TENSION", "DEFAULTS", "PARAM_NAMES", "STATE_NAMES", "LEGACY_CONFIG",
    "CrestConfig", "CrestLumped", "simulate", "initial_states",
    "default_parameters", "gamma_uh", "geomorphic_lag_hours",
    "gradient_activity", "load_parameters", "save_parameters",
    "config_from_meta", "io", "kernels", "metrics", "params", "routing",
    "calibrate", "CalibrationSchedule", "CalibrationResult", "nse_loss",
    "kge_loss",
    "__version__",
]
